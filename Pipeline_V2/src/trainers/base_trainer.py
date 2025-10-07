from __future__ import annotations

import importlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy

from pipeline_v2.callbacks import StatsCallback
from pipeline_v2.ppo_lambda_discrepancy import MultiInputLstmPolicyLD, RecurrentPPOLD
from pipeline_v2.tensorboard_callback import TensorboardCallback


@dataclass
class TrainerArgs:
    variant: str
    config_path: Path
    resume_path: Optional[str] = None


class BaseTrainer:
    """Shared training workflow for all variants."""

    MODEL_REGISTRY = {
        "PPO": (PPO, None),
        "RecurrentPPO": (RecurrentPPO, MultiInputLstmPolicy),
        "RecurrentPPOLD": (RecurrentPPOLD, MultiInputLstmPolicyLD),
    }

    def __init__(self, args: TrainerArgs, cfg: Dict[str, Any]) -> None:
        self.args = args
        self.cfg = cfg

        self.session_root = self._determine_session_root()
        self.dirs = self._make_run_dirs(self.session_root)
        self._save_effective_config()
        self.env_conf = self._prepare_env_config()

        self.vec_env = self._build_vec_env()
        self.model, self.completed_steps = self._build_model()
        self.callbacks = self._build_callbacks()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> None:
        total_timesteps = self._parse_total_timesteps()
        remaining_steps = total_timesteps - self.completed_steps
        if remaining_steps <= 0:
            print("Training already completed!")
            return

        try:
            self.model.learn(
                total_timesteps=remaining_steps,
                callback=CallbackList(self.callbacks),
                tb_log_name=self.args.variant,
                reset_num_timesteps=False,
            )
            self._save_final_checkpoint()
        except KeyboardInterrupt:
            print("Training interrupted - saving emergency checkpoint ...")
            self._save_emergency_checkpoint()
            for cb in self.callbacks:
                if hasattr(cb, "_on_training_end"):
                    cb._on_training_end()
            raise

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _determine_session_root(self) -> Path:
        if self.args.resume_path:
            checkpoint_path = Path(self.args.resume_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
            if checkpoint_path.parent.name == "checkpoints":
                return checkpoint_path.parent.parent
            return checkpoint_path.parent
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(self.cfg["paths"]["session_root"]) / now

    def _make_run_dirs(self, base: Path) -> Dict[str, Path]:
        dirs = {
            "root": base,
            "checkpoints": base / "checkpoints",
            "tensorboard": base / "tensorboard",
            "json_logs": base / "json_logs",
        }
        for d in dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        return dirs

    def _save_effective_config(self) -> None:
        effective_config = {
            "num_cpu": self.cfg.get("num_cpu", 1),
            "max_steps": self.cfg["env"]["max_steps"],
            "n_steps": self.cfg["model"]["n_steps"],
            "save_freq": self.cfg.get("save_freq", 10000),
            "save_freq_stats": self.cfg.get("save_freq_stats", 100),
            "reset_interval": self.cfg["env"]["max_steps"] // self.cfg["model"]["n_steps"],
            "total_timesteps": self.cfg.get("total_timesteps", 1e6),
            "variant": self.args.variant,
            "config_file": str(self.args.config_path),
        }
        with open(self.dirs["root"] / "effective_config.json", "w") as f:
            json.dump(effective_config, f, indent=2)

        if not self.args.resume_path:
            dest = self.dirs["root"] / self.args.config_path.name
            shutil.copyfile(self.args.config_path, dest)

    def _prepare_env_config(self) -> Dict[str, Any]:
        env_conf = self.cfg["env"].copy()
        env_conf["session_path"] = self.dirs["root"]
        if env_conf.get("init_state", "").strip():
            env_conf["init_state"] = str(Path(env_conf["init_state"]).resolve())
        env_conf["num_cpu"] = self.cfg.get("num_cpu", 1)
        return env_conf

    def _make_env_fns(self) -> List[Any]:
        module_name = self.cfg["env"]["module"]
        class_name = self.cfg["env"]["class"]
        num_cpu = self.cfg.get("num_cpu", 1)
        seed = self.cfg.get("seed", 0)

        def make_env(rank: int):
            def _init():
                worker_conf = self.env_conf.copy()
                worker_conf["worker_rank"] = rank
                module = importlib.import_module(f"environments.{module_name}")
                EnvCls = getattr(module, class_name)
                env = EnvCls(worker_conf)
                env.reset(seed=seed + rank)
                return env
            return _init

        return [make_env(i) for i in range(num_cpu)]

    def _build_vec_env(self):
        env_fns = self._make_env_fns()
        num_cpu = self.cfg.get("num_cpu", 1)
        if num_cpu > 1:
            if self.env_conf.get("debug_reset_timing"):
                from utils.debug_vec_env import DebugSubprocVecEnv
                print(f"Using DebugSubprocVecEnv with {num_cpu} parallel workers")
                return DebugSubprocVecEnv(env_fns)
            else:
                print(f"Using SubprocVecEnv with {num_cpu} parallel workers")
                return SubprocVecEnv(env_fns)
        print("Using DummyVecEnv with 1 worker")
        return DummyVecEnv(env_fns)

    def _resolve_model_entry(self, model_type: str) -> Tuple[Any, Optional[Any]]:
        if model_type not in self.MODEL_REGISTRY:
            raise ValueError(f"Unbekannter model.type: {model_type}")
        return self.MODEL_REGISTRY[model_type]

    def _build_model(self):
        model_cfg = self.cfg["model"]
        model_type = model_cfg["type"]
        ModelClass, default_policy = self._resolve_model_entry(model_type)
        policy_key = model_cfg.get("policy", default_policy)

        if isinstance(policy_key, str) and default_policy is not None:
            if policy_key == default_policy.__name__:
                policy_key = default_policy
        base_kwargs = {
            "policy": policy_key,
            "env": self.vec_env,
            "tensorboard_log": str(self.dirs["tensorboard"]),
        }
        for key in [
            "learning_rate",
            "n_steps",
            "batch_size",
            "n_epochs",
            "gamma",
            "gae_lambda",
            "clip_range",
            "clip_range_vf",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "seed",
            "verbose",
            "device",
            "ld_coef",
        ]:
            if key in model_cfg:
                base_kwargs[key] = model_cfg[key]

        if self.args.resume_path:
            model = ModelClass.load(self.args.resume_path, env=self.vec_env)
            completed_steps = model.num_timesteps
            print(f"Loading model from {self.args.resume_path}")
            print(f"Model has completed {completed_steps:,} steps")
            remaining = self._parse_total_timesteps() - completed_steps
            print(f"Will train for {remaining:,} more steps")
            return model, completed_steps

        print(f"Instantiating model: {ModelClass.__name__}")
        model = ModelClass(**base_kwargs)
        return model, 0

    def _build_callbacks(self) -> List[Any]:
        callbacks = []
        save_freq = int(self.cfg.get("save_freq", 10000))
        callbacks.append(
            CheckpointCallback(
                save_freq=save_freq,
                save_path=str(self.dirs["checkpoints"]),
                name_prefix=self.args.variant,
            )
        )
        callbacks.append(TensorboardCallback(str(self.dirs["tensorboard"])))

        stats_freq = int(self.cfg.get("save_freq_stats", 100))
        if stats_freq > 0:
            callbacks.append(
                StatsCallback(
                    save_freq=stats_freq,
                    save_path=str(self.dirs["json_logs"]),
                    verbose=1,
                )
            )
        return callbacks

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _parse_total_timesteps(self) -> int:
        raw_ts = self.cfg.get("total_timesteps", 1e6)
        if isinstance(raw_ts, str):
            try:
                raw_ts = float(raw_ts)
            except ValueError:
                raise ValueError(f"total_timesteps must be numeric, got {raw_ts}")
        return int(raw_ts)

    def _save_final_checkpoint(self) -> None:
        final_checkpoint = self.dirs["checkpoints"] / f"{self.args.variant}_final_model.zip"
        self.model.save(str(final_checkpoint))
        print(f"Final model saved to {final_checkpoint}")

    def _save_emergency_checkpoint(self) -> None:
        emergency_checkpoint = self.dirs["checkpoints"] / f"{self.args.variant}_emergency_model.zip"
        self.model.save(str(emergency_checkpoint))
        print(f"Emergency model saved to {emergency_checkpoint}")

    @staticmethod
    def load_config(path: Path) -> Dict[str, Any]:
        import yaml

        with path.open() as f:
            return yaml.safe_load(f)
