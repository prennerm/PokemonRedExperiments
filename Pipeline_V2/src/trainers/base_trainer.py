from __future__ import annotations

import importlib
import json
import shutil
import os
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

from callbacks import StatsCallback, TensorboardCallback
from models import MultiInputLstmPolicyLD, RecurrentPPOLD
from utils import PackedSubprocVecEnv, SharedMemoryVecEnv


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

        # Apply Peter Whidden's n_steps formula: n_steps = max_steps // num_cpu
        # This ensures exactly 1 reset per rollout buffer
        self._apply_n_steps_formula()

        self.session_root = self._determine_session_root()
        self.dirs = self._make_run_dirs(self.session_root)
        self.logging_cfg = self._resolve_logging_config()
        self.env_conf = self._prepare_env_config()
        self._save_effective_config()

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
        finally:
            try:
                self.vec_env.close()
            except Exception as exc:
                if self.cfg.get("verbose", 0) > 0:
                    print(f"[BaseTrainer] VecEnv close raised {exc.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _apply_n_steps_formula(self) -> None:
        """
        Apply Peter Whidden's n_steps formula: n_steps = max_steps // num_cpu

        This ensures exactly 1 environment reset per rollout buffer, maintaining
        the original training rhythm from baseline_fast_v2.py.

        If n_steps is already specified in config, it will be overridden with a warning.
        """
        num_cpu = self.cfg.get("num_cpu", 1)
        max_steps = self.cfg.get("env", {}).get("max_steps")

        if max_steps is None:
            raise ValueError("env.max_steps must be specified in config for n_steps calculation")

        calculated_n_steps = max_steps // num_cpu
        config_n_steps = self.cfg.get("model", {}).get("n_steps")

        if config_n_steps is not None and config_n_steps != calculated_n_steps:
            print(f"WARNING: Overriding config n_steps={config_n_steps} with calculated n_steps={calculated_n_steps}")
            print(f"    Formula: max_steps ({max_steps}) // num_cpu ({num_cpu}) = {calculated_n_steps}")
            print(f"    This ensures 1 reset per rollout (Peter Whidden's baseline_fast_v2.py)")

        # Set the calculated value
        if "model" not in self.cfg:
            self.cfg["model"] = {}
        self.cfg["model"]["n_steps"] = calculated_n_steps

        if self.cfg.get("verbose", 0) > 0:
            resets_per_rollout = (calculated_n_steps * num_cpu) / max_steps
            print(f"n_steps set to {calculated_n_steps} (resets per rollout: {resets_per_rollout:.2f})")

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
            "logs": base / "logs",
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
            "logging_format": self.logging_cfg["format"],
            "logging_save_freq": self.logging_cfg["save_freq"],
            "reset_interval": self.cfg["env"]["max_steps"] // self.cfg["model"]["n_steps"],
            "total_timesteps": self.cfg.get("total_timesteps", 1e6),
            "variant": self.args.variant,
            "config_file": str(self.args.config_path),
            "send_map_to_agent": bool(self.env_conf.get("send_map_to_agent", True)),
            "pack_bits": bool(self.env_conf.get("pack_bits", True)),
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

    def _resolve_logging_config(self) -> Dict[str, Any]:
        defaults = {
            "format": "json",
            "save_freq": 100,
            "structured": True,
            "verbose": 0,
        }
        user_cfg = self.cfg.get("logging") or {}
        resolved = {**defaults, **user_cfg}
        resolved["format"] = str(resolved["format"]).lower()
        resolved["save_freq"] = max(1, int(resolved.get("save_freq", defaults["save_freq"])))
        resolved["structured"] = bool(resolved.get("structured", True))
        resolved["verbose"] = int(resolved.get("verbose", 0))
        return resolved

    def _make_env_fns(self) -> List[Any]:
        module_name = self.cfg["env"]["module"]
        class_name = self.cfg["env"]["class"]
        num_cpu = self.cfg.get("num_cpu", 1)
        seed = self.cfg.get("seed", 0)
        pack_bits = bool(self.env_conf.get("pack_bits", True)) and num_cpu > 1

        def make_env(rank: int):
            def _init():
                worker_conf = self.env_conf.copy()
                worker_conf["worker_rank"] = rank
                worker_conf["pack_bits"] = pack_bits
                module = importlib.import_module(f"environments.{module_name}")
                EnvCls = getattr(module, class_name)
                env = EnvCls(worker_conf)
                env.reset(seed=seed + rank)
                return env
            return _init

        return [make_env(i) for i in range(num_cpu)]

    def _build_vec_env(self):
        num_cpu = self.cfg.get("num_cpu", 1)
        use_shared_memory = bool(self.env_conf.get("use_shared_memory", True))
        if num_cpu <= 1:
            self.env_conf["pack_bits"] = False
            self.env_conf["use_shared_memory"] = False
            use_shared_memory = False
        elif use_shared_memory:
            # Shared memory transport does not need bit packing.
            self.env_conf["pack_bits"] = False
        env_fns = self._make_env_fns()
        if num_cpu > 1:
            start_method = os.environ.get("PIPELINE_SUBPROC_START_METHOD")
            start_method = start_method.strip() or None if start_method else None
            if self.env_conf.get("debug_reset_timing"):
                from utils.debug_vec_env import DebugSubprocVecEnv
                builder = DebugSubprocVecEnv
                print(f"Using DebugSubprocVecEnv with {num_cpu} parallel workers")
            else:
                use_packed = bool(self.env_conf.get("pack_bits", True))
                if use_shared_memory:
                    builder = SharedMemoryVecEnv
                    print(f"Using SharedMemoryVecEnv with {num_cpu} parallel workers (shared memory)")
                elif use_packed:
                    builder = PackedSubprocVecEnv
                    print(f"Using PackedSubprocVecEnv with {num_cpu} parallel workers (bit-packed observations)")
                else:
                    builder = SubprocVecEnv
                    print(f"Using SubprocVecEnv with {num_cpu} parallel workers")

            attempts = []
            if start_method:
                attempts.append(start_method)
            attempts.append(None)
            if "spawn" not in attempts:
                attempts.append("spawn")

            last_error: Optional[Exception] = None
            for method in attempts:
                try:
                    return builder(env_fns, start_method=method)
                except (PermissionError, OSError) as exc:
                    method_label = method or "default"
                    print(f"[BaseTrainer] VecEnv init failed with start method '{method_label}': {exc}")
                    last_error = exc
            raise RuntimeError("Unable to initialize SubprocVecEnv with any start method") from last_error

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

        logging_cfg = self.logging_cfg
        stats_freq = logging_cfg["save_freq"]
        if stats_freq > 0:
            callbacks.append(
                StatsCallback(
                    save_path=self.dirs["logs"],
                    save_freq=stats_freq,
                    output_format=logging_cfg["format"],
                    structured=logging_cfg["structured"],
                    verbose=logging_cfg["verbose"],
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

        def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(result.get(key), dict):
                    result[key] = _deep_update(result[key], value)
                else:
                    result[key] = value
            return result

        with path.open() as f:
            cfg = yaml.safe_load(f)
        extends = cfg.pop('extends', None)
        if extends:
            base_path = path.parent / extends
            base_cfg = BaseTrainer.load_config(base_path)
            cfg = _deep_update(base_cfg, cfg)
        return cfg
