#!/usr/bin/env python3
"""
train.py
Neu implementiertes Trainingsskript für alle Varianten (v1–v4) mit Stable Baselines3.
"""
import argparse
import yaml
import json
from datetime import datetime
from pathlib import Path
import importlib
import shutil
import logging
logging.getLogger("pyboy.core.sound").setLevel(logging.ERROR)


import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from poke_pipeline.tensorboard_callback import TensorboardCallback
from poke_pipeline.callbacks import StatsCallback
from poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD, MultiInputLstmPolicyLD


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Pokémon Red environments")
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v3", "v4"],
        required=True,
        help="v1: Baseline (RedGymEnv+StreamWrapper), v2/v3: RedGymEnv, v4: RedGymEnvLSTM",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pfad zur YAML-Konfig, z.B. configs/v1.yaml",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def make_run_dirs(base: Path) -> dict:
    dirs = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "tensorboard": base / "tensorboard",
        "json_logs": base / "json_logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def make_env_fn(variant: str, module_name: str, class_name: str, env_conf: dict, rank: int, seed: int):
    def _init():
        # Pro Worker eigene Kopie der env_conf erstellen
        worker_env_conf = env_conf.copy()
        worker_env_conf["worker_rank"] = rank
        worker_env_conf["num_cpu"] = env_conf.get("num_cpu", 1)
        
        module = importlib.import_module(f"poke_pipeline.{module_name}")
        EnvCls = getattr(module, class_name)
        env = EnvCls(worker_env_conf)
        """if variant == "v1":
            from poke_pipeline.stream_agent_wrapper import StreamWrapper
            env = StreamWrapper(env, stream_metadata={
                "user": "v1-default",
                "env_id": rank,
                "color": "#447799",
                "extra": "",
            })"""
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 1) Run-Ordner anlegen
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_root = Path(cfg["paths"]["session_root"]) / now
    dirs = make_run_dirs(session_root)

    # config speichern
    dest = dirs["root"] / args.config.name
    shutil.copyfile(args.config, dest)
    
    # Effective config für Debugging speichern
    effective_config = {
        "num_cpu": cfg.get("num_cpu", 1),
        "max_steps": cfg["env"]["max_steps"],
        "n_steps": cfg["model"]["n_steps"],
        "save_freq": cfg.get("save_freq", 10000),
        "save_freq_stats": cfg.get("save_freq_stats", 100),
        "reset_interval": cfg["env"]["max_steps"] // cfg["model"]["n_steps"],
        "total_timesteps": cfg.get("total_timesteps", 1e6),
        "variant": args.variant,
        "config_file": str(args.config)
    }
    with open(dirs["root"] / "effective_config.json", "w") as f:
        json.dump(effective_config, f, indent=2)
    print(f"Effective config: {effective_config}")
    print(f"Environment resets every {effective_config['reset_interval']} iterations")

    # 2) Environment config
    env_conf = cfg["env"].copy()
    env_conf["session_path"] = dirs["root"]
    if "init_state" in env_conf and env_conf["init_state"].strip():  # Nur resolve wenn nicht leer
        env_conf["init_state"] = str(Path(env_conf["init_state"]).resolve())
    # Falls init_state leer ist, bleibt es leer (für No-State-Loading Tests)

    # 3) Vectorized environments
    num_cpu = cfg.get("num_cpu", 1)
    module_name = cfg["env"]["module"]
    class_name = cfg["env"]["class"]
    
    # num_cpu zur env_conf hinzufügen
    env_conf["num_cpu"] = num_cpu
    
    env_fns = [make_env_fn(args.variant, module_name, class_name, env_conf, i, cfg.get("seed", 0))
               for i in range(num_cpu)]
    if num_cpu > 1:
        vec_env = SubprocVecEnv(env_fns)
        print(f"Using SubprocVecEnv with {num_cpu} parallel workers")
    else:
        vec_env = DummyVecEnv(env_fns)
        print(f"Using DummyVecEnv with {num_cpu} worker")

    # 4) Modell instanziieren
    model_cfg = cfg["model"]
    model_type = model_cfg["type"]
    policy_key = model_cfg["policy"]

    if model_type == "PPO":
        ModelClass = PPO
    elif model_type == "RecurrentPPO":
        ModelClass = RecurrentPPO
        # für RecurrentPPO immer die sb3_contrib‐Policy‐Klasse
        policy_key = MultiInputLstmPolicy  # :contentReference[oaicite:0]{index=0}
    elif model_type == "RecurrentPPOLD":
        ModelClass = RecurrentPPOLD
        # für unsere LD‐Variante die selbstdefinierte Policy‐Klasse
        policy_key = MultiInputLstmPolicyLD  # :contentReference[oaicite:1]{index=1}
    else:
        raise ValueError(f"Unbekannter model.type: {model_type}")

    # Grundlegende kwargs
    base_kwargs = {
        "policy": policy_key,
        "env": vec_env,
        "tensorboard_log": str(dirs["tensorboard"]),
        # seed und verbose können auch hier aufgenommen werden
    }

    # Parse total_timesteps robust gegen Strings wie '1e8'
    raw_ts = cfg.get("total_timesteps", 1e6)
    if isinstance(raw_ts, str):
        try:
            raw_ts = float(raw_ts)
        except ValueError:
            raise ValueError(f"total_timesteps must be numeric, got {raw_ts}")
    total_timesteps = int(raw_ts)

    # Erlaubte zusätzliche Hyperparameter
    for key in [
        "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
        "gae_lambda", "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
        "max_grad_norm", "seed", "verbose", "device", "ld_coef"
    ]:
        if key in model_cfg:
            base_kwargs[key] = model_cfg[key]

    model = ModelClass(**base_kwargs)

    # 5) Callbacks
    cb_list = []
    cb_list.append(
        CheckpointCallback(
            save_freq=int(cfg.get("save_freq", 10000)),
            save_path=str(dirs["checkpoints"]),
            name_prefix=args.variant
        )
    )
    cb_list.append(TensorboardCallback(str(dirs["tensorboard"])))
    
    # StatsCallback nur hinzufügen wenn save_freq_stats > 0
    stats_freq = int(cfg.get("save_freq_stats", 100))
    if stats_freq > 0:
        cb_list.append(
            StatsCallback(
                save_freq=stats_freq,
                save_path=str(dirs["json_logs"]),
                verbose=1
            )
        )

    # 6) Training
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(cb_list),
            tb_log_name=args.variant
        )
        # Finaler Checkpoint nach erfolgreichem Training
        final_checkpoint = dirs["checkpoints"] / f"{args.variant}_final_model.zip"
        model.save(str(final_checkpoint))
        print(f"Final model saved to {final_checkpoint}")
    except KeyboardInterrupt:
        print(" Training interrupted – finalisiere JSON-Logs …")
        # Emergency checkpoint bei Unterbrechung
        emergency_checkpoint = dirs["checkpoints"] / f"{args.variant}_emergency_model.zip"
        model.save(str(emergency_checkpoint))
        print(f"Emergency model saved to {emergency_checkpoint}")
        for cb in cb_list:
            if hasattr(cb, "_on_training_end"):
                cb._on_training_end()
        raise


if __name__ == "__main__":
    main()