# File: Pipeline/src/train.py

import argparse
import yaml
from datetime import datetime
from pathlib import Path
import importlib

import numpy as np
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
from callbacks import StatsCallback
from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPOLD

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=["v2", "v3", "v4"],
        required=True,
        help="Welche Variante: v2/v3 uses RedGymEnv, v4 uses RedGymEnvLSTM",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pfad zur YAML‑Config (z.B. configs/v2.yaml)",
    )
    return parser.parse_args()

def load_config(path: Path) -> dict:
    with path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg

def make_run_dirs(base: Path) -> dict:
    """
    Legt an:
      base/                  (config.paths.session_root)
      base/checkpoints/
      base/tensorboard/
      base/json_logs/
    """
    dirs = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "tensorboard": base / "tensorboard",
        "json_logs": base / "json_logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def make_env_fn(env_mod_name: str, env_class_name: str, env_conf: dict, rank: int, seed: int):
    """
    Factory für SubprocVecEnv – lädt Modul und Klasse dynamisch.
    """
    def _init():
        # Modul aus src/ laden
        mod = importlib.import_module(f"src.{env_mod_name}")
        EnvCls = getattr(mod, env_class_name)
        env = EnvCls(env_conf)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # --- 1) Run‑Ordner anlegen ---
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # falls in YAML ein Datum drinsteht, könntet ihr das stattdessen nehmen:
    session_root = Path(cfg["paths"]["session_root"]) / now
    dirs = make_run_dirs(session_root)

    # --- 2) Environment‑Config ---
    env_conf = cfg["env"].copy()
    # ggf. hier noch absolute Pfade fixen, z.B. init_state = data/…
    env_conf["session_path"] = dirs["root"]

    # --- 3) Vektor‑Wrapper erstellen (Env‑Modul/-Klasse aus Config) ---
    num_cpu = cfg.get("num_cpu", 1)
    env_mod   = cfg["env"]["module"]
    env_cls   = cfg["env"]["class"]
    env_fns = [
        make_env_fn(env_mod, env_cls, env_conf, rank=i, seed=cfg.get("seed", 0))
        for i in range(num_cpu)
    ]
    if num_cpu > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        from stable_baselines3.common.vec_env import DummyVecEnv
        vec_env = DummyVecEnv(env_fns)

    # --- 4) Model instanziieren ---
    model_cfg = cfg["model"]
    if model_cfg["type"] == "PPO":
        ModelClass = PPO
    elif model_cfg["type"] == "RecurrentPPO":
        ModelClass = RecurrentPPO
    elif model_cfg["type"] == "RecurrentPPOLD":
        ModelClass = RecurrentPPOLD
    else:
        raise ValueError(f"Unbekannter model.type: {model_cfg['type']}")

    # Extrahiere alle Hyperparameter außer type und policy
    hyperparams = {k: v for k, v in model_cfg.items()
                   if k not in ("type", "policy")}

    model = ModelClass(
        policy=model_cfg["policy"],
        env=vec_env,
        tensorboard_log=str(dirs["tensorboard"]),
        **hyperparams
    )

    # --- 5) Callbacks vorbereiten ---
    cb_list = []
    # Checkpoints
    cb_list.append(
        CheckpointCallback(
            save_freq=int(cfg["save_freq"]),
            save_path=str(dirs["checkpoints"]),
            name_prefix=args.variant
        )
    )
    # TensorBoard‑Callback
    cb_list.append(TensorboardCallback(str(dirs["tensorboard"])))
    # Stats‑Callback (JSON‑Logs)
    cb_list.append(
        StatsCallback(
            save_freq=int(cfg["save_freq_stats"]),
            save_path=str(dirs["json_logs"]),
            verbose=1
        )
    )

    # --- 6) Training ---
    try:
        model.learn(
            total_timesteps=int(cfg["total_timesteps"]),
            callback=CallbackList(cb_list),
            tb_log_name=args.variant
        )
    except KeyboardInterrupt:
        print("🏁 Training interrupted – letzter JSON‑Flush …")
        for cb in cb_list:
            if hasattr(cb, "_on_training_end"):
                cb._on_training_end()
        raise

if __name__ == "__main__":
    main()
