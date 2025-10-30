#!/usr/bin/env python3
"""
train.py
Neu implementiertes Trainingsskript fÃ¼r alle Varianten (v1â€“v4) mit Stable Baselines3.
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
    parser = argparse.ArgumentParser(description="Train RL agent on PokÃ©mon Red environments")
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
    parser.add_argument(
        "--resume",
        type=str,
        help="Pfad zum Checkpoint (.zip) um Training fortzusetzen",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    import yaml

    def _deep_update(base: dict, override: dict) -> dict:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = _deep_update(base[key], value)
            else:
                base[key] = value
        return base

    def _load(current_path: Path) -> dict:
        with current_path.open() as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {current_path} must define a mapping")
        extends = data.pop("extends", None)
        if extends:
            if isinstance(extends, str):
                extends_list = [extends]
            else:
                extends_list = list(extends)
            merged: dict = {}
            for entry in extends_list:
                base_path = (current_path.parent / entry).resolve()
                merged = _deep_update(merged, _load(base_path))
            return _deep_update(merged, data)
        return data

    return _load(path.resolve())


def make_run_dirs(base: Path) -> dict:
    logs_dir = base / "logs"
    dirs = {
        "root": base,
        "checkpoints": base / "checkpoints",
        "tensorboard": base / "tensorboard",
        "logs": logs_dir,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    dirs["json_logs"] = logs_dir  # Backwards-compatible alias for legacy tooling
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


def apply_n_steps_rule(cfg: dict) -> None:
    """Setzt model.n_steps auf max_steps // num_cpu (mindestens 1)."""
    num_cpu = max(1, int(cfg.get("num_cpu", 1)))
    max_steps = cfg.get("env", {}).get("max_steps")
    if max_steps is None:
        raise ValueError("env.max_steps muss gesetzt sein, um n_steps berechnen zu kÃ¶nnen")
    calculated = max(1, max_steps // num_cpu)
    model_cfg = cfg.setdefault("model", {})
    configured = model_cfg.get("n_steps")
    if configured is not None and configured != calculated:
        print(
            f"Override model.n_steps={configured} mit {calculated} "
            f"(Formel: max_steps ({max_steps}) // num_cpu ({num_cpu}))"
        )
    model_cfg["n_steps"] = calculated


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_n_steps_rule(cfg)
    logging_cfg = cfg.get("logging", {})

    # 1) Run-Ordner anlegen
    if args.resume:
        # Bei Resume: Bestimme session_root aus dem Checkpoint-Pfad
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint nicht gefunden: {checkpoint_path}")
        
        # Annahme: Checkpoint liegt in checkpoints/ Unterordner
        if checkpoint_path.parent.name == "checkpoints":
            session_root = checkpoint_path.parent.parent
        else:
            session_root = checkpoint_path.parent
        
        print(f"Resuming training from {checkpoint_path}")
        print(f"Using existing session directory: {session_root}")
    else:
        # Neues Training: Neuen session_root erstellen
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_root = Path(cfg["paths"]["session_root"]) / now
        print(f"Starting new training session: {session_root}")

    dirs = make_run_dirs(session_root)

    # config speichern (nur bei neuem Training)
    if not args.resume:
        dest = dirs["root"] / args.config.name
        shutil.copyfile(args.config, dest)
    
    # Effective config fÃ¼r Debugging speichern
    effective_config = {
        "num_cpu": cfg.get("num_cpu", 1),
        "max_steps": cfg["env"]["max_steps"],
        "n_steps": cfg["model"]["n_steps"],
        "save_freq": cfg.get("save_freq", 10000),
        "save_freq_stats": cfg.get("save_freq_stats", 100),
        "reset_interval": cfg["env"]["max_steps"] // cfg["model"]["n_steps"],
        "total_timesteps": cfg.get("total_timesteps", 1e6),
        "variant": args.variant,
        "config_file": str(args.config),
        "logging_format": logging_cfg.get("format", "json")
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
    # Falls init_state leer ist, bleibt es leer (fÃ¼r No-State-Loading Tests)

    # 3) Vectorized environments
    num_cpu = cfg.get("num_cpu", 1)
    module_name = cfg["env"]["module"]
    class_name = cfg["env"]["class"]
    
    # num_cpu zur env_conf hinzufÃ¼gen
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
        # fÃ¼r RecurrentPPO immer die sb3_contribâ€Policyâ€Klasse
        policy_key = MultiInputLstmPolicy  # :contentReference[oaicite:0]{index=0}
    elif model_type == "RecurrentPPOLD":
        ModelClass = RecurrentPPOLD
        # fÃ¼r unsere LDâ€Variante die selbstdefinierte Policyâ€Klasse
        policy_key = MultiInputLstmPolicyLD  # :contentReference[oaicite:1]{index=1}
    else:
        raise ValueError(f"Unbekannter model.type: {model_type}")

    # Parse total_timesteps robust gegen Strings wie '1e8'
    raw_ts = cfg.get("total_timesteps", 1e6)
    if isinstance(raw_ts, str):
        try:
            raw_ts = float(raw_ts)
        except ValueError:
            raise ValueError(f"total_timesteps must be numeric, got {raw_ts}")
    total_timesteps = int(raw_ts)

    if args.resume:
        # Modell von Checkpoint laden
        print(f"Loading model from {args.resume}")
        model = ModelClass.load(args.resume, env=vec_env)
        
        # Verbleibende Schritte berechnen
        completed_steps = model.num_timesteps
        remaining_steps = total_timesteps - completed_steps
        print(f"Model has completed {completed_steps:,} steps")
        print(f"Will train for {remaining_steps:,} more steps")
        
        if remaining_steps <= 0:
            print("Training already completed!")
            return
    else:
        # Neues Modell erstellen
        # Grundlegende kwargs
        base_kwargs = {
            "policy": policy_key,
            "env": vec_env,
            "tensorboard_log": str(dirs["tensorboard"]),
            # seed und verbose kÃ¶nnen auch hier aufgenommen werden
        }

        # Erlaubte zusÃ¤tzliche Hyperparameter
        for key in [
            "learning_rate", "n_steps", "batch_size", "n_epochs", "gamma",
            "gae_lambda", "clip_range", "clip_range_vf", "ent_coef", "vf_coef",
            "max_grad_norm", "seed", "verbose", "device", "ld_coef"
        ]:
            if key in model_cfg:
                base_kwargs[key] = model_cfg[key]

        model = ModelClass(**base_kwargs)
        completed_steps = 0
        remaining_steps = total_timesteps

    # 5) Callbacks
    save_freq = int(cfg.get("save_freq", 10000))
    
    # CheckpointCallback mit korrekter Resume-Logik
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(dirs["checkpoints"]),
        name_prefix=args.variant
    )
    
    # WICHTIG: n_calls korrekt setzen fÃ¼r Resume
    if args.resume:
        # Berechne wie viele Callback-Aufrufe bereits stattgefunden haben
        # Pro environment step wird die callback einmal aufgerufen
        # Bei VecEnv mit n_envs wird save_freq durch n_envs geteilt (siehe SB3 Doku)
        expected_calls = completed_steps // num_cpu
        next_checkpoint_at = ((expected_calls // save_freq) + 1) * save_freq
        calls_until_next = next_checkpoint_at - expected_calls
        
        # Setze n_calls so, dass der nÃ¤chste Checkpoint korrekt ausgelÃ¶st wird
        checkpoint_cb.n_calls = expected_calls
        
        print(f"Resume: Setting checkpoint callback n_calls to {expected_calls:,}")
        print(f"Next checkpoint will be saved after {calls_until_next:,} more callback calls")
        print(f"That corresponds to {calls_until_next * num_cpu:,} more environment steps")

    cb_list = [checkpoint_cb]
    cb_list.append(TensorboardCallback(str(dirs["tensorboard"])))
    
    # StatsCallback nur hinzufÃ¼gen wenn save_freq_stats > 0
    stats_freq = int(cfg.get("save_freq_stats", 100))
    stats_format = logging_cfg.get("format", "json")
    raw_structured = logging_cfg.get("structured", True)
    if isinstance(raw_structured, str):
        stats_structured = raw_structured.lower() not in ("0", "false", "no")
    else:
        stats_structured = bool(raw_structured)
    stats_verbose = int(logging_cfg.get("verbose", 1))
    if stats_freq > 0:
        cb_list.append(
            StatsCallback(
                save_freq=stats_freq,
                save_path=str(dirs["logs"]),
                output_format=stats_format,
                structured=stats_structured,
                verbose=stats_verbose
            )
        )

    
    # 6) Training
    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=CallbackList(cb_list),
            tb_log_name=args.variant,
            reset_num_timesteps=False
        )
        # Finaler Checkpoint nach erfolgreichem Training
        final_checkpoint = dirs["checkpoints"] / f"{args.variant}_final_model.zip"
        model.save(str(final_checkpoint))
        print(f"Final model saved to {final_checkpoint}")
    except KeyboardInterrupt:
        print(" Training interrupted â€“ finalisiere JSON-Logs â€¦")
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

