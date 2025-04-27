#!/usr/bin/env python3
import argparse, re, time, uuid
from pathlib import Path

# --- CLI-Argumente ---
parser = argparse.ArgumentParser(description="Interaktive Wiedergabe eines vortrainierten Agenten")
parser.add_argument("--variant",
    choices=["v1","v2","v3","v4"], required=True,
    help="Welcher Env/Algo-Typ: v1/v2=PPO, v3=RecurrentPPO, v4=RecurrentPPOLD")
parser.add_argument("--session-dir", type=Path,
    help="Pfad zum Session-Ordner (experiments/vX/YYYYMMDD_HHMMSS). " +
         "Wenn nicht gesetzt, wird unter experiments/<variant> der neuste gewählt.")
parser.add_argument("--checkpoint", type=Path,
    help="Konkrete .zip-Datei für das Modell. " +
         "Wenn nicht gesetzt, wähle automatisch den weitesten Fortschritt.")
args = parser.parse_args()

# --- 1) Finde session_dir ---
if args.session_dir is None:
    base = Path("experiments") / args.variant
    sessions = [d for d in base.iterdir() if d.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"Kein Session-Ordner unter {base}")
    # nach Ordnernamen sortieren (yyyymmdd_HHMMSS) → letzter ist der jüngste
    args.session_dir = sorted(sessions)[-1]
print(f"[INFO] session-dir = {args.session_dir}")

# --- 2) Finde checkpoint .zip ---
if args.checkpoint and not args.checkpoint.exists():
    raise FileNotFoundError(f"{args.checkpoint} nicht gefunden")
if args.checkpoint is None:
    ckpt_dir = args.session_dir / "checkpoints"
    zips = list(ckpt_dir.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"Keine .zip im {ckpt_dir}")
    # 2a) Versuch, Schrittzahl aus Name zu parsen
    def parse_steps(p: Path):
        m = re.search(r"(\d+)_?steps?", p.stem)
        return int(m.group(1)) if m else -1
    zips_with_steps = [(parse_steps(p), p) for p in zips]
    # falls wenigstens ein Name eine Zahl enthielt, wähle max; sonst neuste mod-Zeit
    if any(s>0 for s,_ in zips_with_steps):
        _, args.checkpoint = max(zips_with_steps, key=lambda x: x[0])
    else:
        args.checkpoint = max(zips, key=lambda p: p.stat().st_mtime)
print(f"[INFO] checkpoint = {args.checkpoint}")

# --- 3) Env- und Algo-Klasse je variant ---
if args.variant == "v1":
    # das originale Whidden-Environment
    from poke_pipeline.red_gym_env_v2 import RedGymEnv as EnvCls
    from stable_baselines3 import PPO as AlgoCls
elif args.variant == "v2":
    # Deine erste, adaptierte Version
    from poke_pipeline.red_gym_env_v2_adapted import RedGymEnv as EnvCls
    from stable_baselines3 import PPO as AlgoCls
elif args.variant == "v3":
    from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM as EnvCls
    from sb3_contrib import RecurrentPPO as AlgoCls
else:  # v4
    from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM as EnvCls
    from poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD as AlgoCls

# --- 4) Environment instanziieren ---
env_conf = {
    "headless": False,
    "save_final_state": True,
    "early_stop": False,
    "action_freq": 24,
    "init_state": "../has_pokedex_nballs.state",
    "max_steps": 2**23,
    "print_rewards": True,
    "save_video": False,
    "fast_video": True,
    "session_path": args.session_dir,
    "gb_path": "../PokemonRed.gb",
    "debug": False
}
env = EnvCls(env_conf)

# --- 5) Modell laden ---
print("[INFO] Lade Modell …")
model = AlgoCls.load(
    str(args.checkpoint),
    env=env,
    custom_objects={"lr_schedule": 0, "clip_range": 0},
)

# --- 6) Interaktive Wiedergabe ---
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, term, trunc, _ = env.step(action)
    env.render()
    done = term or trunc

print("▶ Run beendet.")
env.close()
