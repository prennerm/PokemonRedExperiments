#!/usr/bin/env python3
"""
LSTM-spezifische interaktive Wiedergabe für Varianten V3 und V4

Dieses Skript ist speziell für LSTM-basierte Agenten (RecurrentPPO/RecurrentPPOLD) 
entwickelt und implementiert korrektes LSTM State Management.

V3: RecurrentPPO mit RedGymEnvLSTM
V4: RecurrentPPOLD (Lambda Discrepancy) mit RedGymEnvLSTM
"""

import argparse, re, time, uuid
import numpy as np
from pathlib import Path

# --- CLI-Argumente ---
parser = argparse.ArgumentParser(description="LSTM-Agent Wiedergabe für V3/V4")
parser.add_argument("--variant",
    required=True,
    help="Experiment-Ordner Name (z.B. 'v3', 'v4', 'v4_production', 'v4_fast')")
parser.add_argument("--session-dir", type=Path,
    help="Pfad zum Session-Ordner (experiments/vX/YYYYMMDD_HHMMSS). " +
         "Wenn nicht gesetzt, wird unter experiments/<variant> der neuste gewählt.")
parser.add_argument("--checkpoint", type=Path,
    help="Konkrete .zip-Datei für das Modell. " +
         "Wenn nicht gesetzt, wähle automatisch den weitesten Fortschritt.")
parser.add_argument("--deterministic", action="store_true", default=False,
    help="Nutze deterministische Aktionen (Standard: False für Exploration)")
parser.add_argument("--verbose", action="store_true", default=False,
    help="Ausführliche Logging-Ausgaben")
args = parser.parse_args()

print(f"[INFO] Starte LSTM-Agent für Variante {args.variant}")

# --- 1) Finde session_dir ---
if args.session_dir is None:
    # Direkte Pfad-Auflösung: nutze variant als Ordnername
    base_dir = Path("experiments") / args.variant
    
    if not base_dir.exists() or not base_dir.is_dir():
        available_dirs = [d.name for d in Path("experiments").iterdir() if d.is_dir()]
        raise FileNotFoundError(f"Ordner '{base_dir}' nicht gefunden. "
                              f"Verfügbare Ordner: {available_dirs}")
    
    print(f"[INFO] Nutze Basis-Ordner: {base_dir}")
    
    # Suche Sessions in diesem Basis-Ordner
    sessions = [d for d in base_dir.iterdir() if d.is_dir()]
    if not sessions:
        raise FileNotFoundError(f"Keine Session-Ordner unter {base_dir}")
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

# --- 3) Algorithmus und Environment basierend auf Variante ---
# Bestimme Algorithmus-Typ aus dem Varianten-Namen
if "v3" in args.variant:
    print("[INFO] Lade RecurrentPPO (V3)")
    from sb3_contrib import RecurrentPPO as AlgoCls
    from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
    from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM as EnvCls
    custom_objects = {
        "lr_schedule": 0, 
        "clip_range": 0,
        "policy_class": MultiInputLstmPolicy  # Klassen-Objekt statt String
    }
elif "v4" in args.variant:
    print("[INFO] Lade RecurrentPPOLD (V4) mit Lambda Discrepancy")
    from poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD as AlgoCls, MultiInputLstmPolicyLD
    from poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM as EnvCls
    custom_objects = {
        "lr_schedule": 0, 
        "clip_range": 0,
        "policy_class": MultiInputLstmPolicyLD  # Klassen-Objekt statt String
    }
else:
    raise ValueError(f"Unbekannte Variante: {args.variant}. Muss 'v3' oder 'v4' enthalten.")

# --- 4) Environment instanziieren ---
env_conf = {
    "headless": False,
    "save_final_state": True,
    "early_stop": False,
    "action_freq": 24,           # Wie beim Training (v4_production.yaml)
    "init_state": "data/init.state",
    "max_steps": 2**23,          # Lange Sessions für interaktive Nutzung
    "print_rewards": True,
    "save_video": False,
    "fast_video": True,
    "session_path": args.session_dir,
    "gb_path": "data/PokemonRed.gb",  # Angepasster Pfad
    "debug": args.verbose,
    "reward_scale": 0.5,         # Wie beim Training
    "explore_weight": 0.25       # Wie beim Training
}

print("[INFO] Initialisiere RedGymEnvLSTM...")
env = EnvCls(env_conf)

# --- 5) Modell laden mit korrektem Algorithmus ---
print(f"[INFO] Lade {args.variant.upper()}-Modell...")
try:
    model = AlgoCls.load(
        str(args.checkpoint),
        env=env,
        custom_objects=custom_objects,
    )
    print(f"[INFO] ✅ Modell erfolgreich geladen: {args.checkpoint.name}")
except Exception as e:
    print(f"[ERROR] ❌ Fehler beim Laden des Modells: {e}")
    raise

# --- 6) LSTM State Management Setup ---
print("[INFO] Initialisiere LSTM States...")
lstm_states = None  # Initial: Lass das Modell sie initialisieren
num_envs = 1        # Single Environment
episode_starts = np.ones((num_envs,), dtype=bool)  # Erste Prediction = Episode Start

# --- 7) Interaktive Wiedergabe mit korrektem LSTM Management ---
print("[INFO] Starte LSTM-Agent-Wiedergabe...")
print(f"[INFO] Modus: {'Deterministisch' if args.deterministic else 'Exploratif'}")
print("[INFO] Drücke Ctrl+C zum Beenden")

obs, _ = env.reset()
done = False
step_count = 0
episode_count = 1

try:
    while not done:
        try:
            # Korrekte LSTM-Prediction mit allen erforderlichen Parametern
            action, lstm_states = model.predict(
                obs, 
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=args.deterministic
            )
            
            # Nach der ersten Prediction: episode_start ist False
            episode_starts = np.zeros((num_envs,), dtype=bool)
            
            # Environment-Step
            obs, reward, term, trunc, info = env.step(action)
            env.render()
            
            step_count += 1
            done = term or trunc
            
            # Logging alle 1000 Steps
            if args.verbose and step_count % 1000 == 0:
                print(f"[INFO] Episode {episode_count}, Step {step_count}: Agent läuft...")
                if hasattr(info, 'keys') and 'reward_info' in info:
                    print(f"[DEBUG] Reward Info: {info['reward_info']}")
            
            # Episode-Ende: Reset LSTM States
            if done:
                print(f"[INFO] Episode {episode_count} beendet nach {step_count} Steps")
                if not term:  # Nur bei Truncation, nicht bei echtem Termination
                    print("[INFO] Episode wurde truncated, starte neue Episode...")
                    obs, _ = env.reset()
                    lstm_states = None  # Reset LSTM States
                    episode_starts = np.ones((num_envs,), dtype=bool)
                    step_count = 0
                    episode_count += 1
                    done = False
                    
        except KeyboardInterrupt:
            print("\n[INFO] ⏹️ Benutzer-Unterbrechung erkannt")
            break
        except Exception as e:
            print(f"[ERROR] Exception während Prediction: {e}")
            if args.verbose:
                import traceback
                print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            break

except Exception as e:
    print(f"[ERROR] Allgemeiner Fehler: {e}")
    if args.verbose:
        import traceback
        print(f"[DEBUG] Traceback: {traceback.format_exc()}")

finally:
    print("[INFO] 🔄 Schließe Environment...")
    env.close()
    print("[INFO] ✅ LSTM-Agent Run beendet.")
