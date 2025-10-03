#!/usr/bin/env python3
"""
memory_analysis.py - Live LSTM Memory Tracking für Pokemon Red Agent

Analysiert LSTM Hidden States während des Spielens:
- Lädt trainierte Models und spielt Episoden
- Loggt Hidden States, Game States und Aktionen
- Visualisiert Memory Patterns und Evolution
- t-SNE/UMAP Clustering der Memory States
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import torch
import json
from collections import defaultdict, deque
from datetime import datetime

# Import project utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.plot_helpers import setup_plot_style, save_plot, PLOT_CONFIG

# Import custom classes
from src.poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD, MultiInputLstmPolicyLD
from src.poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM
from stable_baselines3.common.vec_env import DummyVecEnv

class MemoryTracker:
    """Live LSTM Memory State Tracker während Agent Episoden"""
    
    def __init__(self, experiment_dir: Path, verbose: bool = False):
        self.experiment_dir = experiment_dir
        self.verbose = verbose
        self.memory_log = []
        self.episode_data = defaultdict(list)
        self.model = None
        self.env = None
        
        if verbose:
            print(f"🧠 Memory Tracker initialisiert für: {experiment_dir}")
    
    def load_trained_model(self, checkpoint_name: str = None) -> bool:
        """Lade ein trainiertes Model aus den Checkpoints"""
        checkpoint_dir = self.experiment_dir / "checkpoints"
        
        if not checkpoint_dir.exists():
            print(f"❌ Checkpoint-Verzeichnis nicht gefunden: {checkpoint_dir}")
            return False
        
        # Finde Checkpoint
        if checkpoint_name:
            checkpoint_path = checkpoint_dir / f"{checkpoint_name}.zip"
        else:
            # Nimm neuestes Checkpoint
            checkpoints = list(checkpoint_dir.glob("*.zip"))
            if not checkpoints:
                print("❌ Keine Checkpoints gefunden!")
                return False
            checkpoint_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        if not checkpoint_path.exists():
            print(f"❌ Checkpoint nicht gefunden: {checkpoint_path}")
            return False
        
        try:
            # Lade Model (RecurrentPPOLD)
            self.model = RecurrentPPOLD.load(checkpoint_path)
            if self.verbose:
                print(f"✅ Model geladen: {checkpoint_path.name}")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Model-Laden: {e}")
            return False
    
    def create_environment(self, config_override: Dict = None) -> bool:
        """Erstelle Test-Environment basierend auf ursprünglicher Config"""
        try:
            # Standard Config für Testing
            env_config = {
                "headless": True,
                "save_final_state": False,
                "early_stop": False,
                "action_freq": 24,
                "init_state": "data/init.state",
                "max_steps": 5000,  # Kurze Episoden für Testing
                "print_rewards": False,
                "save_video": False,
                "fast_video": True,
                "session_path": self.experiment_dir / "memory_tracking",
                "gb_path": "data/PokemonRed.gb",
                "debug": False,
                "reward_scale": 1.0,
                "explore_weight": 0.25,
                "worker_rank": 0,
                "num_cpu": 1
            }
            
            # Override falls angegeben
            if config_override:
                env_config.update(config_override)
            
            # Environment erstellen
            env = RedGymEnvLSTM(env_config)
            self.env = DummyVecEnv([lambda: env])
            
            if self.verbose:
                print("✅ Test-Environment erstellt")
            return True
            
        except Exception as e:
            print(f"❌ Fehler beim Environment-Erstellen: {e}")
            return False
    
    def extract_lstm_states(self, obs, lstm_states, episode_starts):
        """Extrahiere LSTM Hidden States aus dem Policy Network"""
        try:
            # Konvertiere OrderedDict mit NumPy Arrays zu PyTorch Tensors
            obs_tensors = self._convert_obs_to_tensors(obs)
            
            if self.verbose:
                print(f"� Obs converted to tensors: {type(obs_tensors)}")
            
            # Forward pass durch das Policy Network
            with torch.no_grad():
                features = self.model.policy.extract_features(obs_tensors)
                if self.model.policy.share_features_extractor:
                    pi_feat, vf_feat = features, features
                else:
                    pi_feat, vf_feat = features
                
                # Process sequence durch LSTM
                latent_pi, new_lstm_states = self.model.policy._process_sequence(
                    pi_feat, lstm_states.pi, episode_starts, self.model.policy.lstm_actor
                )
                
                # Extrahiere Hidden State (h) und Cell State (c)
                hidden_state = new_lstm_states[0].cpu().numpy()  # Hidden state
                cell_state = new_lstm_states[1].cpu().numpy()    # Cell state
                
                return {
                    'hidden_state': hidden_state,
                    'cell_state': cell_state,
                    'latent': latent_pi.cpu().numpy(),
                    'new_lstm_states': new_lstm_states
                }
        except Exception as e:
            if self.verbose:
                print(f"⚠️  LSTM State Extraction Fehler: {e}")
                print(f"   Obs Type: {type(obs)}")
                if hasattr(obs, 'keys'):
                    print(f"   Obs Keys: {list(obs.keys())}")
            return None
    
    def _convert_obs_to_tensors(self, obs):
        """Konvertiere OrderedDict mit NumPy Arrays zu PyTorch Tensors"""
        # Bestimme das Device des Models
        device = next(self.model.policy.parameters()).device
        
        if isinstance(obs, dict) or hasattr(obs, 'keys'):
            # OrderedDict/Dict → PyTorch Tensors
            converted = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float().to(device)  # Move to model device
                    
                    # Spezialbehandlung für CNN Inputs - brauchen channels-first Format
                    if key in ['screens', 'map'] and len(tensor.shape) == 4:
                        # Von [batch, height, width, channels] zu [batch, channels, height, width]
                        tensor = tensor.permute(0, 3, 1, 2)
                        if self.verbose:
                            print(f"🔧 {key} reshaped: {value.shape} → {tensor.shape}")
                    elif key in ['screens', 'map'] and len(tensor.shape) == 3:
                        # Von [height, width, channels] zu [channels, height, width] 
                        tensor = tensor.permute(2, 0, 1)
                        if self.verbose:
                            print(f"🔧 {key} reshaped: {value.shape} → {tensor.shape}")
                    
                    converted[key] = tensor
                elif isinstance(value, (list, tuple)):
                    converted[key] = torch.tensor(value, dtype=torch.float32, device=device)
                else:
                    converted[key] = torch.tensor(value, dtype=torch.float32, device=device)
            
            if self.verbose:
                print(f"📍 Tensors moved to device: {device}")
                for k, v in converted.items():
                    device_info = f" (device: {v.device})" if hasattr(v, 'device') else ""
                    print(f"   {k}: {v.shape if hasattr(v, 'shape') else type(v)}{device_info}")
            
            return converted
        elif isinstance(obs, (list, tuple)):
            # List/Tuple → Tensors
            return [torch.from_numpy(o).float().to(device) if isinstance(o, np.ndarray) 
                   else torch.tensor(o, dtype=torch.float32, device=device) for o in obs]
        elif isinstance(obs, np.ndarray):
            # Single Array → Tensor
            return torch.from_numpy(obs).float().to(device)
        else:
            return obs
    
    def run_tracked_episodes(self, num_episodes: int = 5, max_steps_per_episode: int = 1000) -> List[Dict]:
        """Führe Episoden aus und tracke LSTM Memory States"""
        if not self.model or not self.env:
            print("❌ Model oder Environment nicht geladen!")
            return []
        
        episode_logs = []
        
        for episode in range(num_episodes):
            if self.verbose:
                print(f"📺 Episode {episode + 1}/{num_episodes}")
            
            episode_log = {
                'episode': episode,
                'steps': [],
                'total_reward': 0,
                'memory_evolution': []
            }
            
            obs = self.env.reset()
            lstm_states = None
            episode_starts = np.ones((1,), dtype=bool)
            
            for step in range(max_steps_per_episode):
                # Vorhersage mit LSTM State Tracking
                action, lstm_states = self.model.predict(
                    obs, state=lstm_states, episode_start=episode_starts, deterministic=False
                )
                
                # Extrahiere LSTM States
                memory_data = self.extract_lstm_states(obs, lstm_states, episode_starts)
                
                # Environment Step
                new_obs, reward, done, info = self.env.step(action)
                
                # Logge Step-Daten
                step_data = {
                    'step': step,
                    'action': int(action[0]),
                    'reward': float(reward[0]),
                    'done': bool(done[0]),
                    'obs_info': self._get_obs_info(obs),  # Sichere Observation-Info
                    'memory_data': memory_data
                }
                
                episode_log['steps'].append(step_data)
                episode_log['total_reward'] += float(reward[0])
                
                # Memory Evolution tracking
                if memory_data:
                    memory_summary = {
                        'step': step,
                        'hidden_mean': float(np.mean(memory_data['hidden_state'])),
                        'hidden_std': float(np.std(memory_data['hidden_state'])),
                        'cell_mean': float(np.mean(memory_data['cell_state'])),
                        'cell_std': float(np.std(memory_data['cell_state'])),
                        'hidden_norm': float(np.linalg.norm(memory_data['hidden_state'])),
                        'cell_norm': float(np.linalg.norm(memory_data['cell_state']))
                    }
                    episode_log['memory_evolution'].append(memory_summary)
                
                obs = new_obs
                episode_starts = done
                
                if done[0]:
                    if self.verbose:
                        print(f"   Episode beendet nach {step + 1} Steps, Reward: {episode_log['total_reward']:.1f}")
                    break
            
            episode_logs.append(episode_log)
        
        return episode_logs
    
    def visualize_memory_evolution(self, episode_logs: List[Dict], output_dir: Path):
        """Visualisiere die Evolution der LSTM Memory States"""
        setup_plot_style()
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(episode_logs)))
        
        for ep_idx, episode_log in enumerate(episode_logs):
            if not episode_log['memory_evolution']:
                continue
                
            memory_df = pd.DataFrame(episode_log['memory_evolution'])
            color = colors[ep_idx]
            label = f"Episode {episode_log['episode'] + 1}"
            
            # Hidden State Mean
            axes[0].plot(memory_df['step'], memory_df['hidden_mean'], 
                        color=color, alpha=0.7, label=label)
            
            # Hidden State Std
            axes[1].plot(memory_df['step'], memory_df['hidden_std'], 
                        color=color, alpha=0.7, label=label)
            
            # Cell State Mean  
            axes[2].plot(memory_df['step'], memory_df['cell_mean'], 
                        color=color, alpha=0.7, label=label)
            
            # Cell State Std
            axes[3].plot(memory_df['step'], memory_df['cell_std'], 
                        color=color, alpha=0.7, label=label)
            
            # Hidden State Norm
            axes[4].plot(memory_df['step'], memory_df['hidden_norm'], 
                        color=color, alpha=0.7, label=label)
            
            # Cell State Norm
            axes[5].plot(memory_df['step'], memory_df['cell_norm'], 
                        color=color, alpha=0.7, label=label)
        
        # Styling
        titles = [
            'Hidden State Mean', 'Hidden State Std', 'Cell State Mean',
            'Cell State Std', 'Hidden State Norm', 'Cell State Norm'
        ]
        
        for i, (ax, title) in enumerate(zip(axes, titles)):
            ax.set_title(title)
            ax.set_xlabel('Step')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            if i == 0:  # Legend nur im ersten Plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Speichere Plot
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"memory_evolution_{timestamp}"
        save_plot(fig, output_dir, filename, ['png'])
        
        if self.verbose:
            print(f"💾 Memory Evolution Plot gespeichert: {output_dir / f'{filename}.png'}")
    
    def save_memory_data(self, episode_logs: List[Dict], output_dir: Path):
        """Speichere Memory-Daten für weitere Analyse"""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Speichere als Pickle für vollständige Daten
        pickle_path = output_dir / f"memory_data_{timestamp}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(episode_logs, f)
        
        # Speichere als JSON für readability (ohne raw numpy arrays)
        json_logs = []
        for episode_log in episode_logs:
            json_episode = {
                'episode': episode_log['episode'],
                'total_reward': episode_log['total_reward'],
                'num_steps': len(episode_log['steps']),
                'memory_evolution': episode_log['memory_evolution']
            }
            json_logs.append(json_episode)
        
        json_path = output_dir / f"memory_summary_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_logs, f, indent=2)
        
        if self.verbose:
            print(f"💾 Memory-Daten gespeichert:")
            print(f"   Vollständig: {pickle_path}")
            print(f"   Summary: {json_path}")
    
    def _get_obs_info(self, obs):
        """Sichere Extraktion von Observation-Informationen"""
        try:
            if isinstance(obs, dict) or hasattr(obs, 'keys'):
                # OrderedDict oder Dict Observation  
                return {f"key_{k}": str(v.shape) if hasattr(v, 'shape') else str(type(v)) 
                        for k, v in obs.items()}
            elif isinstance(obs, (list, tuple)):
                # List/Tuple von Observations
                return {f"obs_{i}": v.shape if hasattr(v, 'shape') else str(type(v)) 
                        for i, v in enumerate(obs)}
            elif hasattr(obs, 'shape'):
                # Single Array/Tensor
                return {'shape': obs.shape, 'type': str(type(obs))}
            else:
                return {'type': str(type(obs)), 'value': str(obs)[:100]}
        except Exception as e:
            return {'error': str(e), 'type': str(type(obs))}


def main():
    """Hauptfunktion für Live LSTM Memory Analysis"""
    parser = argparse.ArgumentParser(
        description="Live LSTM Memory Tracking für Pokemon Red Agent"
    )
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Name des Experiment-Subdirectories (z.B. 'v4_ld_02')")
    parser.add_argument("--timestamp", type=str, default=None,
                       help="Spezifischer Zeitstempel-Ordner (z.B. '20250907_221057')")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Spezifischer Checkpoint-Name (ohne .zip)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="Anzahl Episoden zu tracken")
    parser.add_argument("--max-steps", type=int, default=1000,
                       help="Maximale Steps pro Episode")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output-Verzeichnis für Plots und Daten")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # Experiment-Verzeichnis finden
        base_path = Path("experiments") / args.experiment_dir
        if not base_path.exists():
            raise FileNotFoundError(f"Experiment-Subdirectory nicht gefunden: {base_path}")
        
        if args.timestamp:
            experiment_dir = base_path / args.timestamp
            if not experiment_dir.exists():
                raise FileNotFoundError(f"Experiment-Ordner nicht gefunden: {experiment_dir}")
        else:
            experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]
            if not experiment_dirs:
                raise FileNotFoundError(f"Keine Experiment-Ordner in {base_path} gefunden")
            experiment_dir = max(experiment_dirs, key=lambda x: x.name)
        
        if args.verbose:
            print(f"🎯 Live Memory Tracking für: {experiment_dir}")
        
        # Output-Verzeichnis
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = experiment_dir / "memory_tracking"
        
        # Memory Tracker initialisieren
        tracker = MemoryTracker(experiment_dir, args.verbose)
        
        # Model laden
        print("🔄 Lade trainiertes Model...")
        if not tracker.load_trained_model(args.checkpoint):
            return
        
        # Environment erstellen  
        print("🔄 Erstelle Test-Environment...")
        if not tracker.create_environment():
            return
        
        # Live Memory Tracking durchführen
        print(f"🧠 Starte Memory Tracking ({args.episodes} Episoden, max {args.max_steps} Steps)...")
        episode_logs = tracker.run_tracked_episodes(args.episodes, args.max_steps)
        
        if not episode_logs:
            print("❌ Keine Episode-Daten erhalten!")
            return
        
        # Visualisierung erstellen
        print("📊 Erstelle Memory Evolution Visualisierung...")
        tracker.visualize_memory_evolution(episode_logs, output_dir)
        
        # Daten speichern
        print("💾 Speichere Memory-Daten...")
        tracker.save_memory_data(episode_logs, output_dir)
        
        print(f"✅ Live Memory Analysis abgeschlossen!")
        print(f"📁 Outputs gespeichert in: {output_dir}")
        print(f"🧠 {len(episode_logs)} Episoden getrackt")
        
        # Summary Statistics
        total_steps = sum(len(ep['steps']) for ep in episode_logs)
        avg_reward = np.mean([ep['total_reward'] for ep in episode_logs])
        print(f"📊 {total_steps} Steps analysiert, Ø Reward: {avg_reward:.1f}")
        
    except Exception as e:
        print(f"❌ Fehler: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()