#!/usr/bin/env python3
"""
Debug script to check TensorBoard data.
"""

from pathlib import Path
import sys

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.tensorboard import extract_tensorboard_data
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to the actual tensorboard logs
log_dir = Path("results/20250718_163601/tensorboard/PPO_1")

print(f"Checking TensorBoard logs in: {log_dir}")
print(f"Directory exists: {log_dir.exists()}")

if log_dir.exists():
    # List files
    files = list(log_dir.iterdir())
    print(f"Files in directory: {[f.name for f in files]}")
    
    # Check available metrics
    try:
        event_accumulator = EventAccumulator(str(log_dir))
        event_accumulator.Reload()
        
        available_scalars = event_accumulator.Tags()['scalars']
        print(f"\nAvailable scalar metrics:")
        for metric in available_scalars:
            print(f"  - {metric}")
        
        # Try to extract default metric
        print(f"\nTrying to extract 'rollout/ep_rew_mean':")
        try:
            rewards, timesteps = extract_tensorboard_data(log_dir, 'rollout/ep_rew_mean')
            print(f"  Success! Got {len(rewards)} data points")
            print(f"  First 5 rewards: {rewards[:5]}")
            print(f"  First 5 timesteps: {timesteps[:5]}")
        except Exception as e:
            print(f"  Failed: {e}")
            
    except Exception as e:
        print(f"Error loading EventAccumulator: {e}")
else:
    print("Directory does not exist!")
