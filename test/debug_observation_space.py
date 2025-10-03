"""
Debug Script: Compare Observation Spaces
Original vs Refactored Environments

This script compares the observation space structure between
the original and refactored environments to identify shape mismatches.
"""

import sys
from pathlib import Path

# Add paths for both pipelines
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "Pipeline" / "src"))

import yaml
import numpy as np

print("=" * 80)
print("OBSERVATION SPACE COMPARISON")
print("=" * 80)

# Load v1 config
config_path = Path(__file__).parent.parent / "configs" / "v1.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

# Prepare minimal config for environment initialization
env_config = {
    **config["env"],
    "session_path": Path("test_output"),
    "worker_rank": 0,
    "num_cpu": 1,
}

print("\n1. ORIGINAL ENVIRONMENT (red_gym_env_v2.py)")
print("-" * 80)
try:
    from poke_pipeline.red_gym_env_v2 import RedGymEnv as OriginalEnv
    orig_env = OriginalEnv(env_config)

    print(f"Observation Space Keys: {list(orig_env.observation_space.spaces.keys())}")
    print("\nObservation Space Details:")
    for key, space in orig_env.observation_space.spaces.items():
        print(f"  {key:15s}: {space}")

    # Try to reset and get observation
    print("\nAttempting reset...")
    orig_obs, _ = orig_env.reset()
    print("[OK] Reset successful")

    print("\nActual Observation Shapes:")
    for key, value in orig_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:15s}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key:15s}: {type(value)}")

    orig_env.pyboy.stop()

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n2. REFACTORED ENVIRONMENT (FrameStackEnv)")
print("-" * 80)
try:
    from environments.frame_stack_env import FrameStackEnv
    new_env = FrameStackEnv(env_config)

    print(f"Observation Space Keys: {list(new_env.observation_space.spaces.keys())}")
    print("\nObservation Space Details:")
    for key, space in new_env.observation_space.spaces.items():
        print(f"  {key:15s}: {space}")

    # Try to reset and get observation
    print("\nAttempting reset...")
    new_obs, _ = new_env.reset()
    print("[OK] Reset successful")

    print("\nActual Observation Shapes:")
    for key, value in new_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key:15s}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key:15s}: {type(value)}")

    new_env.pyboy.stop()

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("ANALYSIS")
print("=" * 80)

# Compare keys
print("\nKey Differences:")
orig_keys = set(orig_env.observation_space.spaces.keys()) if 'orig_env' in locals() else set()
new_keys = set(new_env.observation_space.spaces.keys()) if 'new_env' in locals() else set()

missing_in_refactored = orig_keys - new_keys
extra_in_refactored = new_keys - orig_keys

if missing_in_refactored:
    print(f"  Keys MISSING in refactored: {missing_in_refactored}")
if extra_in_refactored:
    print(f"  Keys EXTRA in refactored: {extra_in_refactored}")
if not missing_in_refactored and not extra_in_refactored:
    print("  [OK] Keys match!")

print("\n" + "=" * 80)
