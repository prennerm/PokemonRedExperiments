# Environment Refactoring Status & Current Issue

## Context

### Original Files (Reference - DO NOT MODIFY)
Located in: `../Pipeline/src/poke_pipeline/`
- `red_gym_env_v2.py` (608 lines) - Used by v1 (Frame Stacking)
- `red_gym_env_v2_adapted.py` (701 lines) - Used by v2 (Single Frame)
- `red_gym_env_lstm.py` (695 lines) - Used by v3/v4 (LSTM)

These files were **successfully running** in the old pipeline before refactoring.

### New Refactored Files
Located in: `src/environments/`
- `base_env.py` (700 lines) - Common logic extracted from all 3 original files
- `frame_stack_env.py` (90 lines) - v1: Extends BaseRedGymEnv
- `single_frame_env.py` (85 lines) - v2: Extends BaseRedGymEnv
- `lstm_env.py` (190 lines) - v3/v4: Extends BaseRedGymEnv

**Total: ~1065 lines** (vs 2004 lines original = **47% reduction**)

## What Has Been Done Successfully

### 1. Code Analysis ✅
- Compared all 3 original environment files
- Identified ~400 lines of duplicated common code
- Documented key differences (see [ENVIRONMENT_ANALYSIS.md](ENVIRONMENT_ANALYSIS.md))

### 2. Architecture Restructure ✅
- Created `src/environments/` directory (spec-compliant)
- Created `src/utils/` directory
- Moved `global_map.py` → `utils/map_utils.py`
- Moved `data/` → `src/environments/data/`
- Updated all `__init__.py` files

### 3. Base Environment Implementation ✅
Created abstract base class `BaseRedGymEnv` with:
- PyBoy initialization & configuration
- Event/map data loading
- Memory reading methods (`read_m()`, `read_bit()`, etc.)
- Game state reading (levels, badges, party, etc.)
- Reward calculation (events, exploration, healing, etc.)
- Stats tracking & video recording
- Reset logic with staggered resets
- Abstract methods for subclasses:
  - `_build_observation_space()`
  - `_get_obs()`
  - `_init_frame_stack()`
  - `_update_frame_stack(screen)`
  - `_get_action_observation()`
  - `_update_action_history(action)`

### 4. Variant Implementations ✅
**FrameStackEnv (v1):**
- 3-frame stacking with rolling window
- `recent_actions` array (3 elements)
- Observation space: `screens`, `enc_coords`, `events`, `map`, `recent_actions`

**SingleFrameEnv (v2):**
- Single frame (no stacking)
- `recent_action` single value
- Observation space: `screens`, `enc_coords`, `events`, `map`, `recent_action`

**LSTMEnv (v3/v4):**
- Single frame + 10-step action history
- **Episode start flag** for LSTM state reset
- Aggressive staggered resets with jitter
- Observation space: `screens`, `enc_coords`, `events`, `map`, `recent_actions`, `episode_start`

### 5. Configuration Updates ✅
All configs updated to use new module names:
- `v1.yaml`: `module: "frame_stack_env"`, `class: "FrameStackEnv"`
- `v2.yaml`: `module: "single_frame_env"`, `class: "SingleFrameEnv"`
- `v3.yaml`: `module: "lstm_env"`, `class: "LSTMEnv"`
- `v4.yaml`: `module: "lstm_env"`, `class: "LSTMEnv"`

### 6. Import Updates ✅
- `train.py`: Changed to import from `environments.{module_name}`
- `ppo_lambda_discrepancy.py`: Removed legacy imports

## Current Problem: Shape Mismatch ⚠️

### Error Message
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x204288 and 256x256)
```

### Analysis
The error occurs during PPO's forward pass when processing observations. The shape `32x204288` suggests:
- Batch size: 32
- Flattened observation: 204288 elements

Expected CNN input should be smaller. This indicates the observation space shape doesn't match what the neural network expects.

### Likely Causes

1. **Observation Space Mismatch:**
   - New environments may produce different observation shapes than originals
   - Possible issue: `recent_screens` initialization or shape

2. **Frame Stack Initialization:**
   - `FrameStackEnv._update_frame_stack()` may not match original behavior
   - Original rolled frames: `self.recent_screens[:, :, 0] = cur_screen[:,:, 0]`
   - New implementation might have different axis handling

3. **Render Output Shape:**
   - `base_env.render()` returns `(72, 80, 1)` after downscaling
   - Frame stack needs to handle this correctly for 3-frame stacking

### Debugging Steps Needed

1. **Compare Observation Shapes:**
   ```python
   # Test original environment
   from Pipeline.src.poke_pipeline.red_gym_env_v2 import RedGymEnv as OriginalEnv
   orig_env = OriginalEnv(config)
   orig_obs, _ = orig_env.reset()
   print("Original obs shapes:", {k: v.shape for k, v in orig_obs.items()})

   # Test new environment
   from environments.frame_stack_env import FrameStackEnv
   new_env = FrameStackEnv(config)
   new_obs, _ = new_env.reset()
   print("New obs shapes:", {k: v.shape for k, v in new_obs.items()})
   ```

2. **Check Frame Stacking Logic:**
   - Verify `_init_frame_stack()` creates correct shape
   - Verify `_update_frame_stack()` maintains correct shape
   - Compare with original `update_recent_screens()` behavior

3. **Check Observation Space Definition:**
   - Ensure `self.output_shape` is set correctly before `_build_observation_space()`
   - Verify all observation dict keys match

### Files to Check

**Original (Reference):**
- `../Pipeline/src/poke_pipeline/red_gym_env_v2.py:401-405` - `update_recent_screens()`
- `../Pipeline/src/poke_pipeline/red_gym_env_v2.py:89` - `self.output_shape` definition
- `../Pipeline/src/poke_pipeline/red_gym_env_v2.py:97-106` - Observation space definition

**New (To Debug):**
- `src/environments/base_env.py:207-209` - Frame stack initialization in reset
- `src/environments/frame_stack_env.py:45-49` - `_init_frame_stack()`
- `src/environments/frame_stack_env.py:51-55` - `_update_frame_stack()`
- `src/environments/frame_stack_env.py:20-48` - Observation space definition

## Next Steps

1. **Create Debug Script:**
   - Load both original and new environments
   - Compare observation shapes element by element
   - Identify exact shape mismatch

2. **Fix Shape Issues:**
   - Adjust frame stacking logic if needed
   - Ensure observation space matches original exactly

3. **Test All Variants:**
   - v1 (FrameStackEnv)
   - v2 (SingleFrameEnv)
   - v3 (LSTMEnv with RecurrentPPO)
   - v4 (LSTMEnv with RecurrentPPOLD)

4. **Cleanup:**
   - Remove old environment files from `src/pipeline_v2/`
   - Complete remaining spec restructure (callbacks/, models/, trainers/)

## Testing Commands

```bash
# Set environment
export KMP_DUPLICATE_LIB_OK=TRUE
export PYTHONPATH=src

# Test v1
"C:\Users\marwi\anaconda3\envs\poke_env\python.exe" -m pipeline_v2.train --variant v1 --config configs/v1.yaml

# Test v2
"C:\Users\marwi\anaconda3\envs\poke_env\python.exe" -m pipeline_v2.train --variant v2 --config configs/v2.yaml

# Test v3
"C:\Users\marwi\anaconda3\envs\poke_env\python.exe" -m pipeline_v2.train --variant v3 --config configs/v3.yaml

# Test v4
"C:\Users\marwi\anaconda3\envs\poke_env\python.exe" -m pipeline_v2.train --variant v4 --config configs/v4.yaml
```

## Old Files to Remove (After Testing)

Located in: `src/pipeline_v2/`
- `red_gym_env_v2.py` (old v1)
- `red_gym_env_v2_adapted.py` (old v2)
- `red_gym_env_lstm.py` (old v3/v4)
- `global_map.py` (moved to utils/)

These should be deleted once new environments are verified working.
