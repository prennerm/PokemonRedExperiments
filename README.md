# Pokemon Red RL Pipeline V2

**Professional, modular reimplementation of Pokemon Red reinforcement learning training pipeline for Lambda Discrepancy research.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-2.0+-green.svg)](https://stable-baselines3.readthedocs.io/)

---

## 🎯 Overview

This pipeline implements **4 training variants** for Pokemon Red using reinforcement learning:

| Variant | Environment | Algorithm | Architecture |
|---------|-------------|-----------|--------------|
| **v1** | FrameStackEnv | PPO | 3-frame stacking |
| **v2** | SingleFrameEnv | PPO | Single frame |
| **v3** | LSTMEnv | RecurrentPPO | LSTM memory |
| **v4** | LSTMEnv | RecurrentPPOLD | LSTM + Lambda Discrepancy |

**Key Achievement:** Refactored ~2000 lines of duplicated environment code into modular structure with **53% code reduction** while preserving exact original functionality.

---

## 📁 Project Structure

```
Pipeline_V2/
├── src/
│   ├── environments/          # Refactored modular environments
│   │   ├── base_env.py        # Common logic (584 lines)
│   │   ├── frame_stack_env.py # v1 (94 lines)
│   │   ├── single_frame_env.py# v2 (87 lines)
│   │   └── lstm_env.py        # v3/v4 (184 lines)
│   ├── pipeline_v2/           # Training logic
│   │   ├── train.py           # Main training entry point
│   │   ├── ppo_lambda_discrepancy.py
│   │   └── callbacks.py       # Logging callbacks
│   └── utils/                 # Utilities
│       └── map_utils.py       # Map coordinate transforms
├── configs/                   # YAML configurations
│   ├── v1.yaml               # PPO + Frame Stack
│   ├── v2.yaml               # PPO + Single Frame
│   ├── v3.yaml               # RecurrentPPO
│   └── v4.yaml               # RecurrentPPOLD
├── data/                      # Game data (ROMs, save states)
├── docs/                      # Documentation
│   ├── PIPELINE_V2_SPECIFICATION.md
│   ├── ENVIRONMENT_ANALYSIS.md
│   └── REFACTORING_STATUS.md
├── test/                      # Testing & debugging
└── experiments/               # Training outputs (gitignored)
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Clone repository
git clone https://github.com/prennerm/PokemonRedExperiments.git
cd PokemonRedExperiments
git checkout pipeline-v2
cd Pipeline_V2

# Create conda environment
conda env create -f environment.yml
conda activate poke_env
```

### 2. Setup Game Data

⚠️ **Required:** Pokemon Red ROM (not included due to copyright)

```bash
# Place PokemonRed.gb in data/
cp /path/to/PokemonRed.gb data/

# Verify setup
ls -lh data/
# Should show:
#   - PokemonRed.gb (1.0M)
#   - init.state (143K) ✓ included
```

### 3. Run Training

```bash
# Train v1 (PPO + Frame Stacking)
python -m pipeline_v2.train --variant v1 --config configs/v1.yaml

# Train v4 (RecurrentPPOLD - Lambda Discrepancy)
python -m pipeline_v2.train --variant v4 --config configs/v4.yaml

# Resume training from checkpoint
python -m pipeline_v2.train --variant v4 --config configs/v4.yaml \
    --resume experiments/v4/20250102_120000/checkpoints/model_latest.zip
```

### 4. Monitor Training

```bash
# TensorBoard
tensorboard --logdir experiments/v4/20250102_120000/tensorboard

# Check training logs
tail -f experiments/v4/20250102_120000/json_logs/stats_*.json
```

---

## 🧪 Testing

All variants tested and verified working:

```bash
# Test environment observation space
python test/debug_observation_space.py

# Quick training test (2 min)
export KMP_DUPLICATE_LIB_OK=TRUE
timeout 120 python -m pipeline_v2.train --variant v1 --config configs/v1.yaml
```

**Test Results:**

| Variant | Status | FPS | Loss | Notes |
|---------|--------|-----|------|-------|
| v1 | ✅ Pass | 1885 | -0.0211 | Frame stacking works |
| v2 | ✅ Pass | ~1800 | -0.0205 | Single frame works |
| v3 | ✅ Pass | ~1700 | -0.0198 | LSTM state management works |
| v4 | ✅ Pass | ~1600 | LD: 0.0584 | Lambda Discrepancy integrated |

---

## 📊 Configuration

Each variant has a YAML config file in `configs/`. Key parameters:

```yaml
# configs/v1.yaml
num_cpu: 32              # Parallel workers
max_steps: 163840        # Episode length
n_steps: 2048            # Rollout length
batch_size: 512
gamma: 0.997
ent_coef: 0.01
reward_scale: 0.5
explore_weight: 0.25
```

**Hyperparameter Philosophy:**
- **Hardware-dependent**: `num_cpu`, `batch_size` (adapt to your GPU)
- **Preserved from original**: `gamma`, `ent_coef`, `reward_scale` (scientific consistency)
- **Variant-specific**: `ld_coef` (v4 only)

---

## 🏗️ Architecture Details

### Environment Refactoring

**Before:**
- 3 environment files: `red_gym_env_v2.py` (608 lines), `red_gym_env_v2_adapted.py` (701 lines), `red_gym_env_lstm.py` (695 lines)
- **Total: ~2004 lines** with ~400 lines duplicated

**After:**
- Base class + 3 variants: ~949 lines total
- **53% code reduction**
- ✅ Exact functionality preserved (verified via observation space tests)

**Key Learnings:**
- Observation space compatibility is **critical** - neural networks expect exact same structure
- Action space must match original (7 PRESS actions, release handled separately)
- Map visualization requires local 48x48 crop (not full explore_map)

### Lambda Discrepancy (v4)

Implementation of Lambda Discrepancy auxiliary loss for improved credit assignment in recurrent policies.

**Key Components:**
- `RecurrentPPOLD` algorithm in `ppo_lambda_discrepancy.py`
- MC-Return vs TD-Lambda discrepancy loss
- Configurable `ld_coef` hyperparameter

See [lambda_discrepancy_paper_prenner.pdf](../Pipeline/lambda_discrepancy_paper_prenner.pdf) for theoretical foundation.

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [PIPELINE_V2_SPECIFICATION.md](docs/PIPELINE_V2_SPECIFICATION.md) | Complete architecture specification |
| [ENVIRONMENT_ANALYSIS.md](docs/ENVIRONMENT_ANALYSIS.md) | Original environment comparison |
| [REFACTORING_STATUS.md](docs/REFACTORING_STATUS.md) | Refactoring progress & debugging |
| [data/README.md](data/README.md) | Data setup instructions |

---

## 🔧 Development

### Adding New Variants

1. Create new environment in `src/environments/`
2. Extend `BaseRedGymEnv` abstract class
3. Implement required methods:
   - `_build_observation_space()`
   - `_get_obs()`
   - `_init_frame_stack()`
   - `_update_frame_stack()`
   - `_get_action_observation()`
   - `_update_action_history()`

4. Add config in `configs/`
5. Test with `test/debug_observation_space.py`

### Code Quality Standards

- ✅ No code duplication (DRY principle)
- ✅ Abstract base classes for common logic
- ✅ Comprehensive documentation
- ✅ Preserve original functionality
- ✅ Professional git workflow

---

## 🌳 Repository Branches

| Branch | Purpose | Status |
|--------|---------|--------|
| `master` | Original experiments (Peter Whidden fork) | Archived |
| `restructure` | Pipeline prototype development | Legacy |
| **`pipeline-v2`** | Production-ready implementation | ✅ **Active** |

---

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@mastersthesis{prenner2025lambda,
  title={Lambda Discrepancy in Recurrent Reinforcement Learning},
  author={Prenner, Martin},
  year={2025},
  school={FH St. Pölten}
}
```

---

## 📄 License

See [LICENSE](../LICENSE)

---

## 🙏 Acknowledgments

- **Peter Whidden**: Original Pokemon Red RL environment ([PokemonRedExperiments](https://github.com/PWhiddy/PokemonRedExperiments))
- **Stable-Baselines3**: RL algorithms framework
- **PyBoy**: GameBoy emulator for Python

---

## 📧 Contact

**Martin Prenner**
Master's Thesis - FH St. Pölten
GitHub: [@prennerm](https://github.com/prennerm)

---

**Status:** ✅ Environment Refactoring Complete (2025-01-02)
**Next:** Modularize training logic & callbacks
