# Pipeline V2 Documentation

This directory contains all documentation for the Pokemon Red RL Pipeline V2 project.

## Files

### Specifications
- **[PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md)** - Complete project specification including architecture requirements, implementation principles, and deployment strategy
- **[HYPERPARAMETER_NOTES.md](HYPERPARAMETER_NOTES.md)** - Hyperparameter configuration rationale, deviations from baseline, scientific transparency notes

### Status & Analysis
- **[REFACTORING_STATUS.md](REFACTORING_STATUS.md)** - Current refactoring status and completed milestones:
  - Environment refactoring (53% code reduction) ✅
  - Trainer framework implementation ✅
  - Architecture cleanup (models, callbacks, CLI) ✅
  - All variants tested and working ✅

- **[RESET_DEADLOCK_ANALYSIS.md](RESET_DEADLOCK_ANALYSIS.md)** - Historical investigation log for reset deadlock (resolved)

## Quick Start for New Context

If starting a fresh conversation about this project:

1. Read [PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md) for overall architecture
2. Read [REFACTORING_STATUS.md](REFACTORING_STATUS.md) for current status
3. Check "Next Steps" section for upcoming work

## Current Architecture (2025-10-10)

```
Pipeline_V2/
├── train.py              # Modern CLI entry point
├── src/
│   ├── models/           # RecurrentPPOLD, MultiInputLstmPolicyLD
│   ├── callbacks/        # Stats, TensorBoard, Writers
│   ├── trainers/         # Base, Default, LSTM, Lambda
│   ├── environments/     # Refactored (53% reduction)
│   └── utils/            # Shared utilities
├── configs/              # YAML with base.yaml hierarchy
└── experiments/          # Training outputs
```

## Training Commands

```bash
# All variants working ✅
python train.py --variant v1 --config configs/v1.yaml  # PPO + Frame Stacking
python train.py --variant v2 --config configs/v2.yaml  # PPO + Single Frame
python train.py --variant v3 --config configs/v3.yaml  # RecurrentPPO + LSTM
python train.py --variant v4 --config configs/v4.yaml  # RecurrentPPO + LSTM + LD
```

## Hyperparameter Configuration

All variants inherit from [configs/base.yaml](../configs/base.yaml) with Peter Whidden's formulas:

**Dynamic n_steps Calculation:**
- Formula: `n_steps = max_steps // num_cpu`
- Implementation: `BaseTrainer._apply_n_steps_formula()` calculates at runtime
- Purpose: Ensures exactly 1 environment reset per rollout buffer
- Example: `163840 // 16 = 10240` steps per worker

**Training Duration:**
- `save_freq: 81920` - Checkpoint every half-episode (crash recovery)
- `total_timesteps: 1e8` - Scaled from Whidden's original (163840 × 64 × 10000)

**PPO Hyperparameters:**
- `n_epochs: 1` - Whidden's baseline (can increase to 3 for better sample efficiency with limited compute)
- `gamma: 0.997`, `ent_coef: 0.01` - Preserved from original

## Next Steps

- **Analysis Module**: Create `analysis/` directory and migrate visualization tools
- **Production Runs**: Execute long training runs with stable architecture
- **Technical Debt**: Regenerate `data/init.state` with PyBoy 2.x (eliminate version warnings)
