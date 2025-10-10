# Pipeline V2 Documentation

This directory contains all documentation for the Pokemon Red RL Pipeline V2 project.

## Files

### Specifications
- **[PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md)** - Complete project specification including architecture requirements, implementation principles, and deployment strategy

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

## Next Steps

- **Analysis Module**: Create `analysis/` directory and migrate visualization tools
- **Production Runs**: Execute long training runs with stable architecture
