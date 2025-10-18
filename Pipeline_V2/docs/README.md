# Pipeline V2 Documentation

This directory collects all project documentation for the Pokemon Red RL Pipeline V2.

## Files

### Specifications & Guides
- **[PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md)** – Complete specification covering architecture, implementation principles, deployment strategy, testing notes, and technical debt.
- **[ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md)** – Step-by-step instructions for the dual-conda setup (poke_env for training, poke_viz_v2 for analysis).
- **[VISUALIZATION_SUGGESTIONS.md](VISUALIZATION_SUGGESTIONS.md)** – Roadmap of figures/metrics required for the Lambda Discrepancy paper.
- **[RESET_DEADLOCK_ANALYSIS.md](RESET_DEADLOCK_ANALYSIS.md)** – Summary of the former reset deadlock investigation and mitigation ideas (kept for reference).

## Quick Context Refresh
1. Start with **PIPELINE_V2_SPECIFICATION.md** for architecture and current constraints.
2. Read **VISUALIZATION_SUGGESTIONS.md** to see outstanding visualization work.
3. Check the **Next Steps** section in the specification for upcoming engineering tasks.

## Current Architecture Snapshot (2025-10-16)
`
Pipeline_V2/
├── train.py                 # CLI entry point
├── src/
│   ├── analysis/             # Streaming sampler + reward & heatmap visualizers ✓
│   ├── models/               # RecurrentPPOLD, MultiInputLstmPolicyLD
│   ├── callbacks/            # Stats, TensorBoard, writers
│   ├── trainers/             # Base, Default, LSTM, Lambda
│   ├── environments/         # Refactored game interface
│   └── utils/                # Shared utilities (map helpers, etc.)
├── configs/                 # YAML hierarchy (base + variants)
├── test/                    # Environment & analysis validation tests
└── experiments/             # Training outputs (logs, checkpoints, plots)
`

## Testing
`ash
# Validate environments
python test/test_environments.py

# Validate analysis module (run inside poke_viz_v2)
conda activate poke_viz_v2
python test/test_phase1_analysis.py
`

## Immediate Next Steps
- Integrate the heatmap output into scripts/visualize_training.py (CLI flag).
- Provide a multi-variant comparison script (e.g. reward + heatmap overlays).
- Regenerate data/init.state with the current PyBoy version to remove compatibility warnings.
