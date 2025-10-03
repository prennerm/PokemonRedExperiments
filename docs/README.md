# Pipeline V2 Documentation

This directory contains all documentation for the Pokemon Red RL Pipeline V2 project.

## Files

### Specifications
- **[PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md)** - Complete project specification including architecture requirements, implementation principles, and deployment strategy

### Analysis & Status
- **[ENVIRONMENT_ANALYSIS.md](ENVIRONMENT_ANALYSIS.md)** - Detailed comparison of the 3 original environment files, identifying common code and refactoring opportunities

- **[REFACTORING_STATUS.md](REFACTORING_STATUS.md)** - Current status of environment refactoring effort, including:
  - What has been completed
  - Current shape mismatch issue
  - Debugging steps needed
  - Next actions

## Quick Start for New Context

If starting a fresh conversation about this project:

1. Read [PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md) for overall architecture
2. Read [REFACTORING_STATUS.md](REFACTORING_STATUS.md) for current work status
3. Check the "Current Problem" section to understand what needs debugging

## Key Locations

- **Original environments** (reference): `../Pipeline/src/poke_pipeline/red_gym_env_*.py`
- **New environments** (refactored): `../src/environments/*.py`
- **Configs**: `../configs/*.yaml`
- **Training script**: `../src/pipeline_v2/train.py`
