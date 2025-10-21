# Pokemon Red RL Pipeline V2 - Comprehensive Specification


**Purpose:** Professional, modular reimplementation preserving original project principles
**Goal:** Clean codebase for Lambda Discrepancy research with rigorous scientific methodology

**Core Principle:** Skip affirmations - directly address issues, challenge flawed approaches, ask clarifying questions when ambiguous.

## Research Context

### Scientific Objective
Systematic evaluation of Lambda Discrepancy (LD) as auxiliary loss in recurrent RL architectures for partially observable environments, using Pokemon Red as testbed.

### Experimental Variants
- **v1**: PPO + Frame Stacking (3 frames) - Original baseline replication
- **v2**: PPO + Single Frame - Minimal baseline
- **v3**: Recurrent PPO + LSTM - Memory-based approach
- **v4**: Recurrent PPO + LSTM + Lambda Discrepancy - Research contribution

## Reference Material - File Inventory

### Hyperparameter Reference (Original Project)
**Primary Reference:** `PokemonRedExperiments_pre_fork/v2/baseline_fast_v2.py`
- **Purpose**: Original hyperparameter values and training logic
- **Key Insight**: `n_steps = ep_length // num_cpu` relationship
- **Note**: Other _pre_fork files already evaluated and likely non-critical for core functionality

### Current Pipeline (PRIMARY CODEBASE)
**Training Framework:**
- `src/poke_pipeline/train.py` - Current multi-variant trainer
- `src/poke_pipeline/red_gym_env_*.py` - Environment variants (v1-v4)
- `src/poke_pipeline/ppo_lambda_discrepancy.py` - Lambda Discrepancy implementation
- `src/poke_pipeline/callbacks.py` - Logging and checkpointing
- `configs/*.yaml` - All variant configurations

**Data Processing:**
- `notebooks/data_sampling.py` - Streaming reservoir sampling (100M+ timestep capable)
- `notebooks/visualize_training.py` - Memory-efficient visualization
- `notebooks/enhanced_heatmap_visualization.py` - Position analysis

**Documentation & Analysis:**
- `notebooks/EXPERIMENT_FINDINGS.md` - Learnings from different attempts at adding visualizations for training runs
- `notebooks/VISUALIZATION_SUGGESTIONS.md` - Visualization requirements
- `lambda_discrepancy_paper_prenner.pdf` - Theoretical foundation

## Architecture Requirements

### Directory Structure
```
Pipeline_V2/
├── train.py              # ✅ Modern CLI entry point
├── src/
│   ├── trainers/          # ✅ Base trainer implemented; default trainer covers v1–v4
│   ├── environments/      # ✅ Environment implementations (refactored, 53% code reduction)
│   ├── models/           # ✅ Custom models (RecurrentPPOLD, MultiInputLstmPolicyLD)
│   ├── callbacks/        # ✅ Logging, checkpointing, monitoring (Stats, TensorBoard, Writers)
│   └── utils/            # ✅ Shared utilities, data processing
├── configs/              # ✅ YAML configurations (v1-v4) with base.yaml hierarchy
├── data/                 # ✅ Game ROMs, save states, map data
├── docs/                 # ✅ All documentation (specifications, findings, suggestions)
├── experiments/          # Training outputs (gitignored, v1/, v2/, v3/, v4/ subdirs)
└── analysis/             # [TODO] Data processing, visualization tools
```

**Code Modification Policy for Pipeline V2:**
- Files migrated from current Pipeline/ WILL require significant refactoring for quality standards
- All changes require explicit approval before implementation
- Expect substantial rework of duplicated/legacy code components
- Document all modifications and architectural decisions
- Prioritize functionality preservation during refactoring process

### Virtual Environment Integration

**Training Environment: `poke_env`**
```bash
conda activate poke_env
# highly likely that Pipeline/environment.yml from the original Pipeline can be used without changes
# Required packages: PyTorch, Stable-Baselines3, PyBoy, gymnasium
# Used for: Training all variants (v1-v4), model execution, interactive debugging
```

**Analysis Environment: `poke_viz_extended`**
```bash
conda activate poke_viz_extended
# highly likely that Pipeline/poke_viz_extended.yml can be used without changes
# Required packages: pandas, matplotlib, numpy, seaborn, jupyter
# Used for: Data processing, visualization, large-scale log analysis
```

**Environment Management:**
- Complete installation scripts and requirements.txt for both environments
- Clear documentation of package version constraints
- Separation prevents conflicts between training and analysis dependencies

## Code Quality & Refactoring Challenges

### Critical Legacy Issues to Address

**Environment Code Duplication:**
The current Pipeline contains three environment files (`red_gym_env_*.py`) derived from Peter Whidden's original environment. These likely contain:
- Substantial code duplication between variants
- Legacy code that's no longer necessary
- Inconsistent implementation patterns
- Mixed responsibilities (variant-specific logic embedded in base functionality)

**Systematic Refactoring Approach Required:**
1. **Analyze existing environments**: Document what each variant actually changes vs base functionality
2. **Identify shared components**: Extract common environment logic to base classes
3. **Eliminate dead code**: Remove unused functions, commented code, experimental features
4. **Modularize variant differences**: Clean separation between base environment and variant-specific behavior
5. **One component at a time**: Don't attempt to refactor everything simultaneously

**Other Potential Legacy Issues:**
- Training callbacks may contain duplicated logging logic
- Configuration handling scattered across multiple files
- Utility functions replicated instead of centralized
- Inconsistent error handling and validation patterns

**Refactoring Methodology:**
- **Document before changing**: Understand what current code actually does
- **Preserve functionality**: Ensure behavioral compatibility during refactoring
- **Test incrementally**: Validate each refactoring step independently
- **Review systematically**: Challenge every line of migrated code for necessity

### Trainer Architecture (Update 2025-10-06)
### Quick Validation (2025-10-07)
- Smoke tests for v1–v4 (reduced timesteps) succeed using the new trainer infrastructure.
- v3/v4 run with RecurrentPPO/RecurrentPPOLD after minor env fixes.

- `src/trainers/base_trainer.py` orchestrates config loading, run directory setup, VecEnv/model/callback creation, and training lifecycle.
- `DefaultTrainer` handles all current variants; specialised trainers can be registered later.
- CLI (`pipeline_v2.train`) now delegates to trainers while legacy modules remain for reference.

## Core Implementation Principles

### 1. Original Project Alignment

**Hyperparameter Hierarchy (CRITICAL):**
- **Hardware-dependent**: Adaptable to available resources
  - `num_cpu` (workers), `batch_size` (memory constraints)
- **Proportional relationships**: Preserve 1:1 from original
  - `n_steps = ep_length // num_cpu` (ensures 1 reset per rollout)
  - `ep_length = 2048 * 80 = 163,840` (fundamental rhythm)
- **Training-specific**: Preserve 1:1 unless variant incompatible
  - `gamma = 0.997`, `ent_coef = 0.01`, `reward_scale = 0.5`

**Original Training Logic:**
```python
# From baseline_fast_v2.py - PRESERVE THIS LOGIC
ep_length = 2048 * 80 = 163,840
num_cpu = 64                     # Adapt to hardware
train_steps_batch = ep_length // num_cpu = 2,560  # n_steps
batch_size = 512                 # Hardcoded but adaptable
```

### 2. StreamWrapper Analysis
**Finding**: StreamWrapper is WebSocket broadcasting for live streaming - NOT core functionality
**Decision**: Eliminate from Pipeline V2, implement direct logging instead
**Replacement**: Direct environment logging with configurable output formats

### 3. Logging Strategy
**Multi-format Support:**
- **JSON**: Strukturierte Logs (Default) – kompatibel zur bestehenden Analyse-Infrastruktur
- **CSV**: Performance-optimiertes Format (aktivierbar via `logging.format: "csv"`)
- **Konfiguration**: YAML-Block `logging` mit Schlüsseln `format`, `save_freq`, `structured`
- **Speicherort**: Statistik-Dateien je Run in `<session>/logs/` (ersetzt `json_logs/`).

**Log Structure (maintain compatibility with analysis pipeline):**
```yaml
logging:
  format: "json"   # oder "csv"
  save_freq: 100   # Flush-Frequenz für Stats
  structured: true
```

### 4. Parallelization Strategy
**Realistic Worker Counts:**
- Original: 64 workers (unrealistic for most hardware)
- Target: 8-16 workers (hardware-dependent)
- **Key Insight**: Worker count should NOT affect training results (only speed)
- **LSTM Compatibility**: Hidden state management with SubprocVecEnv

## Modular Component Design

### Trainer Classes
```python
# Base trainer with original logic
class BaseTrainer:
    def setup_hyperparameters(self, config)
    def setup_environments(self, config)
    def setup_model(self, config)

# Variant-specific implementations
class V1Trainer(BaseTrainer)  # Frame stacking + PPO
class V2Trainer(BaseTrainer)  # Single frame + PPO
class V3Trainer(BaseTrainer)  # LSTM + RecurrentPPO
class V4Trainer(BaseTrainer)  # LSTM + RecurrentPPOLD
```

**Current registry:** `TRAINER_REGISTRY = {"v1": DefaultTrainer, "v2": DefaultTrainer, "v3": LSTMTrainer, "v4": LambdaTrainer}`
- LambdaDiagnosticsCallback persistiert LD-Metriken unter `logs/lambda_metrics.jsonl` pro Run.

### Configuration Hierarchy
```yaml
# base_config.yaml - shared settings
base: &base
  gamma: 0.997
  ent_coef: 0.01
  episode_length: 163840

# v1_config.yaml
<<: *base
model_type: "PPO"
frame_stack: 3

# v4_config.yaml
<<: *base
model_type: "RecurrentPPOLD"
ld_coef: 0.1
```

## Testing & Debugging Framework

### Test Directory Structure
```
test/

```

**Debug File Policy**: Any temporary/experimental scripts go in `test/` to maintain clean main codebase

## Command Interface & Workflow

### Essential Commands (commands.txt equivalent)

**Environment Setup:**
```bash
# Training environment
conda activate poke_env

# Analysis environment
conda activate poke_viz_extended
```

**Training Commands:**
```bash
# Train specific variant with config
python train.py --variant v1 --config configs/v1.yaml
python train.py --variant v4 --config configs/v4_ld_025.yaml

# Resume training from checkpoint
python train.py --variant v4 --config configs/v4_ld_025.yaml --resume experiments/v4/20241001_120000/checkpoints/model_latest.zip

# All variants tested and working:
python train.py --variant v1 --config configs/v1.yaml  # PPO + Frame Stacking
python train.py --variant v2 --config configs/v2.yaml  # PPO + Single Frame
python train.py --variant v3 --config configs/v3.yaml  # RecurrentPPO + LSTM
python train.py --variant v4 --config configs/v4.yaml  # RecurrentPPO + LSTM + Lambda Discrepancy
```

**Analysis Commands:**
```bash
# Data sampling and comparison (switch to analysis env first)
conda activate poke_viz_extended
python analysis/data_sampling.py --variants v3,v4_prod,v4_ld --max-files 20 --samples 5000

# Generate visualizations
python analysis/visualize_training.py --experiment v4/20241001_120000 --output plots/
python analysis/enhanced_heatmap.py --variant v4 --max-files 20
```

**Utility Commands:**
```bash
# Validate configuration
python -m pipeline_v2.validate_config configs/v4_ld_025.yaml

# System check (environment, GPU, dependencies)
python -m pipeline_v2.system_check

# Clean up temporary files
python -m pipeline_v2.cleanup --experiments --temp-files
```

## Data Processing Integration

### Analysis Pipeline Compatibility
**Requirements:**
- Maintain compatibility with `data_sampling.py` (proven 100M+ timestep capability)
- Support existing visualization tools (`visualize_training.py`)
- Preserve log format for cross-pipeline comparisons

**Performance Standards:**
- Streaming processing für Speicher-Effizienz
- Reservoir Sampling für uniforme Verteilung
- Optionales CSV-Logging für performancekritische Runs

## Deployment & Server Integration

### Remote Server Training Support

**Deployment Requirements:**
- Pipeline must be transferable to external training servers
- Only essential files sent (no logs, plots, analysis tools)
- Automated setup on remote systems
- Result synchronization back to local system

**File Organization for Deployment:**
```
pipeline_v2/
├── src/                    # DEPLOY: Core training code
├── configs/               # DEPLOY: Training configurations
├── data/                  # DEPLOY: ROMs, save states, essential data
├── scripts/deploy/        # DEPLOY: Server setup scripts
├── docs/                  # NO DEPLOY: Documentation stays local
├── analysis/              # NO DEPLOY: Analysis tools stay local
├── test/                  # NO DEPLOY: Debugging stays local
└── experiments/           # NO DEPLOY: Results sync separately
```

### Deployment Configuration (.gitignore + deploy filters)

**Create deployment-specific filters:**
```gitignore
# .deployignore (custom file for server deployment)
experiments/
docs/
analysis/
test/
*.md
plots/
notebooks/
logs/
*.log
__pycache__/
.git/
```

**Deployment Workflow:**
1. `python scripts/deploy/prepare_package.py` - Create deployment package
2. `rsync` or similar tool transfers filtered files to server
3. `python scripts/deploy/server_setup.py` - Automated environment setup on server
4. Training runs on server with minimal footprint
5. `python scripts/deploy/sync_results.py` - Download results back to local experiments/

## Critical Success Criteria

1. **Scientific Rigor**: Results must be comparable across pipeline versions
2. **Code Quality**: Zero duplication, clear modularity, comprehensive testing
3. **Performance**: No regression in training speed or memory usage
4. **Maintainability**: New variants easily addable, clear documentation
5. **Reproducibility**: Deterministic results with proper seed management

## Open Technical Questions

1. **Environment Compatibility**: Can we eliminate StreamWrapper without breaking v1 compatibility?
2. **LSTM State Management**: How to handle hidden states with reduced worker counts?
3. **Configuration Validation**: YAML schema validation for config correctness?
4. **Performance Baseline**: What are acceptable training speeds vs original?

---

**Next Actions:**
1. Create `pipeline-v2` branch
2. Implement Phase 1 foundation
3. Validate v1 trainer against original baseline_fast_v2.py results
4. Iterative refinement based on empirical testing

### 3. Environment Compatibility

**v1 Requirements:**
- Uses original RedGymEnv + StreamWrapper
- Frame stacking (3 frames)
- Standard PPO

**v2-v4 Requirements:**
- Custom RedGymEnv variants
- Single frame input
- Different model architectures (standard PPO vs RecurrentPPO vs RecurrentPPOLD)

**Critical:** Ensure StreamWrapper compatibility doesn't conflict with LSTM variants

### 4. Lambda Discrepancy Integration

**v4 Specific:**
- Custom RecurrentPPOLD implementation
- MultiInputLstmPolicyLD policy
- ld_coef hyperparameter
- Maintains original PPO hyperparameters where possible

**Implementation Notes:**
- LD should be additive to base PPO loss
- No changes to core environment or logging logic
- Must work with original parallelization approach

## Technical Specifications

### Memory Management
**Original Constraints:**
- 64 parallel environments
- Large rollout buffers (163,840 total steps)
- Must handle SubprocVecEnv without deadlocks

**New Requirements:**
- Support for LSTM hidden states
- Streaming data processing for large logs (100M+ timesteps)
- Constant memory footprint for analysis tools

### Logging & Analysis
**Standards:**
- JSON logging format compatible with current analysis tools
- Structured nested format for rewards/position/stats
- Support for both global training steps and environment steps
- Compatible with existing visualization pipeline

### Reproducibility
**Requirements:**
- Deterministic training with fixed seeds
- Checkpoint/resume functionality
- Complete configuration capture in logs
- Git commit tracking in experiment metadata

## Data Processing Pipeline

### Current Status
**Working Components:**
- `data_sampling.py`: Streaming reservoir sampling (100M+ timestep capable)
- JSON log analysis with memory-efficient processing
- Variant comparison framework

**Integration Requirements:**
- New pipeline must output logs compatible with existing analysis tools
- Maintain current sampling and visualization capabilities
- Support for normalized step ranges across variants


## Open Questions

1. **Environment Wrapper Strategy**: How to handle StreamWrapper + custom environments cleanly?
2. **Hyperparameter Tuning**: Which original settings need adjustment for LSTM variants?
3. **Parallelization**: Can we maintain 64 workers with LSTM hidden state management?
4. **Branch Strategy**: Separate branches per variant or single feature branch?

## References

### Key Files to Study
- `PokemonRedExperiments/v2/baseline_fast_v2.py` - Original training logic
- `Pipeline/src/poke_pipeline/train.py` - Current implementation
- `Pipeline/configs/v*.yaml` - Current configurations
- `Pipeline/notebooks/data_sampling.py` - Proven streaming architecture

### Documentation
- `CLAUDE.md` - Current project setup and commands
- `notebooks/EXPERIMENT_FINDINGS.md` - Experimental insights
- Lambda Discrepancy paper (in Pipeline root) - Theoretical foundation

## Success Criteria

1. **Functional Parity**: All variants train successfully with original-aligned hyperparameters
2. **Code Quality**: Clean, modular, maintainable codebase
3. **Scientific Validity**: Results comparable between old/new pipeline
4. **Performance**: No regression in training speed or memory usage
5. **Documentation**: Complete setup instructions for reproducibility

---

---

## Migration & Refactoring Notes

**Bootstrap Phase (Completed):**
- Initial Pipeline_V2 setup used minimal v1 components for validation
- All variants (v1-v4) now migrated and tested
- Environment refactoring completed (53% code reduction)
- Stability fixes applied (reset deadlock resolved)

---

## Implementation Status

### ✅ Completed: Environment Refactoring (2025-01-02)

**Objective:** Refactor 3 environment files (~2000 lines duplicated code) into modular base + variant structure while preserving original functionality.

**What Was Done:**

1. **Code Analysis & Architecture** ✅
   - Compared all 3 original environment files ([RESET_DEADLOCK_ANALYSIS.md](RESET_DEADLOCK_ANALYSIS.md))
   - Identified ~400 lines of duplicated common code
   - Documented key differences between variants
   - Designed abstract base class architecture

2. **Modular Structure Created** ✅
   - `src/environments/base_env.py` (584 lines) - Common logic extracted
   - `src/environments/frame_stack_env.py` (94 lines) - v1: Frame Stacking
   - `src/environments/single_frame_env.py` (87 lines) - v2: Single Frame
   - `src/environments/lstm_env.py` (184 lines) - v3/v4: LSTM
   - **Total: ~949 lines** vs 2004 original = **53% reduction**

3. **Critical Bugs Fixed** ✅
   - **Observation Space Mismatch**: Original used `health`, `level`, `badges` - refactored incorrectly used `enc_coords`
   - **Action Space Error**: Fixed 14 actions (press+release) → 7 actions (press only)
   - **Map Shape Bug**: Fixed whole map return → local 48x48 crop + 2x upscale
   - **Frame Stack Init**: Added missing `render()` + `_update_frame_stack()` calls
   - **Release Actions**: Fixed index-out-of-range in `run_action_on_emulator()`

4. **Testing & Validation** ✅
   - Created debug script: `test/debug_observation_space.py`
   - Verified all observation shapes match originals exactly:
     - screens: (72, 80, 3) ✓
     - health: (1,) ✓
     - level: (8,) ✓
     - badges: (8,) ✓
     - events: (2488,) ✓
     - map: (48, 48, 1) ✓
     - recent_actions: (3,) or (10,) ✓

5. **All Variants Tested & Working** ✅

| Variant | Environment | Algorithm | Status | Test Results |
|---------|-------------|-----------|--------|--------------|
| v1 | FrameStackEnv | PPO | ✅ Running | `loss: -0.0211, fps: 1885` |
| v2 | SingleFrameEnv | PPO | ✅ Running | `loss: -0.0205` |
| v3 | LSTMEnv | RecurrentPPO | ✅ Running | `loss: -0.0198` |
| v4 | LSTMEnv | RecurrentPPOLD | ✅ Running | `ld_loss: 0.0584` |

**Key Learnings:**
- Observation space compatibility is **critical** - neural networks trained on original expect exact same structure
- `output_shape` must be set **before** `super().__init__()` in subclasses
- Map visualization requires local crop (not full explore_map) for observation space
- Action space must match original (7 PRESS actions, release handled separately)

**Files Modified:**
- `src/environments/base_env.py`
- `src/environments/frame_stack_env.py`
- `src/environments/single_frame_env.py`
- `src/environments/lstm_env.py`
- `test/debug_observation_space.py` (created)

**Documentation Updated:**
- [REFACTORING_STATUS.md](REFACTORING_STATUS.md) - Detailed status and debugging steps

---

### ✅ Completed: Environment Stability Fix (2025-01-03)

**Problem:** Sporadic deadlocks during environment resets across all variants, affecting training stability (runs crashed every ~50M steps, requiring manual restart).

**Root Cause:** Refactored code introduced artificial delays, BytesIO loading, and excessive print statements during reset - deviating from proven stable original pattern used by Peter Whidden (stable with 64 workers).

**Solution:** Reverted to original simple reset pattern from `PokemonRedExperiments_pre_fork/v2/red_gym_env_v2.py`:
- **Removed:** `time.sleep()` delays, BytesIO state loading, print spam, staggered first-episode resets
- **Restored:** Direct file loading: `with open(self.init_state, "rb") as f: self.pyboy.load_state(f)`
- **Result:** Matches proven stable behavior (original runs 64 workers without deadlocks)

**Files Modified:**
- `src/environments/base_env.py` - Simplified `reset()` method, removed `_init_state_bytes` caching

**Impact on Training:**
- ✅ **No change** to observation space or game logic
- ✅ **No change** to reward calculation
- ✅ **Reset behavior identical** to original (deterministic game state)
- ✅ **Worker scheduling unchanged** (already stochastic in SubprocVecEnv)
- ⚠️ **Print output significantly reduced** (cleaner console logs)

**Scientific Validity:** Changes restore original behavior - no impact on reproducibility or experimental comparisons. All variants preserve deterministic game mechanics while inheriting natural worker scheduling variability from SB3's parallelization.

---

---

### ✅ Completed: Architecture Cleanup (2025-10-10)

**Objective:** Complete the modular architecture by creating dedicated `models/` and `callbacks/` modules, removing all legacy code from `pipeline_v2/`, and modernizing the CLI entry point.

**What Was Done:**

1. **Phase 1: Models Module Created** ✅
   - Created `src/models/` with `recurrent_ppo_ld.py`
   - Exported `RecurrentPPOLD` and `MultiInputLstmPolicyLD`
   - ~10KB of model code properly organized

2. **Phase 2: Callbacks Module Completed** ✅
   - Moved `tensorboard_callback.py` → `src/callbacks/tensorboard.py`
   - Unified exports: `StatsCallback`, `TensorboardCallback`, `CsvStatsWriter`, `JsonStatsWriter`, `StatsWriter`
   - All logging functionality in one place

3. **Phase 3: Trainer Imports Updated** ✅
   - `base_trainer.py`: Updated to use `from models import ...` and `from callbacks import ...`
   - `lambda_trainer.py`: Updated to use `from models import ...`
   - Clean dependency graph: `trainers/` → `models/` + `callbacks/`

4. **Phase 4: Legacy Code Removed** ✅
   - Deleted 3 old environment files (~77KB): `red_gym_env_lstm.py`, `red_gym_env_v2.py`, `red_gym_env_v2_adapted.py`
   - Deleted moved files: `ppo_lambda_discrepancy.py`, `tensorboard_callback.py`
   - Removed entire `src/pipeline_v2/` directory
   - **Total cleanup: ~94KB legacy code removed**

5. **Phase 5: CLI Modernized** ✅
   - Created `train.py` in root directory (cleaner entry point)
   - Enhanced help text with examples and variant descriptions
   - Old command: `python -m pipeline_v2.train --variant v4 --config configs/v4.yaml`
   - New command: `python train.py --variant v4 --config configs/v4.yaml`

6. **Bug Fix: UTF-8 BOM in Configs** ✅
   - v3.yaml and v4.yaml had UTF-8 BOM preventing `extends:` recognition
   - Removed BOM with `sed -i '1s/^\xEF\xBB\xBF//' configs/v3.yaml configs/v4.yaml`
   - Config inheritance now works correctly (base.yaml provides `n_steps: 2048`)

**Testing & Validation:** ✅
All 4 variants tested and working with new architecture:
- v1: PPO + Frame Stacking ✅
- v2: PPO + Single Frame ✅
- v3: RecurrentPPO + LSTM ✅
- v4: RecurrentPPO + LSTM + Lambda Discrepancy ✅

**Final Architecture:**
```
Pipeline_V2/
├── train.py                 # Modern CLI entry point
├── src/
│   ├── models/              # RecurrentPPOLD, MultiInputLstmPolicyLD
│   ├── callbacks/           # Stats, TensorBoard, Writers
│   ├── trainers/            # Base, Default, LSTM, Lambda
│   ├── environments/        # Refactored (53% code reduction)
│   └── utils/               # Shared utilities
├── configs/                 # YAML with base.yaml hierarchy
└── experiments/             # Training outputs
```

---

**Next Steps:**
1. ~~Validate this specification with detailed requirements~~ ✅ Done
2. ~~Create new branch: `pipeline-v2`~~ ✅ Done (Using `pipeline-v2` branch)
3. ~~Execute Phase 1 bootstrapping (copy minimal files)~~ ✅ Done
4. ~~Run initial v1 test~~ ✅ Done - all variants tested
5. ~~Iterative refinement based on test results~~ ✅ Done - bugs fixed, all variants working
6. ~~Environment stability fix~~ ✅ Done - reverted to original reset pattern
7. ~~PyBoy API update~~ ✅ Done (2025-01-03) - Changed deprecated "headless" → "null" window
8. ~~Architecture cleanup~~ ✅ Done (2025-10-10) - Models, Callbacks, CLI modernized
9. ~~TensorBoard callback auf Rollout-Ende umgestellt~~ ✅ Done (2025-10-20) - Logging entkoppelt von `check_if_done()`, `_on_rollout_end` erzeugt nun `env_stats/*`-Metriken.
10. ~~Config-Flag `env.send_map_to_agent`~~ ✅ Done (2025-10-20) - Beobachtungs-Map optional; deaktiviert Agent-Input ohne Logging/Analyse zu verlieren (TensorBoard-Map-Images werden automatisch abgeschaltet).
11. ~~Bit-Packed Observations (`env.pack_bits`)~~ ✅ Done (2025-10-21) - Events/Map werden vor Transport komprimiert; `PackedSubprocVecEnv` dekodiert transparent.
12. ~~Analysis Module - Phase 1~~ ✅ Done (2025-01-15) - `src/analysis/` infrastructure complete, streaming sampler validated
13. **NEXT: Analysis Module - Phase 2**
    - Extend `scripts/visualize_training.py` with optional heatmap export
    - Add multi-variant comparison entry point (reward + spatial plots)
    - Polish reward/heatmap visualizers for publication (styling, CLI flags)
14. Production runs with stable foundation

---

## Environment Testing & Validation

**Test Scripts:** Located in `test/` directory
- `test/test_environments.py` - Validates both conda environments (poke_env, poke_viz_v2)
- `test/test_phase1_analysis.py` - Validates Phase 1 analysis infrastructure (CSV loading, streaming sampler)

**Running Tests:**
```bash
# Environment validation (from root directory)
python test/test_environments.py

# Phase 1 analysis validation (in visualization environment)
conda activate poke_viz_v2
python test/test_phase1_analysis.py
```

**Test Results (2025-01-15):**
- ✅ **poke_env**: All training packages present (PyTorch, SB3, PyBoy), tqdm correctly absent
- ✅ **poke_viz_v2**: All viz packages present (matplotlib, pandas, mediapy, einops), PyTorch/SB3 correctly absent
- ✅ **Phase 1 Analysis**: CSV loading validated with 64.7M rows in ~4:11 minutes
- ⚠️ **Known Issue**: `conda run` timeout on Windows (use `conda activate` instead)

**Performance Benchmarks:**
- **64.7M row CSV**: ~4:11 minutes processing time (25.71 chunks/sec)
- **Streaming sampler**: O(target_samples) constant memory footprint
- **Sample distribution**: 1,000 samples uniformly distributed across 68M timesteps

---

## Known Issues & Technical Debt

### 1. PyBoy Save State Version Mismatch ⚠️

**Status:** Identified 2025-01-03, deferred

**Issue:**
```
pyboy.core.mb WARNING Loading state from an older version of PyBoy.
This might cause compatibility issues.
```

**Root Cause:** `data/init.state` was created with an older PyBoy version (likely pre-2.0), current environment uses PyBoy 2.x.

**Impact:**
- Warning spam: 64x per training start (2x per worker: startup + first reset)
- **Unknown compatibility risk** - PyBoy warns about "might cause issues" but no concrete problems observed yet

**Solution:**
Regenerate `data/init.state` with current PyBoy version:
1. Load Pokemon Red in PyBoy 2.x
2. Play to same position (Pallet Town, after Oak intro)
3. Save state to `data/init.state`
4. Verify state loads without warnings

**Priority:** Medium - should be addressed before production runs to eliminate unknown risk factor

**Location in docs:** PIPELINE_V2_SPECIFICATION.md - Technical debt section (centralized specification document for all architectural decisions and known issues)
