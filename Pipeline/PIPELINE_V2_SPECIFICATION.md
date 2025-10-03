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
pipeline_v2/
├── src/
│   ├── trainers/          # Variant-specific training classes
│   ├── environments/      # Environment implementations
│   ├── models/           # Custom models (RecurrentPPOLD, policies)
│   ├── callbacks/        # Logging, checkpointing, monitoring
│   └── utils/            # Shared utilities, data processing
├── configs/              # YAML configurations (hierarchical)
├── data/                 # Game ROMs, save states, map data
├── docs/                 # All documentation (specifications, findings, suggestions)
├── test/                 # Temporary scripts, debugging tools (organic organization)
└── analysis/             # Data processing, visualization tools
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
- **JSON**: Structured logging (current standard) - nested format for analysis compatibility
- **CSV**: Performance-optimized option for large-scale training
- **Configuration**: YAML setting to choose logging format per experiment

**Log Structure (maintain compatibility with analysis pipeline):**
```yaml
logging:
  format: "json"  # or "csv"
  structured: true
  include_global_steps: true
  save_frequency: 1000
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
python -m pipeline_v2.train --variant v1 --config configs/v1_production.yaml
python -m pipeline_v2.train --variant v4 --config configs/v4_ld_025.yaml

# Resume training from checkpoint
python -m pipeline_v2.train --variant v4 --config configs/v4_ld_025.yaml --resume experiments/v4/20241001_120000/checkpoints/model_latest.zip

# Interactive agent execution (debugging)
python -m pipeline_v2.run --variant v4 --model experiments/v4/20241001_120000/checkpoints/model_latest.zip
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
- Streaming processing for memory efficiency
- Reservoir sampling for uniform data distribution
- CSV logging option for performance-critical training

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

## Pipeline V2 Bootstrapping - Minimale Funktionsfähige Struktur

### Zweck
Vor vollständiger Migration: Minimale lauffähige Pipeline aufbauen, um Fehlerquellen früh zu identifizieren und Zeitverschwendung zu vermeiden. Test mit v1-Variante, dann schrittweise Erweiterung.

### Phase 1: Minimal Viable Training Pipeline (v1 Test Only)

**Erforderliche Kernkomponenten (absolutes Minimum für v1):**

1. **Environment Setup**
   - `environment.yml` → kopieren aus `Pipeline/environment.yml`
   - Zweck: Conda environment `poke_env` erstellen

2. **Training Entry Point**
   - `src/pipeline_v2/train.py` → kopieren aus `Pipeline/src/poke_pipeline/train.py`
   - Note: Enthält Imports für alle Varianten (v1-v4), aber v1 nutzt nur Standard PPO

3. **Environment Implementation**
   - `src/pipeline_v2/red_gym_env_v2.py` → kopieren aus `Pipeline/src/poke_pipeline/red_gym_env_v2.py`
   - Zweck: Basis-Environment (v1 nutzt: `module: red_gym_env_v2`, `class: RedGymEnv`)

4. **Environment Dependencies (von red_gym_env_v2.py benötigt)**
   - `src/pipeline_v2/global_map.py` → kopieren aus `Pipeline/src/poke_pipeline/global_map.py`
   - `src/pipeline_v2/data/map_data.json` → kopieren aus `Pipeline/src/poke_pipeline/data/map_data.json`
   - `src/pipeline_v2/data/events.json` → kopieren aus `Pipeline/src/poke_pipeline/data/events.json`
   - Zweck: Koordinatentransformation + Event/Map-Daten

5. **Callback Dependencies (von train.py benötigt)**
   - `src/pipeline_v2/callbacks.py` → kopieren aus `Pipeline/src/poke_pipeline/callbacks.py`
   - `src/pipeline_v2/tensorboard_callback.py` → kopieren aus `Pipeline/src/poke_pipeline/tensorboard_callback.py`
   - Zweck: StatsCallback, TensorboardCallback für Logging

6. **Game Data**
   - `data/PokemonRed.gb` → kopieren aus `Pipeline/data/PokemonRed.gb`
   - `data/init.state` → kopieren aus `Pipeline/data/init.state`
   - Zweck: ROM + Savestate (referenziert in v1.yaml)

7. **Configuration**
   - `configs/v1.yaml` → kopieren aus `Pipeline/configs/v1.yaml`
   - Zweck: v1-Konfiguration (PPO + MultiInputPolicy)

**Wichtig - NICHT kopieren für v1-Test:**
- ❌ `ppo_lambda_discrepancy.py` - nur für v4 benötigt (RecurrentPPOLD)
- ❌ Weitere Environment-Varianten (red_gym_env_v3/v4) - erst nach v1-Erfolg
- ❌ Weitere save states (has_pokedex.state, etc.) - erst bei Bedarf

**Minimale Verzeichnisstruktur (nur v1-essentials):**
```
pipeline_v2/
├── environment.yml              # COPY
├── data/
│   ├── PokemonRed.gb           # COPY
│   └── init.state              # COPY
├── configs/
│   └── v1.yaml                 # COPY
└── src/pipeline_v2/
    ├── __init__.py             # CREATE (empty)
    ├── train.py                # COPY
    ├── red_gym_env_v2.py       # COPY
    ├── global_map.py           # COPY
    ├── callbacks.py            # COPY
    ├── tensorboard_callback.py # COPY
    └── data/
        ├── map_data.json       # COPY
        └── events.json         # COPY
```

### Phase 2: Validierung & Iteration

**Test-Kommando:**
```bash
# 1. Environment erstellen
conda env create -f environment.yml
conda activate poke_env

# 2. Training starten (kurzer Test)
python -m pipeline_v2.train --variant v1 --config configs/v1.yaml
```

**Erwartetes Resultat:**
- Training startet ohne Import-Fehler
- Environment initialisiert korrekt
- Erste Rollouts werden ausgeführt
- Checkpoints und Logs werden gespeichert

**Bei Erfolg → Nächste Schritte:**
1. Weitere Varianten (v2, v3, v4) hinzufügen
2. Code-Refactoring gemäß Quality Standards
3. Modularisierung und Cleanup
4. Deployment-Scripts ergänzen

**Bei Fehler:**
- Frühe Identifikation von Abhängigkeitsproblemen
- Anpassung Import-Paths
- Konfigurationsvalidierung

---

**Next Steps:**
1. Validate this specification with detailed requirements
2. Create new branch: `pipeline-v2`
3. Execute Phase 1 bootstrapping (copy minimal files)
4. Run initial v1 test
5. Iterative refinement based on test results
