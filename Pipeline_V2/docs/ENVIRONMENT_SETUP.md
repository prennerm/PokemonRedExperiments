# Environment Setup Guide

## Overview: Two-Environment System

Pipeline_V2 uses **two separate conda environments** with strict separation of concerns:

| Environment | Purpose | Key Dependencies | Use Case |
|------------|---------|------------------|----------|
| **poke_env** | Training | PyTorch, CUDA, Stable-Baselines3, PyBoy | Train RL agents on GPU |
| **poke_viz_v2** | Visualization | Matplotlib, Seaborn, Pandas, mediapy | Analyze training data, create plots |

### Why Two Environments?

1. **Dependency Isolation:** PyTorch + CUDA (3GB+) vs lightweight analysis tools
2. **Conflict Prevention:** Training libraries can conflict with visualization packages
3. **Reproducibility:** Each `environment.yml` is self-contained and version-controlled
4. **Server Deployment:** Train on GPU cluster, visualize locally

---

## Quick Start

### Training Environment (poke_env)

```bash
# Create environment
conda env create -f environment.yml

# Activate
conda activate poke_env

# Verify
python -c "import torch, stable_baselines3; print('Training env OK')"

# Run training
python train.py --variant v4 --config configs/v4.yaml
```

### Visualization Environment (poke_viz_v2)

```bash
# Create environment
conda env create -f environment_viz.yml

# Activate
conda activate poke_viz_v2

# Verify
python -c "import matplotlib, pandas, mediapy; print('Viz env OK')"

# Run visualization
python scripts/visualize_training.py --variant v4
```

---

## Detailed Setup

### 1. Training Environment (`poke_env`)

**Location:** `environment.yml` (root directory)

**Key Packages:**
- Python 3.10
- PyTorch 2.8+ with CUDA 12.4
- Stable-Baselines3 2.7.0
- PyBoy (Game Boy emulator)
- Gymnasium (RL environments)
- NumPy, Pandas, Matplotlib (basic analysis)

**Setup:**
```bash
# Create
conda env create -f environment.yml

# Activate
conda activate poke_env

# Test
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python train.py --help
```

**Important:** This environment should ONLY be used for training. Do NOT manually install additional packages!

---

### 2. Visualization Environment (`poke_viz_v2`)

**Location:** `environment_viz.yml` (root directory)

**Key Packages:**
- Python 3.10
- NumPy 2.0+
- Pandas 2.2+
- Matplotlib 3.9+
- Seaborn 0.13+
- SciPy, Scikit-Image
- Pillow (image processing)
- tqdm (progress bars)
- mediapy (map overlays)
- einops (tensor operations for heatmaps)

**Setup:**
```bash
# Create
conda env create -f environment_viz.yml

# Activate
conda activate poke_viz_v2

# Test
python -c "import matplotlib, seaborn, mediapy, einops; print('All viz packages OK')"
python -c "from analysis import TrainingSampler; print('Analysis module OK')"
```

**Important:** This environment should ONLY be used for visualization/analysis. No training code!

---

## Environment Testing

We provide a comprehensive smoke test script:

```bash
python test_environments.py
```

This will:
1. Verify `poke_env` has training packages but NOT viz-specific ones
2. Verify `poke_viz_v2` has viz packages but NOT training ones
3. Test that Phase 1 analysis code works in `poke_viz_v2`

---

## Workflow Examples

### Typical Development Workflow

```bash
# Terminal 1: Training (long-running)
conda activate poke_env
python train.py --variant v4 --config configs/v4.yaml

# Terminal 2: Visualization (while training runs)
conda activate poke_viz_v2
python scripts/visualize_training.py --variant v4 --live
```

### Server Training + Local Visualization

```bash
# On GPU server
conda activate poke_env
python train.py --variant v4 --config configs/v4.yaml

# Sync data to local machine
rsync -avz server:/path/to/experiments/ experiments/

# On local machine
conda activate poke_viz_v2
python scripts/compare_variants.py --v3 experiments/v3 --v4 experiments/v4
```

---

## Environment Maintenance

### Updating Dependencies

**Training Environment:**
```bash
# Edit environment.yml
# Then recreate:
conda env remove -n poke_env
conda env create -f environment.yml
```

**Visualization Environment:**
```bash
# Edit environment_viz.yml
# Then recreate:
conda env remove -n poke_viz_v2
conda env create -f environment_viz.yml
```

### Exporting Current State

Useful for debugging or sharing exact versions:

```bash
# Export poke_env
conda env export -n poke_env > environment_export.yml

# Export poke_viz_v2
conda env export -n poke_viz_v2 > environment_viz_export.yml
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'tqdm'" in poke_env

**Problem:** `tqdm` was manually installed in `poke_env` but is not in `environment.yml`.

**Solution:**
- Use `poke_viz_v2` for analysis code (which has `tqdm`)
- OR add `tqdm` to `environment.yml` if you absolutely need it for training

### "ModuleNotFoundError: No module named 'torch'" in poke_viz_v2

**Problem:** Trying to run training code in visualization environment.

**Solution:**
- Training code MUST run in `poke_env`
- Visualization code MUST run in `poke_viz_v2`
- Check which environment is active: `conda env list`

### CUDA Not Available in poke_env

**Problem:** PyTorch doesn't detect GPU.

**Check:**
```bash
conda activate poke_env
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

**Solution:** Ensure CUDA drivers are installed and match PyTorch CUDA version (12.4).

### Import Errors in Analysis Code

**Problem:** `from analysis import TrainingSampler` fails.

**Solution:**
```bash
# Make sure you're in Pipeline_V2 root directory
cd /path/to/Pipeline_V2

# Activate correct environment
conda activate poke_viz_v2

# Test
python -c "import sys; sys.path.insert(0, 'src'); from analysis import TrainingSampler; print('OK')"
```

---

## Best Practices

### DO:
- ✅ Always activate the correct environment before running code
- ✅ Keep `environment.yml` and `environment_viz.yml` in version control
- ✅ Test both environments after making changes
- ✅ Use `conda env create -f` to recreate environments (not manual installs)

### DON'T:
- ❌ Mix training and visualization code in one environment
- ❌ Manually `pip install` or `conda install` packages without updating YAML files
- ❌ Run training code in `poke_viz_v2` or vice versa
- ❌ Commit `environment_export.yml` files (too verbose, use base YAML files)

---

## Reference

### Environment File Locations
```
Pipeline_V2/
├── environment.yml           # poke_env (Training)
├── environment_viz.yml       # poke_viz_v2 (Visualization)
└── tmp/
    ├── poke_viz_legacy.yml          # Backup of old poke_viz
    └── poke_viz_extended_legacy.yml # Backup of old poke_viz_extended
```

### Related Documentation
- [VISUALIZATION_SUGGESTIONS.md](VISUALIZATION_SUGGESTIONS.md) - Visualization best practices
- [PIPELINE_V2_SPECIFICATION.md](PIPELINE_V2_SPECIFICATION.md) - Overall pipeline architecture
- [README.md](../README.md) - Project overview and quickstart

---

## Changelog

**2025-01-15:**
- Created two-environment system (poke_env + poke_viz_v2)
- Removed old `poke_viz` and `poke_viz_extended` environments
- Cleaned `tqdm` from `poke_env` (moved to `poke_viz_v2`)
- Added comprehensive smoke tests (`test_environments.py`)
