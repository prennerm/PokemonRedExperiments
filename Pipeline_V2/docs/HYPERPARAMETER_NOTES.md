# Hyperparameter Configuration Notes

## Overview

This document explains hyperparameter choices for the Lambda Discrepancy research project, including deviations from Peter Whidden's baseline implementation.

## Reference Implementation

**Source:** `PokemonRedExperiments_pre_fork/v2/baseline_fast_v2.py`

Peter Whidden's original configuration:
```python
ep_length = 2048 * 80 = 163840
num_cpu = 64
n_steps = ep_length // num_cpu = 2560
batch_size = 512
n_epochs = 1
gamma = 0.997
ent_coef = 0.01
save_freq = ep_length // 2 = 81920
total_timesteps = ep_length * num_cpu * 10000 ≈ 105 billion
```

## Pipeline V2 Configuration

### Dynamic n_steps Calculation

**Implementation:** `src/trainers/base_trainer.py::_apply_n_steps_formula()`

**Formula:** `n_steps = max_steps // num_cpu`

**Rationale:**
- Ensures exactly 1 environment reset per rollout buffer
- Maintains training rhythm independent of worker count
- Preserves original relationship from Whidden's implementation

**Example:**
```
max_steps = 163840 (episode length)
num_cpu = 16 (hardware constraint)
→ n_steps = 10240 (calculated at runtime)
```

### Hardware-Adapted Parameters

**num_cpu: 16** (vs original 64)
- Constraint: Limited compute resources
- Impact: Training slower but results unchanged
- Note: Worker count affects speed only, not learning dynamics

**total_timesteps: 1e8** (vs original ~105 billion)
- Constraint: Time/budget limitations
- Scaling: `163840 × 16 × ~38000 ≈ 100 million`
- Tradeoff: Shorter training for feasibility

### Preserved Parameters

**From Whidden's baseline (unchanged):**
- `gamma: 0.997` - Discount factor
- `ent_coef: 0.01` - Entropy coefficient
- `batch_size: 512` - Minibatch size
- `save_freq: 81920` - Checkpoint frequency (half-episode)

## Scientific Transparency: n_epochs

### Baseline Value
Peter Whidden used `n_epochs: 1` (1 gradient update pass per rollout buffer).

### Empirical Observation
During preliminary experiments, v4 (Lambda Discrepancy variant) showed improved learning with `n_epochs: 3`.

### Scientific Considerations

**Arguments for n_epochs = 3:**
- Better sample efficiency with limited compute (fewer environment steps)
- More gradient updates per collected data
- Empirically effective for LSTM + auxiliary loss (Lambda Discrepancy)

**Arguments for n_epochs = 1:**
- Exact replication of Whidden's baseline
- Minimal on-policy data reuse (PPO best practice)
- Fair comparison requires identical hyperparameters

### Current Decision
**Using n_epochs: 1** to preserve baseline alignment.

**For paper:** If changing to n_epochs = 3:
1. Document deviation transparently
2. Justify with empirical evidence (preliminary runs)
3. Consider separate ablation study (n_epochs: 1 vs 3)
4. Acknowledge tradeoff: sample efficiency vs strict replication

## Summary Table

| Parameter | Whidden | Pipeline V2 | Reason |
|-----------|---------|-------------|--------|
| n_steps | 2560 | 10240 (dynamic) | Formula: max_steps / num_cpu |
| num_cpu | 64 | 16 | Hardware constraint |
| total_timesteps | ~105B | 100M | Time/budget constraint |
| save_freq | 81920 | 81920 | ✅ Preserved |
| n_epochs | 1 | 1 | ✅ Preserved (3 optional) |
| gamma | 0.997 | 0.997 | ✅ Preserved |
| ent_coef | 0.01 | 0.01 | ✅ Preserved |
| batch_size | 512 | 512 | ✅ Preserved |

**Legend:**
- ✅ Preserved - Exact match to baseline
- Dynamic - Calculated at runtime
- Hardware constraint - Adapted to available resources
- Time/budget constraint - Practical limitation
