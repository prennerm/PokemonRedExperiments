# Multi-Agent Benchmark Results

Generated: 2025-07-19 14:35:23

## Summary

- **Total environments tested**: 4
- **Total agents tested**: 9
- **Total training time**: 17749.75 seconds
- **Total evaluation episodes**: 540

## Results by Environment

### CartPole-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| standard_ppo | 210.70 | 26.98 | 0.00% | ❌ |
| ppo | 63.55 | 18.21 | 0.00% | ❌ |
| baseline | 232.65 | 52.24 | 0.00% | ❌ |
| lstm | 177.50 | 38.75 | 0.00% | ❌ |
| recurrent | 179.65 | 85.58 | 0.00% | ❌ |
| memory | 12.00 | 1.55 | 0.00% | ❌ |
| ld | 29.35 | 4.32 | 0.00% | ❌ |
| lambda_discrepancy | 44.60 | 6.67 | 0.00% | ❌ |
| partial_obs | 31.85 | 3.32 | 0.00% | ❌ |

![CartPole-v1 Learning Curves](plots/CartPole-v1_comparison.png)

**Winner**: baseline (262.15)

### MountainCar-v0

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| standard_ppo | -200.00 | 0.00 | 0.00% | ❌ |
| ppo | -200.00 | 0.00 | 0.00% | ❌ |
| baseline | -200.00 | 0.00 | 0.00% | ❌ |
| lstm | -200.00 | 0.00 | 0.00% | ❌ |
| recurrent | -200.00 | 0.00 | 0.00% | ❌ |
| memory | -200.00 | 0.00 | 0.00% | ❌ |
| ld | -200.00 | 0.00 | 0.00% | ❌ |
| lambda_discrepancy | -200.00 | 0.00 | 0.00% | ❌ |
| partial_obs | -200.00 | 0.00 | 0.00% | ❌ |

![MountainCar-v0 Learning Curves](plots/MountainCar-v0_comparison.png)

**Winner**: standard_ppo (-200.00)

### FrozenLake-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| standard_ppo | 0.35 | 0.48 | 35.00% | ❌ |
| ppo | 0.35 | 0.48 | 35.00% | ❌ |
| baseline | 0.80 | 0.40 | 80.00% | ✅ |
| lstm | 0.05 | 0.22 | 5.00% | ❌ |
| recurrent | 0.15 | 0.36 | 15.00% | ❌ |
| memory | 0.10 | 0.30 | 10.00% | ❌ |
| ld | 0.60 | 0.49 | 60.00% | ❌ |
| lambda_discrepancy | 0.35 | 0.48 | 35.00% | ❌ |
| partial_obs | 0.15 | 0.36 | 15.00% | ❌ |

![FrozenLake-v1 Learning Curves](plots/FrozenLake-v1_comparison.png)

**Winner**: baseline (0.70)

## Methodology

- **Evaluation**: 20 episodes per agent per environment
- **Reproducibility**: Fixed seeds for deterministic evaluation
- **Success Criteria**: Environment-specific thresholds
- **No Reward Shaping**: Environments used as-is for scientific validity

## Generated Files

- `comparison_report.txt`: Detailed text report
- `results.json`: Machine-readable results
- `summary_table.csv`: CSV summary for analysis
- `plots/`: Learning curve visualizations
- `models/`: Trained model files
- `tensorboard/`: TensorBoard logs
