# Multi-Agent Benchmark Results

Generated: 2025-07-23 13:46:56

## Summary

- **Total environments tested**: 5
- **Total agents tested**: 3
- **Total training time**: 13180.45 seconds
- **Total evaluation episodes**: 300

## Results by Environment

### CartPole-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | 57.35 | 25.79 | 0.00% | ❌ |
| lstm | 71.85 | 7.05 | 0.00% | ❌ |
| ld | 20.05 | 2.62 | 0.00% | ❌ |

![CartPole-v1 Learning Curves](plots/CartPole-v1_comparison.png)

**Winner**: lstm (69.75)

### LunarLander-v3

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | 101.62 | 110.28 | 5.00% | ❌ |
| lstm | 162.03 | 96.90 | 35.00% | ❌ |
| ld | 124.94 | 113.03 | 40.00% | ❌ |

![LunarLander-v3 Learning Curves](plots/LunarLander-v3_comparison.png)

**Winner**: lstm (117.21)

### FrozenLake-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | 0.75 | 0.43 | 75.00% | ✅ |
| lstm | 0.45 | 0.50 | 45.00% | ❌ |
| ld | 0.35 | 0.48 | 35.00% | ❌ |

![FrozenLake-v1 Learning Curves](plots/FrozenLake-v1_comparison.png)

**Winner**: ppo (0.70)

### Acrobot-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -81.85 | 15.48 | 90.00% | ✅ |
| lstm | -79.75 | 10.31 | 95.00% | ✅ |
| ld | -500.00 | 0.00 | 0.00% | ❌ |

![Acrobot-v1 Learning Curves](plots/Acrobot-v1_comparison.png)

**Winner**: ppo (-83.95)

### Acrobot-v1-Partial

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -456.80 | 54.50 | 0.00% | ❌ |
| lstm | -500.00 | 0.00 | 0.00% | ❌ |
| ld | -83.85 | 17.97 | 100.00% | ✅ |

![Acrobot-v1-Partial Learning Curves](plots/Acrobot-v1-Partial_comparison.png)

**Winner**: ld (-85.25)

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
