# Multi-Agent Benchmark Results

Generated: 2025-07-18 17:19:12

## Results by Environment

### CartPole-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| standard_ppo | 115.05 | 10.89 | 0.00% | ❌ |

![CartPole-v1 Learning Curves](plots/CartPole-v1_comparison.png)

**Winner**: standard_ppo (115.05)

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
