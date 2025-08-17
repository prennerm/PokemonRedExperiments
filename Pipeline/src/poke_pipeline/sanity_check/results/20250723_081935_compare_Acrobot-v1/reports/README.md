# Multi-Agent Benchmark Results

Generated: 2025-07-23 09:09:17

## Results by Environment

### Acrobot-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -82.60 | 13.50 | 90.00% | ✅ |
| lstm | -169.55 | 165.64 | 75.00% | ❌ |
| ld | -83.10 | 12.77 | 85.00% | ✅ |

![Acrobot-v1 Learning Curves](plots/Acrobot-v1_comparison.png)

**Winner**: ld (-83.35)

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
