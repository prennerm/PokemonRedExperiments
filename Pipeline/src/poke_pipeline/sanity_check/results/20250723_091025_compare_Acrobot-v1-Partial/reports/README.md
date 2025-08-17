# Multi-Agent Benchmark Results

Generated: 2025-07-23 09:58:39

## Results by Environment

### Acrobot-v1-Partial

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -332.15 | 83.57 | 0.00% | ❌ |
| lstm | -500.00 | 0.00 | 0.00% | ❌ |
| ld | -88.40 | 16.82 | 100.00% | ✅ |

![Acrobot-v1-Partial Learning Curves](plots/Acrobot-v1-Partial_comparison.png)

**Winner**: ld (-91.00)

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
