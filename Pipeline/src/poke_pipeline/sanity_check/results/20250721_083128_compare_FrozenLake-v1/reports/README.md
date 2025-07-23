# Multi-Agent Benchmark Results

Generated: 2025-07-21 08:58:33

## Results by Environment

### FrozenLake-v1

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | 0.70 | 0.46 | 70.00% | ✅ |
| lstm | 0.15 | 0.36 | 15.00% | ❌ |
| ld | 0.25 | 0.43 | 25.00% | ❌ |

![FrozenLake-v1 Learning Curves](plots/FrozenLake-v1_comparison.png)

**Winner**: ppo (0.90)

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
