# Multi-Agent Benchmark Results

Generated: 2025-07-21 10:52:20

## Results by Environment

### MountainCar-v0

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -200.00 | 0.00 | 0.00% | ❌ |
| lstm | -200.00 | 0.00 | 0.00% | ❌ |
| ld | -200.00 | 0.00 | 0.00% | ❌ |

![MountainCar-v0 Learning Curves](plots/MountainCar-v0_comparison.png)

**Winner**: ppo (-200.00)

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
