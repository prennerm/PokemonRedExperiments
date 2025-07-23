# Multi-Agent Benchmark Results

Generated: 2025-07-21 12:12:22

## Results by Environment

### LunarLander-v3

| Agent | Mean Reward | Std Reward | Success Rate | Baseline Passed |
|-------|-------------|------------|--------------|----------------|
| ppo | -7.64 | 149.26 | 40.00% | ❌ |
| lstm | -38.28 | 27.76 | 5.00% | ❌ |
| ld | 208.72 | 99.58 | 95.00% | ✅ |

![LunarLander-v3 Learning Curves](plots/LunarLander-v3_comparison.png)

**Winner**: ld (220.16)

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
