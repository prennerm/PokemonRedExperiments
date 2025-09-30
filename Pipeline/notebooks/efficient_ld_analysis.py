#!/usr/bin/env python3
"""
Lambda Discrepancy Effect Analysis
Focused comparison of v3 vs v4 variants using reusable sampling strategy
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from data_sampling import load_variants_for_comparison

def create_lambda_discrepancy_comparison():
    """Create comparison visualization using reusable sampling strategy"""

    # Define variants to compare
    variant_configs = [
        ("../experiments/v3/20250620_070120/json_logs", "v3_lstm_only"),
        ("../experiments/v4_production/20250813_125101/json_logs", "v4_production"),
        ("../experiments/v4_production_ld_02/20250924_182227/json_logs", "v4_production_ld_02")
    ]

    # Load data with consistent sampling (max 100M steps)
    print("=== Lambda Discrepancy Effect Analysis ===")
    dataframes, sampler = load_variants_for_comparison(
        variant_configs,
        max_steps=100_000_000,  # 100M step limit
        target_samples=4000,    # 4k samples per variant
        max_files=3,            # Max 3 files per variant
        normalize_steps=True    # Ensure same step ranges
    )

    v3_data, v4_prod_data, v4_ld_data = dataframes

    # Create focused visualizations - only Total Reward and Exploration Progress
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plot 1: Step vs Total Reward (Learning Progress)
    ax1 = axes[0]
    for data, label, color in [(v3_data, 'v3 (LSTM only)', 'blue'),
                               (v4_prod_data, 'v4_production', 'orange'),
                               (v4_ld_data, 'v4_production_ld_02', 'red')]:
        if len(data) > 0:
            # Sort by step for better visualization
            data_sorted = data.sort_values('step')
            print(f"\n{label} step range: {data_sorted['step'].min()} - {data_sorted['step'].max()}")
            print(f"{label} total points: {len(data_sorted)}")

            # Use rolling average for smoother curves
            window_size = max(10, len(data_sorted) // 50)  # More granular smoothing
            if len(data_sorted) > window_size:
                smooth_reward = data_sorted['total_reward'].rolling(window=window_size, center=True).mean()
                ax1.plot(data_sorted['step'], smooth_reward, label=label, color=color, linewidth=2.5, alpha=0.9)
            else:
                ax1.plot(data_sorted['step'], data_sorted['total_reward'], label=label, color=color, linewidth=2.5, alpha=0.9)

    ax1.set_title('Total Reward Over Training Steps', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Training Step', fontsize=12)
    ax1.set_ylabel('Total Reward (Rolling Average)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Step vs Exploration Progress
    ax2 = axes[1]
    for data, label, color in [(v3_data, 'v3 (LSTM only)', 'blue'),
                               (v4_prod_data, 'v4_production', 'orange'),
                               (v4_ld_data, 'v4_production_ld_02', 'red')]:
        if len(data) > 0:
            # Sort by step for better visualization
            data_sorted = data.sort_values('step')

            # Use rolling average for smoother curves
            window_size = max(10, len(data_sorted) // 50)
            if len(data_sorted) > window_size:
                smooth_exploration = data_sorted['exploration'].rolling(window=window_size, center=True).mean()
                ax2.plot(data_sorted['step'], smooth_exploration, label=label, color=color, linewidth=2.5, alpha=0.9)
            else:
                ax2.plot(data_sorted['step'], data_sorted['exploration'], label=label, color=color, linewidth=2.5, alpha=0.9)

    ax2.set_title('Exploration Progress Over Training Steps', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Coordinates Explored (Rolling Average)', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'lambda_discrepancy_effect_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Print detailed statistics
    print("\n=== Detailed Analysis Results ===")
    for name, data in [('v3 (LSTM only)', v3_data),
                       ('v4_production', v4_prod_data),
                       ('v4_production_ld_02', v4_ld_data)]:
        if len(data) > 0:
            print(f"\n{name}:")
            print(f"  Samples analyzed: {len(data)}")
            print(f"  Total reward: {data['total_reward'].mean():.3f} ± {data['total_reward'].std():.3f}")
            print(f"  Badge progress: {data['badges'].mean():.3f} (max: {data['badges'].max()})")
            print(f"  Exploration: {data['exploration'].mean():.1f} (max: {data['exploration'].max()})")
            print(f"  Health: {data['health'].mean():.3f}")
            print(f"  Deaths: {data['deaths'].mean():.3f}")
            print(f"  Step range: {data['step'].min()} - {data['step'].max()}")

    # Save summary to file
    with open(output_dir / 'lambda_discrepancy_summary.txt', 'w') as f:
        f.write("Lambda Discrepancy Effect Analysis - Summary\n")
        f.write("="*50 + "\n\n")
        f.write("Research Question: Does LSTM + Lambda Discrepancy lead to more stable and effective policies?\n\n")

        for name, data in [('v3 (LSTM only)', v3_data),
                           ('v4_production', v4_prod_data),
                           ('v4_production_ld_02', v4_ld_data)]:
            if len(data) > 0:
                f.write(f"{name}:\n")
                f.write(f"  Samples: {len(data)}\n")
                f.write(f"  Mean total reward: {data['total_reward'].mean():.3f} ± {data['total_reward'].std():.3f}\n")
                f.write(f"  Max badges: {data['badges'].max()}\n")
                f.write(f"  Max exploration: {data['exploration'].max()}\n")
                f.write(f"  Step range: {data['step'].min()} - {data['step'].max()}\n\n")

    print(f"\nAnalysis complete! Results saved to: {output_dir}")

if __name__ == "__main__":
    create_lambda_discrepancy_comparison()