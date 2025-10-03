#!/usr/bin/env python3
"""
Simple test script to verify data_sampling.py can handle 100M timesteps

Creates one basic plot to verify the sampling works correctly.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from data_sampling import TrainingSampler

def test_100m_timestep_sampling():
    """Test if data_sampling.py can handle 100M timesteps from your three training runs"""

    # Test paths you specified - relative to notebooks directory
    variant_configs = [
        ('../experiments/v3/20250903_100620/json_logs', 'v3'),
        ('../experiments/v4_production/20250813_125101/json_logs', 'v4_production'),
        ('../experiments/v4_production_ld_02/20250924_182227/json_logs', 'v4_ld_02')
    ]

    sampler = TrainingSampler(max_steps=100_000_000, target_samples=2000, verbose=True)

    all_data = []
    variant_names = []

    for experiment_path, variant_name in variant_configs:
        print(f"Testing {variant_name}...")
        df = sampler.load_variant_data(experiment_path, variant_name, max_files=10)

        if len(df) > 0:
            all_data.append(df)
            variant_names.append(variant_name)
            print(f"{variant_name}: {len(df)} samples, step range: {df['step'].min()} - {df['step'].max()}")
        else:
            print(f"{variant_name}: NO DATA")

    if not all_data:
        print("ERROR: No data loaded from any variant")
        return

    # Create simple plot
    plt.figure(figsize=(10, 6))

    colors = ['blue', 'orange', 'red']
    for i, (df, name) in enumerate(zip(all_data, variant_names)):
        plt.scatter(df['step'], df['total_reward'], alpha=0.6, s=10,
                   color=colors[i % len(colors)], label=name)

    plt.xlabel('Training Step')
    plt.ylabel('Total Reward')
    plt.title('100M Timestep Sampling Test')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save to notebooks/plots
    output_dir = Path('plots')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'data_sampling_test.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Test plot saved to notebooks/plots/data_sampling_test.png")
    print(f"Total variants loaded: {len(all_data)}")

if __name__ == "__main__":
    test_100m_timestep_sampling()