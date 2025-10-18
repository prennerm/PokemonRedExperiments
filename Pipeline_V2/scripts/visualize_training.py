#!/usr/bin/env python3
"""
Visualization entry point for Pokemon Red RL training analysis

Creates reward learning curves for a single training run.
For multi-variant comparisons, use compare_variants.py instead.

Usage:
    python scripts/visualize_training.py --variant v4 --experiment experiments/v4/20251012_145635

Requirements:
    - Run in poke_viz_v2 environment: conda activate poke_viz_v2
    - Experiment directory must contain logs/ subdirectory with stats.csv or *.json files
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis import TrainingSampler
from analysis.visualizers import plot_reward_curves, plot_all_maps


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Visualize Pokemon Red RL training data (single run)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize latest v4 training run
  python scripts/visualize_training.py --variant v4 --experiment experiments/v4/20251012_145635

  # Custom output directory and format
  python scripts/visualize_training.py --variant v3 --experiment experiments/v3/20250620_070120 --output-dir plots --format pdf

  # Limit processing for testing
  python scripts/visualize_training.py --variant v4 --experiment experiments/v4/20251012_145635 --max-files 10

Environment:
  This script must run in the poke_viz_v2 conda environment.
  Activate with: conda activate poke_viz_v2
        """
    )

    parser.add_argument(
        '--variant',
        type=str,
        required=True,
        choices=['v1', 'v2', 'v3', 'v4'],
        help='Training variant to visualize'
    )

    parser.add_argument(
        '--experiment',
        type=Path,
        required=True,
        help='Path to experiment directory (e.g., experiments/v4/20251012_145635)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for plots (default: {experiment}/plots/)'
    )

    parser.add_argument(
        '--format',
        choices=['png', 'pdf', 'svg'],
        default='png',
        help='Output format for plots (default: png)'
    )

    parser.add_argument(
        '--smooth-window',
        type=int,
        default=100,
        help='Smoothing window size for reward curves (default: 100)'
    )

    parser.add_argument(
        '--max-files',
        type=int,
        default=None,
        help='Maximum number of JSON files to process (for testing, CSV not affected)'
    )

    parser.add_argument(
        '--target-samples',
        type=int,
        default=10000,
        help='Target number of samples to load (default: 10000)'
    )

    parser.add_argument(
        '--heatmaps',
        action='store_true',
        help='Additionally generate position heatmaps for each visited map'
    )

    parser.add_argument(
        '--heatmap-normalize',
        action='store_true',
        help='Normalise heatmap intensities to [0, 1]'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    return parser.parse_args()


def main():
    """Main entry point for single-run visualization"""
    args = parse_args()

    print(f"=== Pokemon Red RL Training Visualization ===")
    print(f"Variant: {args.variant}")
    print(f"Experiment: {args.experiment}")
    print()

    # Validate experiment directory
    if not args.experiment.exists():
        print(f"[ERROR] Experiment directory not found: {args.experiment}")
        sys.exit(1)

    # Find logs directory
    logs_dir = args.experiment / "logs"
    if not logs_dir.exists():
        print(f"[ERROR] Logs directory not found: {logs_dir}")
        print(f"Expected structure: {args.experiment}/logs/stats.csv or {args.experiment}/logs/*.json")
        sys.exit(1)

    # Set output directory
    if args.output_dir is None:
        output_dir = args.experiment / "plots"
    else:
        output_dir = args.output_dir

    print(f"Output directory: {output_dir}")
    print()

    # Initialize sampler
    print(f"[1/3] Loading training data...")
    sampler = TrainingSampler(
        max_steps=100_000_000,
        target_samples=args.target_samples,
        verbose=args.verbose
    )

    # Load data
    df = sampler.load_variant_data(
        experiment_path=str(logs_dir),
        variant_name=args.variant,
        max_files=args.max_files
    )

    if df.empty:
        print(f"[ERROR] No data loaded from {logs_dir}")
        sys.exit(1)

    # Get statistics
    stats = sampler.get_statistics()

    print()
    print(f"[2/3] Creating reward plot...")

    # Generate reward plot
    experiment_name = args.experiment.name
    output_file = plot_reward_curves(
        df=df,
        stats=stats,
        variant=args.variant,
        experiment_name=experiment_name,
        output_dir=output_dir,
        smooth_window=args.smooth_window,
        output_format=args.format,
        verbose=args.verbose
    )

    if output_file is None:
        print(f"[ERROR] Failed to generate reward plot")
        sys.exit(1)

    heatmap_outputs = {}
    if args.heatmaps:
        print()
        print(f"[3/4] Generating heatmaps...")
        heatmap_outputs = plot_all_maps(
            df,
            variant=args.variant,
            experiment_name=experiment_name,
            output_dir=output_dir,
            normalize=args.heatmap_normalize,
        )
        if heatmap_outputs:
            print(f"    Heatmaps generated for {len(heatmap_outputs)} map(s).")
        else:
            print("    No heatmaps generated (missing position data).")

    print()
    print(f"[{4 if args.heatmaps else 3}/3] Complete!")
    print()
    print(f"=== Summary ===")
    print(f"Variant: {args.variant}")
    print(f"Experiment: {experiment_name}")
    print(f"Data processed: {stats['total_rows']:,} rows")
    print(f"Samples plotted: {stats['final_samples']:,}")
    print(f"Step range: {stats['step_range'][0]:,} - {stats['step_range'][1]:,}")
    print(f"Reward plot: {output_file}")
    if heatmap_outputs:
        print("Heatmaps:")
        for map_id, path in heatmap_outputs.items():
            print(f"  map {map_id:>3}: {path}")
    print()
    print(f"[OK] Visualization complete!")


if __name__ == "__main__":
    main()
