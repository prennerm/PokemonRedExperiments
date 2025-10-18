#!/usr/bin/env python3
"""
Ad-hoc helper to generate position heatmaps for a single experiment run.

Usage:
    python scripts/debug_heatmaps.py --variant v4 --experiment experiments/v4/20251012_145635
"""

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis import TrainingSampler
from analysis.visualizers import plot_all_maps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps for Pokemon Red RL training logs."
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=["v1", "v2", "v3", "v4"],
        help="Variant identifier matching the experiment directory",
    )
    parser.add_argument(
        "--experiment",
        type=Path,
        required=True,
        help="Path to the experiment directory (e.g. experiments/v4/20251012_145635)",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=20000,
        help="Reservoir sample size when loading logs",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalise heatmap intensities to [0, 1]",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logs_dir = args.experiment / "logs"
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

    sampler = TrainingSampler(target_samples=args.target_samples, verbose=False)
    df = sampler.load_variant_data(str(logs_dir), variant_name=args.variant)

    outputs = plot_all_maps(
        df,
        variant=args.variant,
        experiment_name=args.experiment.name,
        output_dir=args.experiment / "plots",
        normalize=args.normalize,
    )

    if outputs:
        print("Generated heatmaps:")
        for map_id, path in outputs.items():
            print(f"  map {map_id:>3}: {path}")
    else:
        print("No heatmaps produced (no matching samples).")


if __name__ == "__main__":
    main()
