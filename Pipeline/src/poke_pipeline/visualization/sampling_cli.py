#!/usr/bin/env python3
"""CLI entrypoint to sample training logs and generate reward/heatmap plots.

Example usage:

    python -m poke_pipeline.visualization.sampling_cli \
        --runs experiments/v4/20251030_165359/logs:v4 \
        --plot-rewards --plot-heatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

from .data_sampling import load_variants_for_comparison
from .heatmap_plots import plot_all_maps
from .reward_plots import plot_rewards_over_time


def parse_run(value: str) -> Tuple[Path, str]:
    """Parse --runs entries of the form `<log_path>:<variant>`."""
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format '<log_path>:<variant>'")
    log_dir = Path(parts[0]).expanduser().resolve()
    variant = parts[1].strip()
    if not log_dir.exists():
        raise argparse.ArgumentTypeError(f"Log directory does not exist: {log_dir}")
    if not variant:
        raise argparse.ArgumentTypeError("Variant must not be empty")
    return log_dir, variant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sampling + visualization helper")
    parser.add_argument(
        "--runs",
        metavar="LOG:VARIANT",
        type=parse_run,
        nargs="+",
        required=True,
        help="Log directory paired with variant label, e.g. experiments/.../logs:v4",
    )
    parser.add_argument("--max-steps", type=int, default=100_000_000, help="Maximum steps to consider")
    parser.add_argument("--target-samples", type=int, default=5000, help="Reservoir size per variant")
    parser.add_argument("--plot-rewards", action="store_true", help="Generate reward plots")
    parser.add_argument("--plot-heatmaps", action="store_true", help="Generate heatmap plots")
    parser.add_argument("--normalize-rewards", action="store_true", help="Normalize reward x-axis to 0..1")
    parser.add_argument("--normalize-heatmaps", action="store_true", help="Normalize heatmap intensities")
    parser.add_argument("--heatmap-cmap", default="magma", help="Colormap for heatmaps")
    parser.add_argument("--max-files", type=int, default=None, help="Limit number of stats files per run")
    parser.add_argument(
        "--max-data-points",
        type=int,
        default=None,
        help="Limit total number of datapoints sampled (spread evenly across the run)",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    runs: List[Tuple[Path, str]] = args.runs
    if not args.plot_rewards and not args.plot_heatmaps:
        args.plot_rewards = args.plot_heatmaps = True

    if args.max_files is not None and args.max_data_points is not None:
        parser.error("--max-files and --max-data-points cannot be used together.")

    variant_configs = [(str(path), variant) for path, variant in runs]
    data_frames, sampler = load_variants_for_comparison(
        variant_configs,
        max_steps=args.max_steps,
        target_samples=args.target_samples,
        normalize_steps=args.normalize_rewards,
        max_files=args.max_files,
        max_data_points=args.max_data_points,
        verbose=not args.quiet,
    )

    sampling_meta = getattr(sampler, "variant_sampling_meta", {})

    for (log_dir, variant), df in zip(runs, data_frames):
        run_dir = log_dir.parent
        experiment_name = run_dir.name  # typically timestamp
        if args.plot_rewards:
            meta = sampling_meta.get(variant, {})
            plot_rewards_over_time(
                df,
                variant=variant,
                experiment_name=experiment_name,
                run_dir=run_dir,
                normalize_steps=args.normalize_rewards,
                samples=meta.get("samples"),
                step_range=meta.get("step_range"),
                sampling_ratio=meta.get("sampling_ratio"),
            )
        if args.plot_heatmaps:
            plot_all_maps(
                df,
                variant=variant,
                experiment_name=experiment_name,
                run_dir=run_dir,
                normalize=args.normalize_heatmaps,
                cmap=args.heatmap_cmap,
            )

    if not args.quiet:
        for log_dir, _ in runs:
            print(f"[sampling_cli] Finished. Plots stored in {(log_dir.parent / 'plots').resolve()}")


if __name__ == "__main__":
    main()
