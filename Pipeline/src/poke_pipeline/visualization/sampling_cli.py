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
from typing import List, Tuple, Optional

from .data_sampling import load_variants_for_comparison
from .heatmap_plots import plot_all_maps
from .reward_plots import plot_rewards_over_time


def _is_log_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for pattern in ("*.csv", "*.json", "*.jsonl"):
        if any(path.glob(pattern)):
            return True
    return False


def _resolve_log_dir(raw_path: Path, variant: str) -> Tuple[Path, Optional[str]]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {raw_path}")

    if _is_log_dir(path):
        return path, None

    logs_dir = path / "logs"
    if logs_dir.is_dir() and _is_log_dir(logs_dir):
        return logs_dir, None

    candidates: List[Path] = []
    if path.is_dir():
        for child in path.iterdir():
            if not child.is_dir():
                continue
            child_logs = child / "logs"
            if child_logs.is_dir() and _is_log_dir(child_logs):
                candidates.append(child_logs)
            elif _is_log_dir(child):
                candidates.append(child)

    if candidates:
        selected = max(candidates, key=lambda p: p.parent.stat().st_mtime)
        message = f"[sampling_cli] Auto-selected latest run: {selected.parent}"
        return selected, message

    raise argparse.ArgumentTypeError(
        f"Could not resolve a logs directory under {raw_path}"
    )


def parse_run(value: str) -> Tuple[Path, str]:
    """Parse --runs entries allowing optional `<log_path>:<variant>`."""
    variant = None

    if ":" in value:
        path_part, variant_part = value.split(":", maxsplit=1)
        variant = variant_part.strip() or None
        raw_path = Path(path_part)
    else:
        raw_path = Path(value)

    log_dir, info = _resolve_log_dir(raw_path, variant or "auto")
    if info:
        print(info)
    if variant is None:
        try:
            variant = log_dir.parent.parent.name or log_dir.parent.name
        except Exception:
            variant = raw_path.name or "run"
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
