"""
CLI to compare total reward trajectories across multiple training runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
import shutil

from .data_sampling import load_variants_for_comparison
from .plot_helpers import setup_plot_style


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare total reward curves across runs.")
    parser.add_argument(
        "--runs",
        metavar="LOG:VARIANT",
        type=str,
        nargs="+",
        required=True,
        help="Log directory paired with variant label, e.g. experiments/.../logs:v4",
    )
    parser.add_argument("--max-data-points", type=int, default=None, help="Evenly sampled datapoints per run")
    parser.add_argument("--max-files", type=int, default=None, help="Alternative: limit number of files")
    parser.add_argument("--target-samples", type=int, default=5000, help="Reservoir size per variant")
    parser.add_argument("--max-steps", type=int, default=100_000_000, help="Maximum step value to consider")
    parser.add_argument("--smooth-window", type=int, default=0, help="Rolling mean window (0 disables smoothing)")
    parser.add_argument(
        "--normalize-steps",
        action="store_true",
        help="Trim all runs to the minimum max step for fair comparison.",
    )
    parser.add_argument(
        "--step-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="Optional inclusive range of steps to plot (after normalization).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_plots",
        help="Subdirectory under common experiment root for comparison outputs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def _format_steps(value: float, _: int) -> str:
    abs_val = abs(value)
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs_val >= 1_000:
        return f"{value/1_000:.1f}k"
    return f"{int(value)}"


_STEP_FORMATTER = FuncFormatter(_format_steps)


def _rolling_mean(series: pd.Series, window: int) -> Optional[pd.Series]:
    if window and window > 1:
        return series.rolling(window=window, min_periods=1).mean()
    return None


def _resolve_run_directory(log_path: Path) -> Path:
    if log_path.name == "logs":
        return log_path.parent
    return log_path


def _extract_timestamp(run_dir: Path) -> str:
    return run_dir.name


def _load_reward_series(
    variant_configs: List[Tuple[str, str]],
    *,
    target_samples: int,
    max_files: Optional[int],
    max_data_points: Optional[int],
    normalize_steps: bool,
    step_range: Optional[Tuple[int, int]],
    verbose: bool,
    max_steps: int,
) -> List[Tuple[Path, str, str, pd.Series, pd.Series]]:
    data_frames, _sampler = load_variants_for_comparison(
        variant_configs,
        target_samples=target_samples,
        max_files=max_files,
        max_data_points=max_data_points,
        normalize_steps=normalize_steps,
        verbose=verbose,
        max_steps=max_steps,
    )

    results: List[Tuple[Path, str, str, pd.Series, pd.Series]] = []
    for (log_dir_str, variant), df in zip(variant_configs, data_frames):
        log_dir = Path(log_dir_str).resolve()
        if df.empty:
            print(f"[compare_rewards] No data for {variant} ({log_dir}); skipping.")
            continue
        if "total_reward" not in df.columns:
            print(f"[compare_rewards] total_reward missing for {variant} ({log_dir}); skipping.")
            continue
        step_col = "total_steps" if "total_steps" in df.columns else "step"
        steps = pd.to_numeric(df[step_col], errors="coerce")

        rewards = pd.to_numeric(df["total_reward"], errors="coerce")
        valid = steps.notna() & rewards.notna()
        steps = steps[valid]
        rewards = rewards[valid]

        if step_range:
            start, end = step_range
            mask = (steps >= start) & (steps <= end)
            steps = steps[mask]
            rewards = rewards[mask]
            if steps.empty:
                print(f"[compare_rewards] Step range {step_range} empty for {variant}; skipping.")
                continue

        run_dir = _resolve_run_directory(log_dir)
        timestamp = _extract_timestamp(run_dir)
        variant_label = variant
        results.append(
            (
                run_dir,
                variant_label,
                timestamp,
                steps.reset_index(drop=True),
                rewards.reset_index(drop=True),
            )
        )
    return results


def _build_output_path(
    output_root: Path,
    variants: Iterable[str],
) -> Path:
    safe_variants = ["_".join(filter(None, v.split())) for v in variants]
    safe_variants = [v.replace("/", "-") for v in safe_variants]
    label = "_".join(safe_variants)
    label = label if label else "comparison"
    filename = f"total_reward_{label}.png"
    return output_root / filename


def _plot_total_reward(
    series: List[Tuple[Path, str, str, pd.Series, pd.Series]],
    *,
    smooth_window: int,
    output_dir: Path,
) -> Tuple[Path, List[Path]]:
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(series), 1)))
    variants: List[str] = []
    legend_labels: List[str] = []

    for idx, (_, variant, timestamp, steps, rewards) in enumerate(series):
        color = colors[idx % len(colors)]
        variants.append(variant)
        legend_labels.append(f"{variant}/{timestamp}")
        steps_arr = steps.to_numpy()
        rewards_arr = rewards.to_numpy()
        smoothed = _rolling_mean(rewards, smooth_window)

        label = f"{variant}/{timestamp}"
        ax.plot(
            steps_arr,
            rewards_arr,
            label=label if smoothed is None else None,
            color=color,
            alpha=0.4 if smoothed is not None else 0.9,
            linewidth=1.2,
        )
        if smoothed is not None:
            ax.plot(
                steps_arr,
                smoothed.to_numpy(),
                label=label,
                color=color,
                linewidth=2,
            )

    ax.set_xlabel("total_steps")
    ax.set_ylabel("Total Reward")
    title_variants = ", ".join(legend_labels)
    ax.set_title(f"Total Reward Comparison ({title_variants})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(_STEP_FORMATTER)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = _build_output_path(output_dir, variants)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[compare_rewards] Saved comparison plot: {output_path}")
    run_dirs = [entry[0] for entry in series]
    return output_path, run_dirs


def run_cli(args: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    if parsed.max_files is not None and parsed.max_data_points is not None:
        raise ValueError("Use either --max-files or --max-data-points, not both.")

    run_specs: List[Tuple[str, str]] = []
    run_dirs: List[Path] = []
    for item in parsed.runs:
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"Run must be in LOG:VARIANT format (got {item})")
        path_str, variant = item.split(":", maxsplit=1)
        log_path = Path(path_str).resolve()
        run_specs.append((str(log_path), variant))
        run_dirs.append(_resolve_run_directory(log_path))

    runs_data = _load_reward_series(
        run_specs,
        target_samples=parsed.target_samples,
        max_files=parsed.max_files,
        max_data_points=parsed.max_data_points,
        normalize_steps=parsed.normalize_steps,
        step_range=tuple(parsed.step_range) if parsed.step_range else None,
        verbose=not parsed.quiet,
        max_steps=parsed.max_steps,
    )

    if len(runs_data) < 2:
        print("[compare_rewards] Need at least two runs with reward data to compare.")
        return

    reference_dir = run_dirs[0]
    output_dir = reference_dir / "plots" / parsed.output_dir

    output_path, run_dirs = _plot_total_reward(
        runs_data,
        smooth_window=parsed.smooth_window,
        output_dir=output_dir,
    )

    for run_dir in run_dirs:
        target_dir = run_dir / "plots" / parsed.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / output_path.name
        if target_path != output_path:
            shutil.copy2(output_path, target_path)
            print(f"[compare_rewards] Copied comparison plot to: {target_path}")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
