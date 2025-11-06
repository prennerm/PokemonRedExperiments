"""
Visualization CLI for plotting lambda-discrepancy metrics from training logs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .data_sampling import load_variants_for_comparison
from .plot_helpers import setup_plot_style


def _is_log_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    for pattern in ("*.csv", "*.json", "*.jsonl"):
        if any(path.glob(pattern)):
            return True
    return False


def _resolve_log_dir(raw_path: Path) -> Tuple[Path, str]:
    path = raw_path.expanduser().resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Path does not exist: {raw_path}")

    if _is_log_dir(path):
        return path, path.parent.name or path.name

    logs_dir = path / "logs"
    if logs_dir.is_dir() and _is_log_dir(logs_dir):
        return logs_dir, logs_dir.parent.name

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
        print(f"[ld_metrics] Auto-selected latest run: {selected.parent}")
        variant_guess = selected.parent.name
        return selected, variant_guess

    raise argparse.ArgumentTypeError(f"Could not resolve logs under {raw_path}")


def parse_run(value: str) -> Tuple[Path, str]:
    if ":" in value:
        path_part, variant_part = value.split(":", maxsplit=1)
        variant = variant_part.strip() or None
        log_dir, variant_guess = _resolve_log_dir(Path(path_part))
        if variant is None:
            variant = variant_guess
    else:
        log_dir, variant = _resolve_log_dir(Path(value))
    if not variant:
        raise argparse.ArgumentTypeError("Variant must not be empty")
    return log_dir, variant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot lambda-discrepancy metrics for training runs.")
    parser.add_argument(
        "--runs",
        metavar="LOG:VARIANT",
        type=parse_run,
        nargs="+",
        required=True,
        help="Log directory paired with variant label, e.g. experiments/.../logs:v4",
    )
    parser.add_argument("--max-data-points", type=int, default=None, help="Evenly sampled datapoints over run")
    parser.add_argument("--max-files", type=int, default=None, help="Alternative: limit number of files")
    parser.add_argument("--target-samples", type=int, default=5000, help="Reservoir size per variant")
    parser.add_argument("--output-subdir", type=str, default="ld_prototype", help="Subdirectory under plots/")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


def _detect_step_column(df: pd.DataFrame) -> str:
    for candidate in ("total_steps", "step", "timesteps", "global_step"):
        if candidate in df.columns:
            return candidate
    raise KeyError("No step column found in dataframe")


def _select_series(df: pd.DataFrame, *candidates: str) -> Optional[pd.Series]:
    for name in candidates:
        if name in df.columns:
            series = df[name]
            if isinstance(series, pd.Series):
                return series
    return None


def plot_ld_curves(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    run_dir: Path,
    output_subdir: str,
) -> Optional[Path]:
    step_col = _detect_step_column(df)
    steps = pd.to_numeric(df[step_col], errors="coerce")

    ld_term_raw = _select_series(df, "training.ld.term", "training_ld_term")
    ld_ratio_raw = _select_series(df, "training.ld.ratio", "training_ld_ratio")
    ld_component_raw = _select_series(df, "training.ld.component", "training_ld_component")
    if ld_term_raw is None or ld_ratio_raw is None or ld_component_raw is None:
        print(f"[ld_metrics] Missing LD columns for {variant} – skipping plot.")
        return None

    ld_term = pd.to_numeric(ld_term_raw, errors="coerce")
    ld_ratio = pd.to_numeric(ld_ratio_raw, errors="coerce")
    ld_component = pd.to_numeric(ld_component_raw, errors="coerce")

    valid_mask = ld_term.notna() & ld_ratio.notna() & ld_component.notna()
    if not valid_mask.any():
        print(f"[ld_metrics] No LD samples available for {variant} (all NaN); skipping plot.")
        return None

    ld_term = ld_term[valid_mask]
    ld_ratio = ld_ratio[valid_mask]
    ld_component = ld_component[valid_mask]
    steps = steps[valid_mask]

    reward_series = _select_series(df, "total_reward", "reward_total", "rewards_total")
    rewards: Optional[pd.Series] = None
    if reward_series is not None:
        rewards_numeric = pd.to_numeric(reward_series, errors="coerce")
        rewards = rewards_numeric.reindex(ld_term.index)

    plots_dir = run_dir / "plots" / output_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Plot LD term and ratio separately
    setup_plot_style()
    fig_term, ax_term = plt.subplots(figsize=(12, 4))
    ax_term.plot(steps, ld_term, label="LD Term", color="#ff7f0e")
    ax_term.set_xlabel(step_col)
    ax_term.set_ylabel("LD Term")
    ax_term.set_title(f"{variant.upper()} – LD Term over Time")
    ax_term.grid(True, alpha=0.3)
    ax_term.legend()
    fig_term.tight_layout()
    term_path = plots_dir / f"ld_term_{variant}_{experiment_name}.png"
    fig_term.savefig(term_path, dpi=300)
    plt.close(fig_term)

    fig_ratio, ax_ratio = plt.subplots(figsize=(12, 4))
    ax_ratio.plot(steps, ld_ratio, label="LD Ratio", color="#1f77b4")
    ax_ratio.set_xlabel(step_col)
    ax_ratio.set_ylabel("LD Ratio")
    ax_ratio.set_title(f"{variant.upper()} – LD Ratio over Time")
    ax_ratio.grid(True, alpha=0.3)
    ax_ratio.legend()
    fig_ratio.tight_layout()
    ratio_path = plots_dir / f"ld_ratio_{variant}_{experiment_name}.png"
    fig_ratio.savefig(ratio_path, dpi=300)
    plt.close(fig_ratio)

    # Plot LD component vs reward in separate figure
    fig_comp, ax_comp = plt.subplots(figsize=(12, 4))
    ax_comp.plot(steps, ld_component, label="LD Component", color="#d62728")
    ax_comp.set_xlabel(step_col)
    ax_comp.set_ylabel("LD Component", color="#d62728")
    ax_comp.tick_params(axis="y", labelcolor="#d62728")
    ax_comp.set_title(f"{variant.upper()} – LD Component vs Reward")
    ax_comp.grid(True, alpha=0.3)

    if rewards is not None and rewards.notna().any():
        reward_plot = rewards
        ax_reward = ax_comp.twinx()
        ax_reward.plot(steps, reward_plot, label="Reward Total", color="#2ca02c", alpha=0.6)
        ax_reward.set_ylabel("Reward Total", color="#2ca02c")
        ax_reward.tick_params(axis="y", labelcolor="#2ca02c")
        ax_reward.legend(loc="upper right")
    ax_comp.legend(loc="upper left")
    fig_comp.tight_layout()
    component_path = plots_dir / f"ld_component_reward_{variant}_{experiment_name}.png"
    fig_comp.savefig(component_path, dpi=300)
    plt.close(fig_comp)

    print(f"[ld_metrics] Saved LD term plot: {term_path}")
    print(f"[ld_metrics] Saved LD ratio plot: {ratio_path}")
    print(f"[ld_metrics] Saved LD component vs reward plot: {component_path}")
    return component_path


def run_cli(args: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    runs: List[Tuple[Path, str]] = parsed.runs
    variant_configs = [(str(path), variant) for path, variant in runs]
    data_frames, sampler = load_variants_for_comparison(
        variant_configs,
        target_samples=parsed.target_samples,
        max_files=parsed.max_files,
        max_data_points=parsed.max_data_points,
        normalize_steps=False,
        verbose=not parsed.quiet,
    )

    for (log_dir, variant), df in zip(runs, data_frames):
        run_dir = log_dir.parent
        experiment_name = run_dir.name
        if df.empty:
            print(f"[ld_metrics] No data for {variant}; skipping.")
            continue
        plot_ld_curves(
            df,
            variant=variant,
            experiment_name=experiment_name,
            run_dir=run_dir,
            output_subdir=parsed.output_subdir,
        )


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
