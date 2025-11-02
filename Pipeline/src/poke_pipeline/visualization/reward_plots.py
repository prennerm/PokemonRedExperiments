"""Reward plotting utilities built on top of the shared visualization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from .plot_helpers import (
    PLOT_CONFIG,
    REWARD_COLORS,
    create_plot_title,
    get_reward_columns,
    save_plot,
    setup_plot_style,
    smooth_series,
)


def plot_rewards_over_time(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    run_dir: Path,
    window: int = 200,
    normalize_steps: bool = False,
    formats: Optional[list[str]] = None,
    samples: Optional[int] = None,
    step_range: Optional[Tuple[int, int]] = None,
    sampling_ratio: Optional[float] = None,
) -> Optional[Path]:
    """Plot reward components over time and save the figure."""
    reward_cols = get_reward_columns(df)
    if not reward_cols:
        print("[reward_plots] No reward columns detected; skipping plot.")
        return None

    step_col = None
    for candidate in ("total_steps", "step", "timesteps", "global_step"):
        if candidate in df.columns:
            step_col = candidate
            break
    if step_col is None:
        print("[reward_plots] No step column detected; skipping plot.")
        return None

    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])

    steps = pd.to_numeric(df[step_col], errors="coerce")
    if normalize_steps and steps.max() > 0:
        x_values = steps / steps.max()
        x_label = "Progress (0-1)"
    else:
        x_values = steps
        x_label = "Total steps"

    for key, column in reward_cols.items():
        series = pd.to_numeric(df[column], errors="coerce").fillna(0.0)
        smoothed = smooth_series(series, window=window)
        color = REWARD_COLORS.get(key, None)
        label = key.capitalize()
        ax.plot(x_values, smoothed, label=label, color=color)

    title = create_plot_title(variant, experiment_name, additional="Reward Components")
    total_samples = samples if samples is not None else len(df)
    subtitle_parts = [f"samples={total_samples:,}"]
    valid_steps = steps.dropna()
    computed_step_range = step_range
    if computed_step_range is None and not valid_steps.empty:
        computed_step_range = (int(valid_steps.min()), int(valid_steps.max()))
    if computed_step_range is not None:
        subtitle_parts.append(
            f"steps={computed_step_range[0]:,}-{computed_step_range[1]:,}"
        )
    if sampling_ratio is not None:
        subtitle_parts.append(f"sampling={sampling_ratio * 100:.2f}%")
    subtitle = " | ".join(subtitle_parts)
    if subtitle:
        full_title = f"{title}\n{subtitle}"
    else:
        full_title = title
    ax.set_title(full_title, fontsize=PLOT_CONFIG["title_size"], pad=12)
    ax.set_xlabel(x_label, fontsize=PLOT_CONFIG["label_size"])
    ax.set_ylabel("Reward (smoothed)", fontsize=PLOT_CONFIG["label_size"])
    ax.legend(fontsize=PLOT_CONFIG["legend_size"])
    ax.grid(True, alpha=PLOT_CONFIG["grid_alpha"])

    plots_dir = Path(run_dir) / "plots"
    filename = f"rewards_{variant}_{experiment_name}"
    save_plot(fig, plots_dir, filename, formats=formats or ["png"])
    plt.close(fig)

    return plots_dir / f"{filename}.png"


__all__ = ["plot_rewards_over_time"]
