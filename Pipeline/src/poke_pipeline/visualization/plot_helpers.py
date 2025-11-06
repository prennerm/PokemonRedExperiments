#!/usr/bin/env python3
"""Common plotting helpers shared across visualization modules.

Ported from Pipeline_V2/utils/plot_helpers.py and adapted to live inside the
``poke_pipeline.visualization`` package.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches  # noqa: F401 (kept for parity with V2 helpers)
import numpy as np  # noqa: F401
import pandas as pd
import seaborn as sns

# Configuration constants reused across plots
PLOT_CONFIG: Dict[str, object] = {
    "figsize": (12, 8),
    "dpi": 300,
    "style": "whitegrid",
    "palette": "tab10",
    "font_size": 12,
    "title_size": 14,
    "label_size": 11,
    "legend_size": 10,
    "line_width": 1.5,
    "alpha": 0.7,
    "grid_alpha": 0.3,
}

REWARD_COLORS: Dict[str, str] = {
    "total": "#1f77b4",
    "event": "#ff7f0e",
    "level": "#2ca02c",
    "heal": "#d62728",
    "badge": "#9467bd",
    "explore": "#8c564b",
    "dead": "#e377c2",
    "stuck": "#7f7f7f",
}

OUTPUT_FORMATS = ["png", "pdf", "svg"]


def setup_plot_style() -> None:
    """Configure global matplotlib/seaborn styling."""
    plt.style.use("default")
    sns.set_style(PLOT_CONFIG["style"])
    sns.set_palette(PLOT_CONFIG["palette"])

    plt.rcParams.update(
        {
            "font.size": PLOT_CONFIG["font_size"],
            "axes.titlesize": PLOT_CONFIG["title_size"],
            "axes.labelsize": PLOT_CONFIG["label_size"],
            "legend.fontsize": PLOT_CONFIG["legend_size"],
            "figure.dpi": PLOT_CONFIG["dpi"],
            "savefig.dpi": PLOT_CONFIG["dpi"],
            "figure.figsize": PLOT_CONFIG["figsize"],
        }
    )


def smooth_series(series: pd.Series, window: int = 100) -> pd.Series:
    """Apply a centred moving average to smooth noisy time series."""
    if len(series) < window:
        window = max(1, len(series) // 4)
    return series.rolling(window=window, center=True, min_periods=1).mean()


def get_reward_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Identify reward-related columns in a DataFrame."""
    mapping: Dict[str, str] = {}
    for col in df.columns:
        col_lower = col.lower()
        if "reward" not in col_lower:
            continue
        if any(tag in col_lower for tag in ("total", "sum")):
            mapping.setdefault("total", col)
        elif "event" in col_lower:
            mapping["event"] = col
        elif "level" in col_lower:
            mapping["level"] = col
        elif "heal" in col_lower:
            mapping["heal"] = col
        elif "badge" in col_lower:
            mapping["badge"] = col
        elif "explore" in col_lower:
            mapping["explore"] = col
        elif "dead" in col_lower:
            mapping["dead"] = col
        elif "stuck" in col_lower:
            mapping["stuck"] = col
        else:
            mapping.setdefault("total", col)
    return mapping


def create_plot_title(variant: str, timestamp: str, additional: str = "") -> str:
    """Compose a consistent plot title."""
    base = f"Agent {variant.upper()} - Training {timestamp}"
    return f"{base} - {additional}" if additional else base


def save_plot(fig: plt.Figure, output_path: Path, filename: str, formats: Optional[List[str]] = None) -> None:
    """Persist a figure using the configured formats."""
    formats = formats or ["png"]
    output_path.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        if fmt not in OUTPUT_FORMATS:
            print(f"[plot_helpers] Warning: unsupported format '{fmt}', skipping.")
            continue
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, bbox_inches="tight", dpi=PLOT_CONFIG["dpi"], facecolor="white")
        print(f"[plot_helpers] Plot saved: {filepath}")


__all__ = [
    "PLOT_CONFIG",
    "REWARD_COLORS",
    "OUTPUT_FORMATS",
    "setup_plot_style",
    "smooth_series",
    "get_reward_columns",
    "create_plot_title",
    "save_plot",
]
