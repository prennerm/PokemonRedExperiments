"""Heatmap visualization utilities for sampled training data.

Ported from Pipeline_V2 and integrated into the library package so the
plotting helpers can be used programmatically and via the CLI.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_helpers import PLOT_CONFIG, save_plot, setup_plot_style
from .map_utils import MAP_DATA

HeatmapResult = Dict[str, np.ndarray]


def _resolve_position_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """Resolve column names for map, x and y coordinates."""
    column_map = {"map": None, "x": None, "y": None}

    for col in df.columns:
        lower = col.lower()

        if column_map["map"] is None:
            if lower in {"map", "map_id", "position.map", "position_map"}:
                column_map["map"] = col
            elif lower.endswith(".map") or lower.endswith("_map"):
                column_map["map"] = col
            elif lower.endswith(".map_id") or lower.endswith("_map_id"):
                column_map["map"] = col

        if column_map["x"] is None:
            if (
                "position.x" in lower
                or "position_x" in lower
                or (lower.endswith("_x") and "position" in lower)
                or lower == "x"
            ):
                column_map["x"] = col

        if column_map["y"] is None:
            if (
                "position.y" in lower
                or "position_y" in lower
                or (lower.endswith("_y") and "position" in lower)
                or lower == "y"
            ):
                column_map["y"] = col

    missing = [name for name, col in column_map.items() if col is None]
    if missing:
        raise KeyError(f"Missing expected position columns: {missing}")

    return column_map["map"], column_map["x"], column_map["y"]


def prepare_heatmap_data(
    df: pd.DataFrame,
    map_id: int,
    *,
    bins: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
) -> Optional[HeatmapResult]:
    """Prepare position counts for a specific map as 2D histogram data."""
    map_col, x_col, y_col = _resolve_position_columns(df)

    map_series = pd.to_numeric(df[map_col], errors="coerce")
    region = df[map_series == float(map_id)]
    if region.empty:
        return None

    x_values = region[x_col].to_numpy(dtype=float)
    y_values = region[y_col].to_numpy(dtype=float)

    x_min, x_max = np.nanmin(x_values), np.nanmax(x_values)
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)

    if bins is None:
        x_bins = max(int(x_max - x_min + 1), 1)
        y_bins = max(int(y_max - y_min + 1), 1)
    else:
        x_bins, y_bins = bins

    counts, x_edges, y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    matrix = counts / counts.max() if normalize and counts.max() > 0 else counts

    return {
        "matrix": matrix.T,
        "extent": (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        "counts": int(region.shape[0]),
        "bins": (x_bins, y_bins),
    }


def plot_position_heatmap(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    map_id: int,
    run_dir: Path,
    bins: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    cmap: str = "magma",
    colorbar: bool = True,
    verbose: bool = True,
) -> Optional[Path]:
    """Create and save a heatmap visualising agent visit density for a map."""
    heatmap_data = prepare_heatmap_data(df, map_id, bins=bins, normalize=normalize)
    if heatmap_data is None:
        if verbose:
            print(f"[heatmap_plots] No samples for map {map_id}; skipping.")
        return None

    setup_plot_style()

    fig, ax = plt.subplots(figsize=PLOT_CONFIG["figsize"])
    img = ax.imshow(
        heatmap_data["matrix"],
        origin="lower",
        cmap=cmap,
        aspect="auto",
        extent=heatmap_data["extent"],
    )

    map_label = MAP_DATA.get(int(map_id), {}).get("name", f"Map {map_id}")
    title = f"{variant.upper()} - {map_label} - Visit Density\n{experiment_name}"
    ax.set_title(title, fontsize=PLOT_CONFIG["title_size"])
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(False)

    if colorbar:
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Visits (normalised)" if normalize else "Visits", rotation=270, labelpad=15)

    plots_dir = Path(run_dir) / "plots" / "heatmaps"
    filename = f"heatmap_{variant}_{experiment_name}_map{map_id}"
    save_plot(fig, plots_dir, filename, formats=["png"])
    plt.close(fig)

    result_path = plots_dir / f"{filename}.png"
    if verbose:
        print(
            f"[heatmap_plots] Heatmap saved for map {map_id}: {result_path} "
            f"(samples={heatmap_data['counts']:,}, bins={heatmap_data['bins']})"
        )
    return result_path


def plot_all_maps(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    run_dir: Path,
    maps: Optional[Iterable[int]] = None,
    bins: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    cmap: str = "magma",
) -> Dict[int, Path]:
    """Generate heatmaps for multiple maps present in the dataset."""
    map_col, _, _ = _resolve_position_columns(df)
    map_series = pd.to_numeric(df[map_col], errors="coerce")
    map_ids = (
        maps
        if maps is not None
        else sorted(map_series.dropna().astype(int).unique().tolist())
    )

    outputs: Dict[int, Path] = {}
    for target_map in map_ids:
        plot_path = plot_position_heatmap(
            df,
            variant=variant,
            experiment_name=experiment_name,
            map_id=int(target_map),
            run_dir=run_dir,
            bins=bins,
            normalize=normalize,
            cmap=cmap,
            verbose=False,
        )
        if plot_path is not None:
            outputs[int(target_map)] = plot_path
    return outputs


__all__ = [
    "prepare_heatmap_data",
    "plot_position_heatmap",
    "plot_all_maps",
]
