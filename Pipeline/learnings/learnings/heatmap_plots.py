"""
Heatmap visualization utilities for Pokemon Red RL training analysis.

These helpers operate on DataFrames produced by ``TrainingSampler`` and allow
creating position-density heatmaps for individual map regions.  The intent is
to provide quick insight into exploration coverage without any additional
post-processing scripts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.plot_helpers import (
    PLOT_CONFIG,
    save_plot,
    setup_plot_style,
)
from utils.map_utils import MAP_DATA

# Type alias for clarity
HeatmapResult = Dict[str, np.ndarray]


def _resolve_position_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    """
    Resolve column names for map, x and y coordinates.

    Returns:
        Tuple of (map_col, x_col, y_col)
    """
    column_map = {
        "map": None,
        "x": None,
        "y": None,
    }

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
    """
    Prepare position counts for a specific map as 2D histogram data.

    Args:
        df: DataFrame containing the training samples.
        map_id: Map identifier to filter for (matching ``position.map``).
        bins: Optional tuple specifying (x_bins, y_bins).  If omitted the
            algorithm derives bin counts from the observed coordinate ranges.
        normalize: Whether to normalise the heatmap values to [0, 1].

    Returns:
        Dictionary with keys ``matrix`` (2D numpy array), ``extent`` (tuple
        for imshow), ``counts`` (total samples for the map) and ``bins``.
        Returns ``None`` if the filtered data frame is empty.
    """
    map_col, x_col, y_col = _resolve_position_columns(df)

    map_series = pd.to_numeric(df[map_col], errors="coerce")
    region = df[map_series == float(map_id)]
    if region.empty:
        return None

    x_values = region[x_col].to_numpy(dtype=float)
    y_values = region[y_col].to_numpy(dtype=float)

    x_min, x_max = np.nanmin(x_values), np.nanmax(x_values)
    y_min, y_max = np.nanmin(y_values), np.nanmax(y_values)

    # Derive bins from coordinate range if not provided
    if bins is None:
        x_bins = max(int(x_max - x_min + 1), 1)
        y_bins = max(int(y_max - y_min + 1), 1)
    else:
        x_bins, y_bins = bins

    # histogram2d expects x first; transpose later for imshow orientation
    counts, x_edges, y_edges = np.histogram2d(
        x_values,
        y_values,
        bins=[x_bins, y_bins],
        range=[[x_min, x_max], [y_min, y_max]],
    )

    if normalize and counts.max() > 0:
        matrix = counts / counts.max()
    else:
        matrix = counts

    heatmap = {
        "matrix": matrix.T,  # transpose so that x corresponds to horizontal axis
        "extent": (x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        "counts": int(region.shape[0]),
        "bins": (x_bins, y_bins),
    }
    return heatmap


def plot_position_heatmap(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    map_id: int,
    output_dir: Path,
    bins: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    cmap: str = "magma",
    colorbar: bool = True,
    verbose: bool = True,
) -> Optional[Path]:
    """
    Create and save a heatmap visualising agent visit density for a map.

    Args:
        df: DataFrame with training samples.
        variant: Variant identifier (e.g. ``"v4"``).
        experiment_name: Timestamp or descriptor of the experiment.
        map_id: Map identifier to filter.
        output_dir: Base directory for saving plots (heatmaps folder will be created inside).
        bins: Optional bin specification passed to :func:`prepare_heatmap_data`.
        normalize: Normalise heatmap values to [0, 1].
        cmap: Matplotlib colourmap name.
        colorbar: Attach a colourbar to the figure.
        verbose: Print informative messages.

    Returns:
        Path to the generated plot file, or ``None`` if no data was available.
    """
    heatmap_data = prepare_heatmap_data(df, map_id, bins=bins, normalize=normalize)
    if heatmap_data is None:
        if verbose:
            print(f"[WARN] No samples for map {map_id}; skipping heatmap.")
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

    output_path = Path(output_dir) / "heatmaps"
    filename = f"heatmap_{variant}_{experiment_name}_map{map_id}"
    save_plot(fig, output_path, filename, formats=["png"])
    plt.close(fig)

    result_path = output_path / f"{filename}.png"
    if verbose:
        print(
            f"[OK] Heatmap saved for map {map_id}: {result_path} "
            f"(samples={heatmap_data['counts']:,}, bins={heatmap_data['bins']})"
        )
    return result_path


def plot_all_maps(
    df: pd.DataFrame,
    *,
    variant: str,
    experiment_name: str,
    output_dir: Path,
    maps: Optional[Iterable[int]] = None,
    bins: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    cmap: str = "magma",
) -> Dict[int, Path]:
    """
    Generate heatmaps for multiple maps present in the dataset.

    Args:
        df: Sampled training data.
        variant: Variant identifier.
        experiment_name: Experiment descriptor (used in filenames).
        output_dir: Base directory for plots.
        maps: Optional iterable of map IDs to render; if ``None`` all unique IDs found in the data are used.
        bins: Bin specification passed to :func:`plot_position_heatmap`.
        normalize: Normalise heatmap values.
        cmap: Colourmap for all heatmaps.

    Returns:
        Dictionary mapping map IDs to generated plot paths (maps with no data are omitted).
    """
    map_col, _, _ = _resolve_position_columns(df)
    map_series = pd.to_numeric(df[map_col], errors="coerce")
    map_ids = (
        maps
        if maps is not None
        else sorted(map_series.dropna().astype(int).unique().tolist())
    )

    outputs: Dict[int, Path] = {}
    for map_id in map_ids:
        path = plot_position_heatmap(
            df,
            variant=variant,
            experiment_name=experiment_name,
            map_id=int(map_id),
            output_dir=output_dir,
            bins=bins,
            normalize=normalize,
            cmap=cmap,
            verbose=False,
        )
        if path is not None:
            outputs[int(map_id)] = path
    return outputs
