#!/usr/bin/env python3
"""
Quick prototype to overlay sampled heatmaps with a stitched world-map background.

Usage example (from repository root):

    python notebooks/overlay_heatmap_prototype.py \
        --runs experiments/v4_ld_01/20251030_165359/logs:v4 \
        --overlay-map data/kanto_world_7200.png \
        --map-ids 0 12 37 \
        --max-data-points 2000 \
        --output-dir overlay_prototype/v4_ld_01

The script will:
  * Sample the run using the existing streaming sampler (respecting max-data-points or max-files).
  * For each requested map id, crop the corresponding area from the large map sprite using MAP_DATA
    metadata and render the heatmap on top of it.
  * Store the resulting figures under `<run_dir>/plots/<output-subdir>/`.

Note: This prototype does not alter the production visualization package; it is a sandbox to validate
      overlay feasibility before we integrate the feature properly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from poke_pipeline.visualization import (
    MAP_DATA,
    GLOBAL_MAP_SHAPE,
    load_variants_for_comparison,
    setup_plot_style,
    save_plot,
)
from poke_pipeline.visualization.heatmap_plots import prepare_heatmap_data


def _detect_map_column(df) -> str:
    """Return the name of the map column in the sampled dataframe."""
    for candidate in ("map_id", "map", "position_map", "position.map"):
        if candidate in df.columns:
            return candidate
    raise KeyError("Could not detect map column in dataframe")


class MapOverlayHelper:
    """Utility to crop and serve map tiles from a stitched world sprite."""

    def __init__(self, map_image_path: Path, verbose: bool = True) -> None:
        if not map_image_path.is_file():
            raise FileNotFoundError(f"Overlay map not found: {map_image_path}")
        self.image = Image.open(map_image_path).convert("RGBA")
        self.verbose = verbose

        world_meta = MAP_DATA.get(-1)
        if not world_meta:
            raise RuntimeError("MAP_DATA does not contain the global (-1) entry.")
        tile_width, tile_height = world_meta.get("tileSize", GLOBAL_MAP_SHAPE[::-1])

        self.scale_x = self.image.width / tile_width
        self.scale_y = self.image.height / tile_height

        if self.verbose:
            print(
                f"[Overlay] Loaded map sprite {self.image.size[0]}x{self.image.size[1]} "
                f"(scale factors ≈ {self.scale_x:.3f}, {self.scale_y:.3f})"
            )

    def crop_for_map(self, map_id: int) -> Optional[np.ndarray]:
        """Return a cropped RGBA array for the requested map id."""
        meta = MAP_DATA.get(map_id)
        if not meta:
            if self.verbose:
                print(f"[Overlay] map_id {map_id} not found in MAP_DATA.")
            return None

        coords = meta.get("coordinates")
        size = meta.get("tileSize")
        if not coords or not size:
            if self.verbose:
                print(f"[Overlay] map_id {map_id} missing 'coordinates' or 'tileSize'.")
            return None

        x, y = coords
        w, h = size
        left = int(round(x * self.scale_x))
        top = int(round(y * self.scale_y))
        right = int(round((x + w) * self.scale_x))
        bottom = int(round((y + h) * self.scale_y))

        # Guard against boundaries
        left = max(0, min(left, self.image.width))
        right = max(left, min(right, self.image.width))
        top = max(0, min(top, self.image.height))
        bottom = max(top, min(bottom, self.image.height))

        if right - left == 0 or bottom - top == 0:
            if self.verbose:
                print(f"[Overlay] Computed empty crop for map {map_id} (check metadata).")
            return None

        cropped = self.image.crop((left, top, right, bottom))
        return np.asarray(cropped)


def plot_heatmap_with_overlay(
    df,
    map_id: int,
    variant: str,
    experiment_name: str,
    run_dir: Path,
    overlay_helper: MapOverlayHelper,
    *,
    normalize: bool = False,
    cmap: str = "magma",
    alpha: float = 0.55,
    output_subdir: str = "overlay_prototype",
) -> Optional[Path]:
    """Render a heatmap for map_id with the cropped overlay underneath."""
    heatmap_data = prepare_heatmap_data(df, map_id, normalize=normalize)
    if heatmap_data is None:
        print(f"[Overlay] No samples for map {map_id}; skipping.")
        return None

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    background = overlay_helper.crop_for_map(map_id)
    if background is not None:
        ax.imshow(
            background,
            origin="upper",
            extent=(
                heatmap_data["extent"][0],
                heatmap_data["extent"][1],
                heatmap_data["extent"][2],
                heatmap_data["extent"][3],
            ),
            alpha=alpha,
        )

    masked_matrix = np.ma.masked_where(heatmap_data["matrix"] <= 0, heatmap_data["matrix"])

    base_cmap = plt.colormaps[cmap]
    cmap_with_alpha = base_cmap(np.linspace(0, 1, base_cmap.N))
    cmap_with_alpha[:, -1] = 0.5
    transparent_cmap = plt.matplotlib.colors.ListedColormap(cmap_with_alpha)
    transparent_cmap.set_bad(alpha=0.0)

    ax.imshow(
        masked_matrix,
        origin="lower",
        cmap=transparent_cmap,
        aspect="auto",
        extent=heatmap_data["extent"],
    )

    meta = MAP_DATA.get(map_id, {})
    map_label = meta.get("name", f"Map {map_id}")
    ax.set_title(f"{variant.upper()} - {map_label}\n{experiment_name}", fontsize=14)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    plots_dir = run_dir / "plots" / output_subdir
    filename = f"overlay_heatmap_{variant}_map{map_id}"
    save_plot(fig, plots_dir, filename, formats=["png"])
    plt.close(fig)

    output_path = plots_dir / f"{filename}.png"
    print(f"[Overlay] Saved overlay heatmap for map {map_id}: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prototype: Heatmap overlay with stitched world sprite.")
    parser.add_argument(
        "--runs",
        metavar="LOG:VARIANT",
        type=str,
        nargs="+",
        required=True,
        help="Log path paired with variant label (e.g. experiments/.../logs:v4)",
    )
    parser.add_argument("--overlay-map", type=Path, required=True, help="Path to stitched world-map PNG.")
    parser.add_argument(
        "--map-ids",
        type=int,
        nargs="+",
        default=None,
        help="Specific map ids to render. If omitted, the script uses the most frequent maps in the data.",
    )
    parser.add_argument("--max-files", type=int, default=None, help="Optional: limit number of stats files.")
    parser.add_argument("--max-data-points", type=int, default=2000, help="Optional: evenly sampled data points.")
    parser.add_argument("--target-samples", type=int, default=1000, help="Reservoir size for sampling.")
    parser.add_argument("--normalize-heatmaps", action="store_true", help="Normalize heatmap intensities.")
    parser.add_argument("--alpha", type=float, default=0.55, help="Overlay transparency (0..1).")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="overlay_prototype",
        help="Subdirectory under run_dir/plots/ for the prototype outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_files is not None and args.max_data_points is not None:
        raise ValueError("Use either --max-files or --max-data-points (but not both) for this prototype.")

    overlay_helper = MapOverlayHelper(args.overlay_map, verbose=not args.quiet)

    variant_configs = []
    for item in args.runs:
        path_str, variant = item.split(":", maxsplit=1)
        variant_configs.append((path_str, variant))

    data_frames, sampler = load_variants_for_comparison(
        variant_configs,
        target_samples=args.target_samples,
        max_files=args.max_files,
        max_data_points=args.max_data_points,
        normalize_steps=False,
        verbose=not args.quiet,
    )

    for (log_dir_str, variant), df in zip(variant_configs, data_frames):
        log_dir = Path(log_dir_str)
        if df.empty:
            print(f"[Overlay] No data sampled for {variant} @ {log_dir}; skipping.")
            continue

        map_column = _detect_map_column(df)
        available_maps = (
            df[map_column].dropna().astype(int).value_counts().sort_values(ascending=False)
        )
        if args.map_ids:
            target_maps: Sequence[int] = args.map_ids
        else:
            target_maps = list(available_maps.head(5).index)

        experiment_dir = log_dir.parent
        experiment_name = experiment_dir.name
        print(
            f"[Overlay] Rendering maps {target_maps} for variant {variant} "
            f"(run {experiment_name}, {len(df)} samples)."
        )

        for map_id in target_maps:
            plot_heatmap_with_overlay(
                df,
                map_id=map_id,
                variant=variant,
                experiment_name=experiment_name,
                run_dir=experiment_dir,
                overlay_helper=overlay_helper,
                normalize=args.normalize_heatmaps,
                alpha=args.alpha,
                output_subdir=args.output_dir,
            )


if __name__ == "__main__":
    main()
