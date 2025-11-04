#!/usr/bin/env python3
"""
Generate cumulative heatmap animations for one or multiple training runs.

Example usage:

  conda run -n poke_viz python notebooks/animated_heatmaps.py \
      --runs experiments/v4_ld_01/20251030_165359/logs:v4 \
      --max-data-points 2000 \
      --map-ids 0 12 37 \
      --frames 50 \
      --fps 10
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from poke_pipeline.visualization import MAP_DATA, load_variants_for_comparison, setup_plot_style
from poke_pipeline.visualization.heatmap_plots import prepare_heatmap_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create cumulative animated heatmaps for training runs.")
    parser.add_argument(
        "--runs",
        metavar="LOG:VARIANT",
        type=str,
        nargs="+",
        required=True,
        help="Log directory paired with variant label, e.g. experiments/.../logs:v4",
    )
    parser.add_argument("--max-data-points", type=int, default=None, help="Evenly sampled datapoints over run")
    parser.add_argument("--max-files", type=int, default=None, help="Alternate: limit number of files (exclusive)")
    parser.add_argument("--target-samples", type=int, default=2000, help="Reservoir size per variant")
    parser.add_argument("--frames", type=int, default=40, help="Number of frames per animation (cumulative)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for exported animation")
    parser.add_argument("--map-ids", type=int, nargs="+", default=None, help="Specific map ids to animate")
    parser.add_argument("--normalize", action="store_true", help="Normalize heatmap intensities per frame")
    parser.add_argument("--overlay-map", type=Path, default=None, help="Optional stitched map PNG for overlay")
    parser.add_argument("--output-dir", type=str, default="animations", help="Subdir under run/plots/ for outputs")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Output format (requires pillow/mp4)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser.parse_args()


class OverlayHelper:
    """Optional helper to crop map backgrounds from a stitched world sprite."""

    def __init__(self, image_path: Path, verbose: bool = True) -> None:
        from PIL import Image

        if not image_path:
            raise ValueError("Image path required")
        if not image_path.is_file():
            raise FileNotFoundError(f"Overlay map not found: {image_path}")
        self.verbose = verbose
        self.image = Image.open(image_path).convert("RGBA")
        world_meta = MAP_DATA.get(-1)
        if not world_meta:
            raise RuntimeError("MAP_DATA missing global entry (-1).")
        tile_width, tile_height = world_meta.get("tileSize", (436, 444))
        self.scale_x = self.image.width / tile_width
        self.scale_y = self.image.height / tile_height
        if self.verbose:
            print(
                f"[Overlay] loaded {self.image.size[0]}x{self.image.size[1]}"
                f" (scale≈{self.scale_x:.2f}/{self.scale_y:.2f})"
            )

    def crop(self, map_id: int) -> Optional[np.ndarray]:
        from PIL import Image

        meta = MAP_DATA.get(map_id)
        if not meta:
            if self.verbose:
                print(f"[Overlay] map_id {map_id} not in MAP_DATA.")
            return None
        coords = meta.get("coordinates")
        size = meta.get("tileSize")
        if not coords or not size:
            if self.verbose:
                print(f"[Overlay] map_id {map_id} missing coordinate metadata.")
            return None
        x, y = coords
        w, h = size
        left = int(round(x * self.scale_x))
        top = int(round(y * self.scale_y))
        right = int(round((x + w) * self.scale_x))
        bottom = int(round((y + h) * self.scale_y))
        left = max(0, min(left, self.image.width))
        right = max(left, min(right, self.image.width))
        top = max(0, min(top, self.image.height))
        bottom = max(top, min(bottom, self.image.height))
        if right - left == 0 or bottom - top == 0:
            if self.verbose:
                print(f"[Overlay] empty crop for map {map_id}")
            return None
        cropped = self.image.crop((left, top, right, bottom))
        return np.asarray(cropped)


def _detect_map_column(df: pd.DataFrame) -> str:
    for candidate in ("map_id", "map", "position_map", "position.map"):
        if candidate in df.columns:
            return candidate
    raise KeyError("map column not found in dataframe")


def _prepare_frames(
    df: pd.DataFrame,
    map_id: int,
    *,
    frames: int,
    normalize: bool,
    cumulative: bool = True,
) -> Tuple[List[Dict[str, np.ndarray]], List[Tuple[int, int]]]:
    if frames <= 0:
        raise ValueError("frames must be > 0")

    map_column = _detect_map_column(df)
    map_df = df[pd.to_numeric(df[map_column], errors="coerce") == float(map_id)]
    if map_df.empty:
        return [], []

    step_col = None
    for candidate in ("total_steps", "step", "timesteps", "global_step"):
        if candidate in map_df.columns:
            step_col = candidate
            break
    if step_col is None:
        step_col = "step"
    steps = pd.to_numeric(map_df[step_col], errors="coerce").fillna(0)
    map_df = map_df.assign(_steps=steps).sort_values("_steps")

    step_min = int(map_df["_steps"].min())
    step_max = int(map_df["_steps"].max())
    if step_max <= step_min:
        boundaries = [step_max] * frames
    else:
        boundaries = np.linspace(step_min, step_max, frames + 1, dtype=int)[1:]

    cumulative_frames: List[pd.DataFrame] = []
    frame_step_ranges: List[Tuple[int, int]] = []
    current_df = pd.DataFrame()

    for upper in boundaries:
        if cumulative:
            current_df = map_df[map_df["_steps"] <= upper]
        else:
            lower = boundaries[0] if len(frame_step_ranges) == 0 else frame_step_ranges[-1][1]
            current_df = map_df[(map_df["_steps"] > lower) & (map_df["_steps"] <= upper)]
        cumulative_frames.append(current_df.copy())
        lower_bound = step_min if cumulative else (frame_step_ranges[-1][1] if frame_step_ranges else step_min)
        frame_step_ranges.append((lower_bound, upper))

    heatmap_frames = []
    for frame_df in cumulative_frames:
        frame_data = prepare_heatmap_data(frame_df, map_id, normalize=normalize)
        heatmap_frames.append(frame_data)

    return heatmap_frames, frame_step_ranges


def _animate_heatmap(
    frames: List[Dict[str, np.ndarray]],
    step_ranges: List[Tuple[int, int]],
    *,
    variant: str,
    experiment_name: str,
    map_id: int,
    run_dir: Path,
    overlay_helper: Optional[OverlayHelper],
    fps: int,
    cmap: str = "magma",
    fmt: str = "gif",
) -> Optional[Path]:
    if not frames or all(frame is None for frame in frames):
        print(f"[animate] No data for map {map_id}; skipping animation.")
        return None

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))

    first_valid = next((frame for frame in frames if frame), None)
    if first_valid is None:
        print(f"[animate] No valid heatmap frames for map {map_id}")
        return None

    extent = first_valid["extent"]
    background = overlay_helper.crop(map_id) if overlay_helper else None
    if background is not None:
        ax.imshow(background, origin="upper", extent=extent, alpha=0.5)

    heat_img = ax.imshow(
        first_valid["matrix"],
        origin="lower",
        cmap=cmap,
        aspect="auto",
        extent=extent,
    )
    cbar = fig.colorbar(heat_img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Visits", rotation=270, labelpad=15)

    map_label = MAP_DATA.get(map_id, {}).get("name", f"Map {map_id}")
    base_title = f"{variant.upper()} – {map_label}\n{experiment_name}"
    ax.set_title(base_title, fontsize=14, pad=16)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")

    def update(frame_index: int):
        frame = frames[frame_index]
        if frame is None:
            heat_img.set_data(np.zeros_like(first_valid["matrix"]))
        else:
            heat_img.set_data(frame["matrix"])
        lower, upper = step_ranges[frame_index]
        ax.set_title(f"{base_title}\nSteps {lower:,} – {upper:,}", fontsize=14, pad=16)
        return (heat_img,)

    writer_cls = animation.PillowWriter if fmt == "gif" else animation.FFMpegWriter
    writer_kwargs = {"fps": fps}
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / max(fps, 1),
        blit=False,
    )

    output_dir = run_dir / "plots" / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"heatmap_anim_{variant}_{experiment_name}_map{map_id}.{fmt}"

    try:
        anim.save(filename, writer=writer_cls(**writer_kwargs))
        print(f"[animate] Saved animation for map {map_id}: {filename}")
    except Exception as exc:
        print(f"[animate] Failed to save animation for map {map_id}: {exc}")
        filename = None
    finally:
        plt.close(fig)

    return filename


def main() -> None:
    args = parse_args()
    if args.max_files is not None and args.max_data_points is not None:
        raise ValueError("Use either --max-files or --max-data-points (not both).")

    overlay_helper = None
    if args.overlay_map:
        overlay_helper = OverlayHelper(args.overlay_map, verbose=not args.quiet)

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
        run_dir = log_dir.parent
        experiment_name = run_dir.name
        if df.empty:
            print(f"[animated] No data sampled for {variant}; skipping.")
            continue

        map_column = _detect_map_column(df)
        counts = df[map_column].dropna().astype(int).value_counts().sort_values(ascending=False)
        if args.map_ids:
            map_ids: Sequence[int] = args.map_ids
        else:
            top_n = min(5, len(counts))
            map_ids = list(counts.head(top_n).index)

        print(f"[animated] Variant {variant} -> animating maps {map_ids}")

        for map_id in map_ids:
            frames, step_ranges = _prepare_frames(
                df,
                map_id=map_id,
                frames=args.frames,
                normalize=args.normalize,
                cumulative=True,
            )
            _animate_heatmap(
                frames,
                step_ranges,
                variant=variant,
                experiment_name=experiment_name,
                map_id=map_id,
                run_dir=run_dir,
                overlay_helper=overlay_helper,
                fps=args.fps,
                fmt=args.format,
            )


if __name__ == "__main__":
    main()
