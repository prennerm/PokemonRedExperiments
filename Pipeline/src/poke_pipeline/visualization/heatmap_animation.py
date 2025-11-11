"""
CLI utilities for generating cumulative heatmap animations from training logs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from .data_sampling import load_variants_for_comparison
from .heatmap_plots import prepare_heatmap_data
from .map_utils import MAP_DATA
from .plot_helpers import setup_plot_style


def build_arg_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--max-steps", type=int, default=100_000_000, help="Maximum step value to consider")
    parser.add_argument("--frames", type=int, default=40, help="Number of frames per animation (cumulative)")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for exported animation")
    parser.add_argument("--map-ids", type=int, nargs="+", default=None, help="Specific map ids to animate")
    parser.add_argument("--normalize", action="store_true", help="Normalize heatmap intensities per frame")
    parser.add_argument("--overlay-map", type=Path, default=None, help="Optional stitched map PNG for overlay")
    parser.add_argument("--output-dir", type=str, default="animations", help="Subdir under run/plots/ for outputs")
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif", help="Output format (requires pillow/mp4)")
    parser.add_argument("--quiet", action="store_true", help="Reduce console output")
    return parser


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


def _split_into_frames(
    df: pd.DataFrame,
    map_id: int,
    *,
    frames: int,
    normalize: bool,
    cumulative: bool = True,
) -> Tuple[List[Optional[Dict[str, np.ndarray]]], List[Tuple[int, int]]]:
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

    for index, upper in enumerate(boundaries):
        if cumulative:
            current_df = map_df[map_df["_steps"] <= upper]
        else:
            lower_bound = frame_step_ranges[-1][1] if frame_step_ranges else step_min
            current_df = map_df[(map_df["_steps"] > lower_bound) & (map_df["_steps"] <= upper)]
        cumulative_frames.append(current_df.copy())
        lower_bound = step_min if cumulative else (frame_step_ranges[-1][1] if frame_step_ranges else step_min)
        frame_step_ranges.append((lower_bound, int(upper)))

    heatmap_frames: List[Optional[Dict[str, np.ndarray]]] = []
    for frame_df in cumulative_frames:
        frame_data = prepare_heatmap_data(frame_df, map_id, normalize=normalize)
        heatmap_frames.append(frame_data)

    return heatmap_frames, frame_step_ranges


def _animate_heatmap(
    frames: List[Optional[Dict[str, np.ndarray]]],
    step_ranges: List[Tuple[int, int]],
    *,
    variant: str,
    experiment_name: str,
    map_id: int,
    run_dir: Path,
    overlay_helper: Optional[OverlayHelper],
    fps: int,
    fmt: str,
    output_subdir: str,
) -> Optional[Path]:
    if not frames:
        print(f"[heatmap_animation] No frames for map {map_id}, skipping animation.")
        return None

    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    cmap = plt.get_cmap("inferno")
    first_valid = next((frame for frame in frames if frame), None)
    if first_valid is None:
        print(f"[heatmap_animation] No valid heatmap frames for map {map_id}")
        plt.close(fig)
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
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

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

    output_dir = run_dir / "plots" / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / f"heatmap_anim_{variant}_{experiment_name}_map{map_id}.{fmt}"

    try:
        anim.save(filename, writer=writer_cls(**writer_kwargs))
        print(f"[heatmap_animation] Saved animation for map {map_id}: {filename}")
    except Exception as exc:  # pragma: no cover - side effect logging
        print(f"[heatmap_animation] Failed to save animation for map {map_id}: {exc}")
        filename = None
    finally:
        plt.close(fig)

    return filename


def _prepare_runs(
    run_items: Iterable[str],
    *,
    target_samples: int,
    max_files: Optional[int],
    max_data_points: Optional[int],
    verbose: bool,
    max_steps: int,
) -> Tuple[List[Tuple[Path, str]], List[pd.DataFrame]]:
    variant_configs: List[Tuple[str, str]] = []
    resolved_runs: List[Tuple[Path, str]] = []
    for item in run_items:
        path_str, variant = item.split(":", maxsplit=1)
        path = Path(path_str)
        resolved_runs.append((path, variant))
        variant_configs.append((str(path), variant))

    data_frames, sampler = load_variants_for_comparison(
        variant_configs,
        target_samples=target_samples,
        max_files=max_files,
        max_data_points=max_data_points,
        normalize_steps=False,
        verbose=verbose,
        max_steps=max_steps,
    )
    return resolved_runs, data_frames


def run_cli(args: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)
    if parsed.max_files is not None and parsed.max_data_points is not None:
        raise ValueError("Use either --max-files or --max-data-points (not both).")

    overlay_helper = None
    if parsed.overlay_map:
        overlay_helper = OverlayHelper(parsed.overlay_map, verbose=not parsed.quiet)

    runs, data_frames = _prepare_runs(
        parsed.runs,
        target_samples=parsed.target_samples,
        max_files=parsed.max_files,
        max_data_points=parsed.max_data_points,
        verbose=not parsed.quiet,
        max_steps=parsed.max_steps,
    )

    for (log_dir, variant), df in zip(runs, data_frames):
        run_dir = log_dir.parent
        experiment_name = run_dir.name
        if df.empty:
            print(f"[heatmap_animation] No data sampled for {variant}; skipping.")
            continue

        map_column = _detect_map_column(df)
        counts = pd.to_numeric(df[map_column], errors="coerce").dropna().astype(int)
        counts = counts.value_counts().sort_values(ascending=False)
        if parsed.map_ids:
            map_ids: Sequence[int] = parsed.map_ids
        else:
            top_n = min(5, len(counts))
            map_ids = list(counts.head(top_n).index)

        if not parsed.quiet:
            print(f"[heatmap_animation] Variant {variant} -> animating maps {map_ids}")

        for map_id in map_ids:
            frames, step_ranges = _split_into_frames(
                df,
                map_id=map_id,
                frames=parsed.frames,
                normalize=parsed.normalize,
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
                fps=parsed.fps,
                fmt=parsed.format,
                output_subdir=parsed.output_dir,
            )


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
