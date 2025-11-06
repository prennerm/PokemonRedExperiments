# adapted from Pipeline_V2/utils/map_utils.py with graceful fallbacks

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

# Resolve default map data location relative to the package
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAP_PATH = PACKAGE_ROOT / "data" / "map_data.json"

# Attempt fallback for local execution where repo root is the current working directory
if not DEFAULT_MAP_PATH.is_file():
    alt_path = Path.cwd() / "src" / "poke_pipeline" / "data" / "map_data.json"
    if alt_path.is_file():
        DEFAULT_MAP_PATH = alt_path


def _load_map_data(path: Path) -> Dict[int, dict]:
    if not path.is_file():
        print(f"[map_utils] Warning: map_data.json not found at {path}. Falling back to empty map metadata.")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        regions = data.get("regions", [])
        return {int(entry["id"]): entry for entry in regions if "id" in entry}
    except Exception as exc:
        print(f"[map_utils] Warning: failed to load map data from {path}: {exc}")
        return {}


MAP_DATA: Dict[int, dict] = _load_map_data(DEFAULT_MAP_PATH)


def local_to_global(r: int, c: int, map_n: int) -> Tuple[int, int]:
    """Convert local (row/col) coordinates to the global heatmap grid."""
    entry = MAP_DATA.get(map_n)
    if not entry:
        print(f"[map_utils] Map id {map_n} not found; using centre fallback.")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2

    map_x, map_y = entry.get("coordinates", (0, 0))
    gy = r + map_y + MAP_ROW_OFFSET
    gx = c + map_x + MAP_COL_OFFSET

    if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
        return gy, gx

    print(f"[map_utils] coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
    return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2


__all__ = ["MAP_DATA", "GLOBAL_MAP_SHAPE", "local_to_global"]
