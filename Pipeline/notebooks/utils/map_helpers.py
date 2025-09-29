"""
Map visualization helpers for Pokemon Red heatmap overlays
Extracted and adapted from visualization/BetterMapVis_script_version.py
"""

import numpy as np
from pathlib import Path

def game_coord_to_pixel_coord(x, y, map_idx, base_y=None):
    """
    Convert Pokemon Red game coordinates to pixel coordinates on the global map

    Args:
        x, y: Game coordinates within the map
        map_idx: Map ID from the game
        base_y: Base Y coordinate for flipping (optional)

    Returns:
        np.array: [pixel_x, pixel_y] coordinates
    """

    global_offset = np.array([1056-16*12, 331])
    map_offsets = {
        # https://bulbapedia.bulbagarden.net/wiki/List_of_locations_by_index_number_(Generation_I)
        0: np.array([0,0]),      # pallet town
        1: np.array([-10, 72]),  # viridian
        2: np.array([-10, 180]), # pewter
        12: np.array([0, 36]),   # route 1
        13: np.array([0, 144]),  # route 2
        14: np.array([30, 172]), # Route 3
        15: np.array([80, 190]), # Route 4
        33: np.array([-50, 64]), # route 22
        37: np.array([-9, 2]),   # red house first
        38: np.array([-9, 25-32]), # red house second
        39: np.array([9+12, 2]), # blues house
        40: np.array([25-4, -6]), # oaks lab
        41: np.array([30, 47]),  # Pokémon Center (Viridian City)
        42: np.array([30, 55]),  # Poké Mart (Viridian City)
        43: np.array([30, 72]),  # School (Viridian City)
        44: np.array([30, 64]),  # House 1 (Viridian City)
        47: np.array([21,136]),  # Gate (Viridian City/Pewter City) (Route 2)
        49: np.array([21,108]),  # Gate (Route 2)
        50: np.array([21,108]),  # Gate (Route 2/Viridian Forest) (Route 2)
        51: np.array([-35, 137]), # viridian forest
        52: np.array([-10, 189]), # Pewter Museum (floor 1)
        53: np.array([-10, 198]), # Pewter Museum (floor 2)
        54: np.array([-21, 169]), # Pokémon Gym (Pewter City)
        55: np.array([-19, 177]), # House with disobedient Nidoran♂ (Pewter City)
        56: np.array([-30, 163]), # Poké Mart (Pewter City)
        57: np.array([-19, 177]), # House with two Trainers (Pewter City)
        58: np.array([-25, 154]), # Pokémon Center (Pewter City)
        59: np.array([83, 227]),  # Mt. Moon (Route 3 entrance)
        60: np.array([123, 227]), # Mt. Moon
        61: np.array([152, 227]), # Mt. Moon
        68: np.array([65, 190]),  # Pokémon Center (Route 4)
        193: None # Badges check gate (Route 22)
    }

    if map_idx in map_offsets.keys() and map_offsets[map_idx] is not None:
        offset = map_offsets[map_idx]
    else:
        # Unknown map - place at origin
        offset = np.array([0, 0])
        x, y = 0, 0

    coord = global_offset + 16 * (offset + np.array([x, y]))

    if base_y is not None:
        coord[1] = base_y - coord[1]

    return coord

def get_map_name(map_idx):
    """Get human-readable map name from map index"""
    map_names = {
        0: "Pallet Town",
        1: "Viridian City",
        2: "Pewter City",
        12: "Route 1",
        13: "Route 2",
        14: "Route 3",
        15: "Route 4",
        33: "Route 22",
        37: "Red's House (1F)",
        38: "Red's House (2F)",
        39: "Blue's House",
        40: "Oak's Lab",
        41: "Pokémon Center (Viridian)",
        42: "Poké Mart (Viridian)",
        43: "School (Viridian)",
        44: "House (Viridian)",
        47: "Gate (Route 2)",
        49: "Gate (Route 2)",
        50: "Gate (Route 2/Viridian Forest)",
        51: "Viridian Forest",
        52: "Pewter Museum (1F)",
        53: "Pewter Museum (2F)",
        54: "Pokémon Gym (Pewter)",
        55: "House (Pewter)",
        56: "Poké Mart (Pewter)",
        57: "House (Pewter)",
        58: "Pokémon Center (Pewter)",
        59: "Mt. Moon (Entrance)",
        60: "Mt. Moon",
        61: "Mt. Moon",
        68: "Pokémon Center (Route 4)",
        193: "Badges Gate (Route 22)"
    }
    return map_names.get(map_idx, f"Unknown Map {map_idx}")

def load_pokemon_map_background(utils_dir=None):
    """Load the Pokemon Red map background image"""
    if utils_dir is None:
        utils_dir = Path(__file__).parent

    map_path = utils_dir / "pokemap_full_calibrated_CROPPED_1.png"

    if not map_path.exists():
        raise FileNotFoundError(f"Pokemon map background not found at {map_path}")

    from PIL import Image
    return np.array(Image.open(map_path))