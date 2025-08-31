#!/usr/bin/env python3
"""Test script to analyze map data"""

import sys
sys.path.append('src')

from poke_pipeline.global_map import MAP_DATA, GLOBAL_MAP_SHAPE

print(f'Global Map Shape: {GLOBAL_MAP_SHAPE}')
print(f'Total regions: {len(MAP_DATA)}')
print('\nKey regions:')
for k, v in list(MAP_DATA.items())[:15]:
    print(f'  {k}: {v["name"]} at {v["coordinates"]} size {v.get("tileSize", "N/A")}')

print(f'\nImportant cities:')
cities = [
    'Pallet Town', 'Viridian City', 'Pewter City', 'Cerulean City', 
    'Vermilion City', 'Lavender Town', 'Celadon City', 'Fuchsia City', 
    'Saffron City', 'Cinnabar island'
]

for region_id, data in MAP_DATA.items():
    if data['name'] in cities:
        print(f'  {data["name"]}: {data["coordinates"]} (ID: {region_id})')
