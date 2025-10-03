# Data Directory

This directory contains game data files required for training.

## Required Files (Not in Git)

Download these files separately due to copyright/size restrictions:

### Game ROM
- **File**: `PokemonRed.gb` (1MB)
- **Source**: Legally obtain Pokemon Red ROM
- **Purpose**: GameBoy emulator (PyBoy) requires the ROM file

### Save States
- **File**: `init.state` (143KB) - ✅ **Included in repo**
- **Purpose**: Initial game state for training (right after Oak intro)

### Map Data
- **File**: `map_data.json` - **To be created** (currently in `src/environments/data/`)
- **File**: `events.json` - **To be created** (currently in `src/environments/data/`)

## Setup

```bash
# 1. Place PokemonRed.gb in this directory
cp /path/to/PokemonRed.gb data/

# 2. Verify files
ls -lh data/
# Should show:
#   - PokemonRed.gb (1.0M)
#   - init.state (143K)
#   - README.md (this file)
```

## File Sizes

| File | Size | In Git? | Purpose |
|------|------|---------|---------|
| `PokemonRed.gb` | 1.0 MB | ❌ No (copyright) | Game ROM for emulator |
| `init.state` | 143 KB | ✅ Yes | Starting game state |
| `events.json` | ~50 KB | ⚠️ TBD | Event flag definitions |
| `map_data.json` | ~30 KB | ⚠️ TBD | Map coordinate data |

## Important Notes

⚠️ **DO NOT commit `PokemonRed.gb` to GitHub** - it's copyrighted content!

The `.gitignore` is configured to exclude:
- `*.gb` files
- Large training outputs
- Generated files

Only small, essential reference files are tracked in git.
