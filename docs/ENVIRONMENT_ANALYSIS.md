# Environment Code Analysis

## Files Compared
- `red_gym_env_v2.py` (608 lines) - v1: Frame Stacking
- `red_gym_env_v2_adapted.py` (701 lines) - v2: Single Frame
- `red_gym_env_lstm.py` (695 lines) - v3/v4: LSTM

## Key Differences

### Frame Handling
| Feature | v1 (v2.py) | v2 (adapted) | v3/v4 (lstm) |
|---------|------------|--------------|--------------|
| `frame_stacks` | 3 | 1 | 1 |
| Screen obs | 3 stacked frames | 1 frame | 1 frame |
| Action history | `recent_actions` (3) | `recent_action` (1) | `recent_actions` (10) |
| Obs space key | `recent_actions` | `recent_action` | `recent_actions` |
| LSTM support | No | No | **Yes - episode_start flag** |

### Reset Logic
| Feature | v1 (v2.py) | v2 (adapted) | v3/v4 (lstm) |
|---------|------------|--------------|--------------|
| Staggered resets | No | **Yes** (worker_rank) | **Yes** (worker_rank + jitter) |
| Init state loading | File | **In-memory** | **In-memory** |
| Reset delays | No | Basic stagger | **Aggressive jitter** |
| Episode offset | No | Yes (0-10% offset) | Yes (0-50% offset) |

### Other Differences
| Feature | v1 (v2.py) | v2 (adapted) | v3/v4 (lstm) |
|---------|------------|--------------|--------------|
| Sound handling | Normal | **DummySound class** | Normal |
| `history_len` | - | - | **10** (for action history) |
| Episode tracking | - | - | **episode_start** flag |

## Common Code (Present in ALL 3 files)

### Initialization
- PyBoy setup
- Valid actions list (WindowEvent constants)
- Event names loading from JSON
- Observation space definition (screens, events, map, actions)
- Reward/episode tracking variables

### Core Methods (Identical or Nearly Identical)
1. `init_map_mem()` - Map memory initialization
2. `render()` - Screen rendering
3. `run_action_on_emulator()` - Action execution
4. `append_agent_stats()` - Stats tracking
5. `start_video()` / `add_video_frame()` - Video recording
6. `get_game_coords()` - Read player position
7. `update_seen_coords()` - Track exploration
8. `get_explore_map()` - Exploration visualization
9. `update_reward()` - Reward calculation
10. `group_rewards()` - Reward components
11. `check_if_done()` - Episode termination
12. `save_and_print_info()` - Logging
13. Memory reading methods: `read_m()`, `read_bit()`, `read_event_bits()`
14. Game state methods: `get_levels_sum()`, `get_badges()`, `read_party()`, etc.

### Estimated Code Duplication
- **Core game logic**: ~400-450 lines duplicated across all 3 files
- **Variant-specific**: ~150-250 lines per file

## Refactoring Strategy

### BaseRedGymEnv (Common Logic)
Extract to `src/environments/base_env.py`:
- PyBoy initialization
- Event/map data loading
- Memory reading methods
- Reward calculation
- Exploration tracking
- Stats/video recording
- Core game state reading

### Variant-Specific Environments

**FrameStackEnv (v1):**
- `frame_stacks = 3`
- Screen stacking logic
- `recent_actions` array (3 elements)
- No staggered resets
- File-based init state loading

**SingleFrameEnv (v2):**
- `frame_stacks = 1`
- Single frame observation
- `recent_action` single value
- Staggered resets
- In-memory init state
- Optional DummySound

**LSTMEnv (v3/v4):**
- `frame_stacks = 1`
- Action history (10 steps)
- **episode_start flag** in observation
- Staggered resets with aggressive jitter
- In-memory init state
- Episode tracking for LSTM state management

## Implementation Plan

1. Create `BaseRedGymEnv` with ~400 lines of common code
2. Create 3 subclasses with ~100-150 lines each
3. Total reduction: ~2000 lines → ~800 lines (60% reduction)
