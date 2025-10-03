"""
LSTM Environment (v3/v4)

This variant uses single frame with LSTM support:
- Action history of 10 steps
- Episode start flag for LSTM state reset
- Aggressive staggered resets with jitter
"""

import numpy as np
import io
import time
import random
from gymnasium import spaces

from environments.base_env import BaseRedGymEnv, EVENT_FLAGS_START, EVENT_FLAGS_END


class LSTMEnv(BaseRedGymEnv):
    """
    v3/v4: LSTM environment with action history and episode tracking.

    Used by both:
    - v3: RecurrentPPO
    - v4: RecurrentPPOLD
    """

    def __init__(self, config=None):
        # Set frame stack, history length and output shape BEFORE calling super().__init__()
        self.frame_stacks = 1
        self.history_len = 10  # Action history for LSTM
        self.output_shape = (72, 80, self.frame_stacks)
        super().__init__(config)

    def _build_observation_space(self):
        """Build observation space matching original v3/v4 (red_gym_env_lstm.py)."""
        return spaces.Dict({
            "screens": spaces.Box(
                low=0, high=255,
                shape=self.output_shape,
                dtype=np.uint8
            ),
            "health": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            ),
            "level": spaces.Box(
                low=-1, high=1,
                shape=(self.enc_freqs,),
                dtype=np.float32
            ),
            "badges": spaces.MultiBinary(8),
            "events": spaces.MultiBinary(
                (EVENT_FLAGS_END - EVENT_FLAGS_START) * 8
            ),
            "map": spaces.Box(
                low=0, high=255,
                shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                dtype=np.uint8
            ),
            "recent_actions": spaces.MultiDiscrete(
                [len(self.valid_actions)] * self.history_len
            ),
            # IMPORTANT: episode_start flag for LSTM state management
            "episode_start": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            )
        })

    def reset(self, seed=None, options={}):
        """Reset with LSTM-specific initialization."""
        self.seed = seed
        self.episode_start = True  # Signal LSTM to reset hidden state
        self.action_history = np.zeros((self.history_len,), dtype=np.int32)

        # Staggered reset with aggressive jitter (LSTM-specific)
        if self.reset_count == 0 and self.num_cpu > 1:
            offset_factor = self.worker_rank / max(1, self.num_cpu)
            episode_offset = int(self.base_max_steps * offset_factor * 0.5)  # Up to 50% offset
            self.max_steps = max(1000, self.base_max_steps - episode_offset)
            print(f"Worker {self.worker_rank}: First episode shortened to {self.max_steps} steps (offset: {episode_offset})")
        else:
            self.max_steps = self.base_max_steps

        # Load game state with aggressive jitter against thundering herd
        if self._init_state_bytes:
            base_delay = (self.worker_rank % 16) * 0.025  # 0-375ms staggered
            random_jitter = random.uniform(0, 0.1)  # +0-100ms random
            total_delay = base_delay + random_jitter
            print(f"Worker {self.worker_rank}: Delaying {total_delay:.3f}s before state load...")
            time.sleep(total_delay)

            print(f"Worker {self.worker_rank}: Loading state from memory...")
            self.pyboy.load_state(io.BytesIO(self._init_state_bytes))
            print(f"Worker {self.worker_rank}: State loaded successfully")

        # Initialize environment state
        self.init_map_mem()
        self.agent_stats = []
        self.explore_map_dim = (484, 476)  # GLOBAL_MAP_SHAPE
        from utils.map_utils import GLOBAL_MAP_SHAPE
        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        # Initialize frame stack (single frame for LSTM)
        self._init_frame_stack()

        # Render first frame and update frame stack
        first_frame = self.render()
        self._update_frame_stack(first_frame)

        # Game state tracking
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0

        # Event tracking
        self.base_event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        ])
        self.current_event_flags_set = {}

        # Reward tracking
        self.max_map_progress = 0
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])
        self.last_total_reward = self.total_reward

        self.reset_count += 1
        return self._get_obs(), {}

    def _init_frame_stack(self):
        """Initialize single frame buffer and action history."""
        self.current_screen = np.zeros(self.output_shape, dtype=np.uint8)
        self.action_history = np.zeros((self.history_len,), dtype=np.int32)

    def _update_frame_stack(self, screen):
        """Update current screen (no stacking for LSTM)."""
        # Screen is already (72, 80, 1), just keep it
        self.recent_screens = screen

    def _get_action_observation(self):
        """Return action history for LSTM."""
        return self.action_history

    def _update_action_history(self, action):
        """Update action history by rolling and adding new action."""
        self.action_history = np.roll(self.action_history, 1)
        self.action_history[0] = action

    def step(self, action):
        """Execute step and set episode_start to False after first step."""
        obs, reward, terminated, truncated, info = super().step(action)

        # After first step, episode_start becomes False
        if self.episode_start:
            self.episode_start = False

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Get current observation with episode_start flag (matching original v3/v4)."""
        # Calculate level sum (matching original)
        level_sum = 0.02 * sum([
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ])

        return {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()], dtype=np.float64),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.get_badges():08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "map": self.get_explore_map()[:, :, None],
            "recent_actions": self.action_history,
            "episode_start": np.array([float(self.episode_start)], dtype=np.float32)
        }
