"""
LSTM Environment (v3/v4)

This variant uses single frame with LSTM support:
- Action history of 10 steps
- Episode start flag for LSTM state reset
"""

import numpy as np
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
        self.send_map_to_agent = bool((config or {}).get("send_map_to_agent", True))
        self.frame_stacks = 1
        self.history_len = 10  # Action history for LSTM
        self.output_shape = (72, 80, self.frame_stacks)
        super().__init__(config)

    def _build_observation_space(self):
        """Build observation space matching original v3/v4 (red_gym_env_lstm.py)."""
        space_dict = {
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
            "recent_actions": spaces.MultiDiscrete(
                [len(self.valid_actions)] * self.history_len
            ),
            # IMPORTANT: episode_start flag for LSTM state management
            "episode_start": spaces.Box(
                low=0, high=1,
                shape=(1,),
                dtype=np.float32
            )
        }
        if self.send_map_to_agent:
            space_dict["map"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                dtype=np.uint8,
            )
        return spaces.Dict(space_dict)

    def reset(self, seed=None, options=None):
        """Reset environment without staggered timing hacks."""
        self.episode_start = True
        self.max_steps = self.base_max_steps
        options = options or {}
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

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

        obs = {
            "screens": self.recent_screens,
            "health": np.array([self.read_hp_fraction()], dtype=np.float64),
            "level": self.fourier_encode(level_sum),
            "badges": np.array([int(bit) for bit in f"{self.get_badges():08b}"], dtype=np.int8),
            "events": np.array(self.read_event_bits(), dtype=np.int8),
            "recent_actions": self.action_history,
            "episode_start": np.array([float(self.episode_start)], dtype=np.float32)
        }
        if self.send_map_to_agent:
            obs["map"] = self.get_explore_map()[:, :, None]
        return obs
