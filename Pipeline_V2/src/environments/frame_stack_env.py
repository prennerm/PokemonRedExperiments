"""
Frame Stacking Environment (v1)

This variant uses 3-frame stacking and tracks the last 3 actions.
Original baseline configuration from Peter Whidden's work.
"""

import numpy as np
from gymnasium import spaces

from environments.base_env import BaseRedGymEnv, EVENT_FLAGS_START, EVENT_FLAGS_END


class FrameStackEnv(BaseRedGymEnv):
    """
    v1: Frame stacking environment with 3 frames and 3-action history.
    """

    def __init__(self, config=None):
        # Set frame stack size and output shape BEFORE calling super().__init__()
        # This is critical because _build_observation_space() needs output_shape
        self.frame_stacks = 3
        self.output_shape = (72, 80, self.frame_stacks)
        super().__init__(config)

    def _build_observation_space(self):
        """Build observation space matching original v1 (red_gym_env_v2.py)."""
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
                [len(self.valid_actions)] * self.frame_stacks
            )
        })

    def _init_frame_stack(self):
        """Initialize 3-frame screen stack and 3-action history."""
        self.recent_screens = np.zeros(self.output_shape, dtype=np.uint8)
        self.recent_actions = np.zeros((self.frame_stacks,), dtype=np.uint8)

    def _update_frame_stack(self, screen):
        """Update frame stack by shifting and adding new frame."""
        # Roll screens: move frame 1->2, 2->3, add new at 0
        self.recent_screens = np.roll(self.recent_screens, 1, axis=2)
        self.recent_screens[:, :, 0] = screen[:, :, 0]

    def _get_action_observation(self):
        """Return recent 3 actions."""
        return self.recent_actions

    def _update_action_history(self, action):
        """Update action history by rolling and adding new action."""
        self.recent_actions = np.roll(self.recent_actions, 1)
        self.recent_actions[0] = action

    def _get_obs(self):
        """Get current observation matching original v1."""
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
            "recent_actions": self.recent_actions
        }
