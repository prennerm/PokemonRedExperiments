"""
Single Frame Environment (v2)

This variant uses single frame (no stacking) and tracks only the last action.
Includes staggered reset logic for better parallelization.
"""

import numpy as np
from gymnasium import spaces

from environments.base_env import BaseRedGymEnv, EVENT_FLAGS_START, EVENT_FLAGS_END


class SingleFrameEnv(BaseRedGymEnv):
    """
    v2: Single frame environment with single action tracking.
    """

    def __init__(self, config=None):
        # Set frame stack size and output shape BEFORE calling super().__init__()
        self.send_map_to_agent = bool((config or {}).get("send_map_to_agent", True))
        self.frame_stacks = 1
        self.output_shape = (72, 80, self.frame_stacks)
        super().__init__(config)

    def _build_observation_space(self):
        """Build observation space matching original v2 (red_gym_env_v2_adapted.py)."""
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
            "recent_action": spaces.Discrete(len(self.valid_actions))
        }
        if self.send_map_to_agent:
            space_dict["map"] = spaces.Box(
                low=0,
                high=255,
                shape=(self.coords_pad * 4, self.coords_pad * 4, 1),
                dtype=np.uint8,
            )
        return spaces.Dict(space_dict)

    def _init_frame_stack(self):
        """Initialize single frame buffer and last action."""
        self.current_screen = np.zeros(self.output_shape, dtype=np.uint8)
        self.last_action = 0

    def _update_frame_stack(self, screen):
        """Update current screen (no stacking)."""
        # Screen is already (72, 80, 1), just keep it
        self.recent_screens = screen

    def _get_action_observation(self):
        """Return last action only."""
        return self.last_action

    def _update_action_history(self, action):
        """Update last action."""
        self.last_action = action

    def _get_obs(self):
        """Get current observation matching original v2."""
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
            "recent_action": self.last_action
        }
        if self.send_map_to_agent:
            obs["map"] = self.get_explore_map()[:, :, None]
        return obs
