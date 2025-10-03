"""
Base Environment for Pokemon Red RL Training

This module contains the common logic shared across all environment variants.
Subclasses implement variant-specific observation spaces and frame handling.

Originally adapted from Peter Whidden's work and pokemonred_puffer.
"""

import uuid
import json
import io
from pathlib import Path
from abc import ABC, abstractmethod
import pkg_resources

import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent

from utils.map_utils import local_to_global, GLOBAL_MAP_SHAPE

# Event tracking addresses
EVENT_FLAGS_START = 0xD747
EVENT_FLAGS_END = 0xD87E  # expanded for SS Anne
MUSEUM_TICKET = (0xD754, 0)


class BaseRedGymEnv(Env, ABC):
    """
    Base environment with common game logic.

    Subclasses must implement:
    - _build_observation_space() -> spaces.Dict
    - _get_obs() -> dict
    - _init_frame_stack() -> None
    - _update_frame_stack(screen) -> None
    - _get_action_observation() -> Any
    - _update_action_history(action) -> None
    """

    def __init__(self, config=None):
        """Initialize common environment components."""
        # Basic config
        self.s_path = Path(config["session_path"])
        self.save_final_state = config["save_final_state"]
        self.print_rewards = config["print_rewards"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.base_max_steps = config["max_steps"]
        self.max_steps = self.base_max_steps
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]

        # Worker info for staggered resets
        self.worker_rank = config.get("worker_rank", 0)
        self.num_cpu = config.get("num_cpu", 1)

        print(f"[EnvInit] Worker {self.worker_rank}/{self.num_cpu} initialized")

        # Load init state into memory for better performance
        self._init_state_bytes = None
        if self.init_state and self.init_state.strip():
            try:
                with open(self.init_state, "rb") as f:
                    self._init_state_bytes = f.read()
                print(f"Worker {self.worker_rank}: Loaded init state into memory ({len(self._init_state_bytes)} bytes)")
            except Exception as e:
                print(f"Worker {self.worker_rank}: Could not load init state: {e}")

        # Reward scaling
        self.explore_weight = config.get("explore_weight", 1.0)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])

        # Create session directory
        self.s_path.mkdir(exist_ok=True)

        # Video writers
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.map_frame_writer = None
        self.reset_count = 0

        # Valid actions (only PRESS actions, matching original)
        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        # Release actions (used in step() but not in action space)
        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        # Load event names
        data_dir = Path(pkg_resources.resource_filename(__name__, "data"))
        with open(data_dir / "events.json") as f:
            self.event_names = json.load(f)

        # Subclass must define output_shape BEFORE calling super().__init__()
        # If not set by subclass, use a default (should not happen)
        if not hasattr(self, 'output_shape'):
            self.output_shape = None
        self.coords_pad = 12

        # Action space is always the same
        self.action_space = spaces.Discrete(len(self.valid_actions))

        # Frequency encoding for positional information
        self.enc_freqs = 8

        # Observation space defined by subclass
        self.observation_space = self._build_observation_space()

        # Initialize PyBoy
        head = "headless" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
            sound_emulated=False,
        )

        # Set emulation speed if not headless
        if not self.headless:
            self.pyboy.set_emulation_speed(6)

    @abstractmethod
    def _build_observation_space(self) -> spaces.Dict:
        """Build the observation space for this environment variant."""
        pass

    @abstractmethod
    def _get_obs(self) -> dict:
        """Get current observation."""
        pass

    @abstractmethod
    def _init_frame_stack(self):
        """Initialize frame stacking/history for this variant."""
        pass

    @abstractmethod
    def _update_frame_stack(self, screen: np.ndarray):
        """Update frame stack with new screen."""
        pass

    @abstractmethod
    def _get_action_observation(self):
        """Get action history observation for current variant."""
        pass

    @abstractmethod
    def _update_action_history(self, action: int):
        """Update action history."""
        pass

    def reset(self, seed=None, options={}):
        """Reset environment. Can be overridden for variant-specific reset logic."""
        self.seed = seed

        # Staggered resets: first episode of each worker is shortened
        if self.reset_count == 0 and self.num_cpu > 1:
            offset_factor = self.worker_rank / max(1, self.num_cpu)
            episode_offset = int(self.base_max_steps * 0.1 * offset_factor)
            self.max_steps = max(1000, self.base_max_steps - episode_offset)
            print(f"Worker {self.worker_rank}: First episode shortened to {self.max_steps} steps (offset: {episode_offset})")
        else:
            self.max_steps = self.base_max_steps

        # Load game state
        if self._init_state_bytes:
            import time
            base_delay = (self.worker_rank % 16) * 0.025
            extra_delay = (self.reset_count % 4) * 0.01
            total_delay = base_delay + extra_delay
            print(f"Worker {self.worker_rank}: Delaying {total_delay:.3f}s before state load...")
            time.sleep(total_delay)
            print(f"Worker {self.worker_rank}: Loading state from memory...")
            self.pyboy.load_state(io.BytesIO(self._init_state_bytes))
            print(f"Worker {self.worker_rank}: State loaded successfully")
        elif self.init_state and self.init_state.strip():
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)

        # Initialize environment state
        self.init_map_mem()
        self.agent_stats = []
        self.explore_map_dim = GLOBAL_MAP_SHAPE
        self.explore_map = np.zeros(self.explore_map_dim, dtype=np.uint8)

        # Initialize frame stack (variant-specific)
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

    def init_map_mem(self):
        """Initialize map memory."""
        self.seen_coords = {}

    def render(self, reduce_res=True):
        """Render the current game screen."""
        game_pixels_render = self.pyboy.screen.ndarray[:, :, 0:1]
        if reduce_res:
            game_pixels_render = downscale_local_mean(game_pixels_render, (2, 2, 1)).astype(np.uint8)
        return game_pixels_render

    def step(self, action):
        """Execute one step in the environment."""
        self.run_action_on_emulator(action)
        self.append_agent_stats(action)

        # Update action history
        self._update_action_history(action)

        # Get screen and update frame stack
        self.update_recent_screens(self.render())

        # Update rewards
        self.update_reward()
        self.step_count += 1

        # Check if episode is done
        step_limit_reached = self.step_count >= self.max_steps
        done = step_limit_reached or self.check_if_done()

        # Get observation and info
        obs = self._get_obs()
        info = {}

        # Save stats if done
        if done:
            self.save_and_print_info(done, obs)

        # Calculate reward
        reward = self.total_reward - self.last_total_reward
        self.last_total_reward = self.total_reward

        # Add video frame if recording
        if self.save_video and self.step_count % 50 == 0:
            self.add_video_frame()

        return obs, reward * 0.1 * self.reward_scale, False, done, info

    def run_action_on_emulator(self, action):
        """Execute action on emulator."""
        # Press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        # Release uses same index as press
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)

    def append_agent_stats(self, action):
        """Append current stats to agent_stats list."""
        x, y, map_n = self.get_game_coords()

        stat = {
            "step": self.step_count,
            "total_steps": self.step_count + self.reset_count * self.base_max_steps,
            "position": {"x": int(x), "y": int(y), "map": int(map_n)},
            "rewards": {
                "total": float(self.total_reward),
                "step": float(self.total_reward - self.last_total_reward),
                "components": {k: float(v) for k, v in self.group_rewards().items()}
            },
            "player_status": {
                "health": float(self.last_health),
                "levels": [int(l) for l in self.read_party()],
                "levels_sum": int(self.get_levels_sum()),
                "badges": int(self.get_badges()),
                "pokemon_count": int(self.party_size),
                "pokemon_types": [int(t) for t in self.read_party_types()]
            },
            "actions": {"last_action": int(action)},
            "statistics": {
                "deaths": int(self.died_count),
                "exploration_coords": int(len(self.seen_coords)),
                "map_progress": int(self.max_map_progress),
                "healing_reward": float(self.total_healing_rew),
                "event_progress": float(self.max_event_rew)
            }
        }

        self.agent_stats.append(stat)

    def update_recent_screens(self, cur_screen):
        """Update screen buffer - implemented by subclass."""
        self._update_frame_stack(cur_screen)

    def update_reward(self):
        """Update reward based on game state."""
        self.progress_reward = self.get_game_state_reward()
        self.total_reward = sum([val for _, val in self.progress_reward.items()])

    def group_rewards(self):
        """Group rewards by category."""
        prog = self.progress_reward
        return {
            "event": prog["event"],
            "level": prog["level"],
            "heal": prog["heal"],
            "badge": prog.get("badge", 0),
            "explore": prog["explore"],
            "dead": prog.get("died", 0),
            "stuck": prog.get("stuck", 0)
        }

    def check_if_done(self):
        """Check if episode should end."""
        return False  # Override in subclass if needed

    def save_and_print_info(self, done, obs):
        """Save episode statistics."""
        if self.print_rewards:
            prog_string = f"step: {self.step_count:6d}"
            for key, val in self.progress_reward.items():
                prog_string += f" | {key}: {val:5.2f}"
            prog_string += f" | sum: {self.total_reward:5.2f}"
            print(f"\\r{prog_string}", end="", flush=True)

        if self.save_final_state:
            state_path = self.s_path / f"final_state_{self.reset_count}.state"
            with open(state_path, "wb") as f:
                self.pyboy.save_state(f)

        if self.save_video:
            self.full_frame_writer.close()
            self.model_frame_writer.close()
            self.map_frame_writer.close()

    # ========== GAME STATE READING METHODS ==========

    def read_m(self, addr):
        """Read byte from memory."""
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        """Read specific bit from memory address."""
        return bool(self.read_m(addr) & (1 << bit))

    def bit_count(self, bits):
        """Count number of set bits."""
        return bin(bits).count("1")

    def read_event_bits(self):
        """Read all event bits."""
        return [
            self.read_bit(i, j)
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
            for j in range(8)
        ]

    def get_levels_sum(self):
        """Get sum of all party Pokemon levels."""
        min_level = 2
        max_level = 100
        level_sum = 0
        party = self.read_party()
        for level in party:
            if level >= min_level and level <= max_level:
                level_sum += level
        return level_sum

    def get_levels_reward(self):
        """Calculate reward from Pokemon levels."""
        level_sum = self.get_levels_sum()
        if level_sum < 15:
            return level_sum / 30
        else:
            self.levels_satisfied = True
            return 0.5 + (level_sum - 15) / 240

    def get_badges(self):
        """Read number of badges."""
        return self.bit_count(self.read_m(0xD356))

    def read_party(self):
        """Read party Pokemon levels."""
        party_size = self.read_m(0xD163)
        party_levels = [self.read_m(addr) for addr in range(0xD18C, 0xD18C + party_size)]
        self.party_size = party_size
        return party_levels

    def read_party_types(self):
        """Read party Pokemon types."""
        party_size = self.read_m(0xD163)
        party_types = [self.read_m(addr) for addr in range(0xD170, 0xD170 + party_size)]
        return party_types

    def get_all_events_reward(self):
        """Calculate reward from events."""
        event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        ])
        return (event_flags - self.base_event_flags) * 0.1

    def get_game_coords(self):
        """Get current player coordinates."""
        x = self.read_m(0xD362)
        y = self.read_m(0xD361)
        map_n = self.read_m(0xD35E)
        return x, y, map_n

    def get_global_coords(self):
        """Convert local coords to global map coords."""
        x, y, map_n = self.get_game_coords()
        return local_to_global(y, x, map_n)

    def update_seen_coords(self):
        """Track exploration."""
        x, y, map_n = self.get_game_coords()
        coord_string = f"x:{x} y:{y} map:{map_n}"
        self.seen_coords[coord_string] = self.step_count

    def get_current_coord_count_reward(self):
        """Get reward for exploration."""
        self.update_seen_coords()
        return len(self.seen_coords) * 0.005

    def update_explore_map(self):
        """Update exploration map."""
        gy, gx = self.get_global_coords()
        self.explore_map[gy, gx] = 1
        map_coverage = self.explore_map.sum()
        self.max_map_progress = max(self.max_map_progress, map_coverage)

    def get_explore_map(self):
        """Get exploration map visualization (local crop, upscaled 2x)."""
        c = self.get_global_coords()
        # Get local crop around current position
        if c[0] >= self.explore_map.shape[0] or c[1] >= self.explore_map.shape[1]:
            out = np.zeros((self.coords_pad * 2, self.coords_pad * 2), dtype=np.uint8)
        else:
            out = self.explore_map[
                c[0] - self.coords_pad:c[0] + self.coords_pad,
                c[1] - self.coords_pad:c[1] + self.coords_pad
            ]
        # Upscale 2x to match original (24x24 -> 48x48)
        return repeat(out, 'h w -> (h h2) (w w2)', h2=2, w2=2)

    def get_game_state_reward(self):
        """Calculate total reward from game state."""
        self.update_explore_map()

        reward = {
            "event": self.get_all_events_reward(),
            "level": self.get_levels_reward(),
            "heal": self.get_healing_reward(),
            "badge": self.get_badge_reward(),
            "explore": self.explore_weight * self.get_current_coord_count_reward(),
            "died": self.get_died_reward(),
            "stuck": self.get_stuck_reward()
        }

        self.max_event_rew = max(self.max_event_rew, reward["event"])
        self.max_level_rew = max(self.max_level_rew, reward["level"])

        return reward

    def get_healing_reward(self):
        """Calculate healing reward."""
        cur_health = self.read_m(0xD16C) / 255.0
        if cur_health > self.last_health:
            heal_amount = cur_health - self.last_health
            self.total_healing_rew += heal_amount
        self.last_health = cur_health
        return self.total_healing_rew * 0.05

    def get_badge_reward(self):
        """Calculate badge reward."""
        return self.get_badges() * 5.0

    def get_died_reward(self):
        """Penalty for dying."""
        if self.read_m(0xD057) == 0:
            self.died_count += 1
        return -1.0 * self.died_count

    def get_stuck_reward(self):
        """Penalty for getting stuck."""
        return 0.0  # Can be implemented if needed

    # ========== VIDEO RECORDING ==========

    def start_video(self):
        """Start video recording."""
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        if self.map_frame_writer is not None:
            self.map_frame_writer.close()

        base_dir = self.s_path / "video"
        base_dir.mkdir(exist_ok=True)

        self.full_frame_writer = media.VideoWriter(
            base_dir / f"full_frame_{self.reset_count}.mp4",
            (144, 160), fps=60, input_format="gray"
        )
        self.model_frame_writer = media.VideoWriter(
            base_dir / f"model_frame_{self.reset_count}.mp4",
            self.output_shape[:2], fps=60, input_format="gray"
        )
        self.map_frame_writer = media.VideoWriter(
            base_dir / f"map_{self.reset_count}.mp4",
            (self.coords_pad * 4, self.coords_pad * 4), fps=60, input_format="gray"
        )

    def add_video_frame(self):
        """Add frame to video."""
        if self.full_frame_writer is not None:
            self.full_frame_writer.add_image(self.pyboy.screen.ndarray[:, :, 0])
            self.model_frame_writer.add_image(self.render()[:, :, 0])
            self.map_frame_writer.add_image(self.get_explore_map()[:, :, None][:, :, 0] * 128)

    def read_hp(self, start):
        """Read HP value from 2-byte address."""
        return 256 * self.read_m(start) + self.read_m(start + 1)

    def read_hp_fraction(self):
        """Calculate current HP fraction across party."""
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def fourier_encode(self, val):
        """Fourier encoding for positional information."""
        return np.sin(val * 2 ** np.arange(self.enc_freqs))

    def get_badges(self):
        """Get badge byte (0xD356)."""
        return self.read_m(0xD356)
