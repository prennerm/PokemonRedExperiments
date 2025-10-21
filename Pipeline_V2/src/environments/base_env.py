"""
Base Environment for Pokemon Red RL Training

This module contains the common logic shared across all environment variants.
Subclasses implement variant-specific observation spaces and frame handling.

Originally adapted from Peter Whidden's work and pokemonred_puffer.
"""

import uuid
import json
import time
from pathlib import Path
from abc import ABC, abstractmethod
import logging
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


class _SuppressOldStateWarning(logging.Filter):
    MESSAGE = "Loading state from an older version of PyBoy."

    def filter(self, record: logging.LogRecord) -> bool:
        return self.MESSAGE not in record.getMessage()


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
        self._init_state_bytes = None
        self.worker_rank = config.get("worker_rank", 0)
        self.num_cpu = config.get("num_cpu", 1)
        self.send_map_to_agent = bool(config.get("send_map_to_agent", True))
        self.debug_reset_timing = config.get("debug_reset_timing", False)
        self._episode_first_step_logged = True

        # Reward scaling
        self.explore_weight = config.get("explore_weight", 1.0)
        self.reward_scale = config.get("reward_scale", 1.0)
        self.instance_id = config.get("instance_id", str(uuid.uuid4())[:8])

        # Create session directory
        self.s_path.mkdir(exist_ok=True)

        if self.debug_reset_timing:
            self.reset_log_path = self.s_path / f"reset_timing_worker_{self.worker_rank}.log"
            # Overwrite previous logs for clean runs
            self.reset_log_path.write_text("", encoding="utf-8")

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
        events_path = data_dir / "events.json"
        if events_path.exists():
            with open(events_path) as f:
                self.event_names = json.load(f)
        else:
            print(f"[BaseRedGymEnv] Warning: events.json not found at {events_path}. Event tracking disabled.")
            self.event_names = {}

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
        head = "null" if self.headless else "SDL2"
        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
            sound_emulated=False,
        )

        # Suppress legacy state-load warnings in spawned workers
        for logger_name in ("pyboy.core.mb", "pyboy.core.sound"):
            pyboy_logger = logging.getLogger(logger_name)
            pyboy_logger.setLevel(logging.ERROR)
            pyboy_logger.propagate = False
            if not pyboy_logger.handlers:
                pyboy_logger.addHandler(logging.NullHandler())
            pyboy_logger.addFilter(_SuppressOldStateWarning())
        logging.getLogger().addFilter(_SuppressOldStateWarning())

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

        # Load game state (directly from file, matching original stable pattern)
        start_ts = time.perf_counter() if self.debug_reset_timing else None
        if self.debug_reset_timing:
            self._log_reset_timing(f"start reset={self.reset_count} seed={seed}")

        try:
            with open(self.init_state, "rb") as f:
                self.pyboy.load_state(f)
            if self.debug_reset_timing and start_ts is not None:
                duration = time.perf_counter() - start_ts
                self._log_reset_timing(
                    f"done reset={self.reset_count} duration={duration:.6f}"
                )
        except Exception as exc:
            if self.debug_reset_timing and start_ts is not None:
                self._log_reset_timing(
                    f"error reset={self.reset_count} exception={type(exc).__name__}: {exc}"
                )
            raise

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
        self._episode_first_step_logged = False
        return self._get_obs(), {}

    def _log_reset_timing(self, message: str):
        """Write reset timing diagnostics to per-worker log."""
        if not self.debug_reset_timing:
            return
        timestamp = time.time()
        line = f"{timestamp:.6f} rank={self.worker_rank} {message}\n"
        with self.reset_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(line)

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

        if self.debug_reset_timing and not self._episode_first_step_logged:
            self._log_reset_timing(
                f"step_first reset={self.reset_count} step_count={self.step_count}"
            )
            self._episode_first_step_logged = True

        # Get screen and update frame stack
        self.update_recent_screens(self.render())

        # Update seen coords and exploration
        self.update_seen_coords()
        self.update_explore_map()

        # Update heal reward and death tracking
        self.update_heal_reward()
        self.party_size = self.read_m(0xD163)

        # Update rewards
        self.update_reward()

        # Update last_health AFTER reward calculation
        self.last_health = self.read_hp_fraction()

        self.step_count += 1

        # Check if episode is done
        step_limit_reached = self.step_count >= self.max_steps
        done = step_limit_reached or self.check_if_done()

        if self.debug_reset_timing and done:
            self._log_reset_timing(
                f"step_done reset={self.reset_count} step_count={self.step_count}"
            )
        elif self.debug_reset_timing and self.step_count % 512 == 0:
            self._log_reset_timing(
                f"step_progress reset={self.reset_count} step_count={self.step_count}"
            )

        if self.debug_reset_timing and self.step_count <= 4:
            self._log_reset_timing(
                f"step_return reset={self.reset_count} step_count={self.step_count} done={int(done)}"
            )

        # Get observation and info
        obs = self._get_obs()
        info = {
            "debug_step_count": self.step_count,
            "debug_reset": self.reset_count,
            "debug_worker_rank": self.worker_rank,
        }

        # Save stats if done
        if done:
            self.save_and_print_info(done, obs)

        # Calculate reward
        reward = self.total_reward - self.last_total_reward
        self.last_total_reward = self.total_reward

        # Add video frame if recording
        if self.save_video and self.step_count % 50 == 0:
            self.add_video_frame()

        # BUG FIX #3: Removed "* 0.1 * self.reward_scale" - reward_scale already applied in get_game_state_reward()
        return obs, reward, False, done, info

    def run_action_on_emulator(self, action):
        """Execute action on emulator."""
        log_action = self.debug_reset_timing and self.step_count < 4
        if log_action:
            self._log_reset_timing(
                f"action_start reset={self.reset_count} step_count={self.step_count} action={action}"
            )
        # Press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        render_screen = self.save_video or not self.headless
        press_step = 8
        self.pyboy.tick(press_step, render_screen)
        # Release uses same index as press
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if log_action:
            self._log_reset_timing(
                f"action_end reset={self.reset_count} step_count={self.step_count}"
            )

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
            "badge": prog["badge"],
            "explore": prog["explore"],
            "dead": prog["dead"],
            "stuck": prog["stuck"]
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
        """Get sum of all party Pokemon levels (adjusted)."""
        min_poke_level = 2
        starter_additional_levels = 4
        poke_levels = [
            max(self.read_m(a) - min_poke_level, 0)
            for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        return max(sum(poke_levels) - starter_additional_levels, 0)

    def get_levels_reward(self):
        """Calculate reward from Pokemon levels."""
        explore_thresh = 22
        scale_factor = 4
        level_sum = self.get_levels_sum()
        if level_sum < explore_thresh:
            scaled = level_sum
        else:
            scaled = (level_sum - explore_thresh) / scale_factor + explore_thresh
        self.max_level_rew = max(self.max_level_rew, scaled)
        return self.max_level_rew

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
        """Read party Pokemon species/types."""
        party_size = self.read_m(0xD163)
        # BUG FIX #1: Changed from 0xD170 (OT Names) to 0xD164 (Pokemon Species/Types)
        party_types = [self.read_m(addr) for addr in range(0xD164, 0xD164 + party_size)]
        return party_types

    def get_all_events_reward(self):
        """Calculate reward from events (excludes museum ticket)."""
        event_flags = sum([
            self.bit_count(self.read_m(i))
            for i in range(EVENT_FLAGS_START, EVENT_FLAGS_END)
        ])
        # Exclude museum ticket from event count
        return max(
            event_flags
            - self.base_event_flags
            - int(self.read_bit(MUSEUM_TICKET[0], MUSEUM_TICKET[1])),
            0
        )

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
        """Track exploration - only when not in battle."""
        # Only track when not in battle (0xD057 == 0)
        if self.read_m(0xD057) == 0:
            x, y, map_n = self.get_game_coords()
            coord_string = f"x:{x} y:{y} m:{map_n}"
            if coord_string in self.seen_coords:
                self.seen_coords[coord_string] += 1
            else:
                self.seen_coords[coord_string] = 1

    def get_current_coord_count_reward(self):
        """Get penalty for being stuck at same coordinate."""
        x, y, map_n = self.get_game_coords()
        coord_string = f"x:{x} y:{y} m:{map_n}"
        count = self.seen_coords.get(coord_string, 0)
        return 0 if count < 300 else 1

    def update_explore_map(self):
        """Update exploration map."""
        gy, gx = self.get_global_coords()
        # BUG FIX #4: Added bounds checking to prevent IndexError
        if gy >= self.explore_map.shape[0] or gx >= self.explore_map.shape[1]:
            return  # Skip update if coordinates out of bounds
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
        # Match original formula exactly (line 625-634 in red_gym_env_lstm.py)
        reward = {
            "event": self.reward_scale * self.update_max_event_rew() * 4,
            "level": self.reward_scale * self.get_levels_reward(),
            "heal": self.reward_scale * self.total_healing_rew * 30,
            "dead": self.reward_scale * self.died_count * -0.1,
            "badge": self.reward_scale * self.get_badges() * 10,
            "explore": self.reward_scale * self.explore_weight * len(self.seen_coords) * 0.1,
            "stuck": self.reward_scale * self.get_current_coord_count_reward() * -0.05
        }

        return reward

    def update_heal_reward(self):
        """Update healing reward and detect deaths."""
        cur_health = self.read_hp_fraction()
        # if health increased and party size did not change
        if cur_health > self.last_health and self.read_m(0xD163) == self.party_size:
            if self.last_health > 0:
                heal_amount = cur_health - self.last_health
                self.total_healing_rew += heal_amount * heal_amount
            else:
                # Agent respawned from death (was at 0 HP, now has HP)
                self.died_count += 1

    def update_max_event_rew(self):
        """Update and return maximum event reward."""
        cur_rew = self.get_all_events_reward()
        self.max_event_rew = max(cur_rew, self.max_event_rew)
        return self.max_event_rew

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

    # BUG FIX #2: Removed duplicate get_badges() definition that returned byte value instead of bit count
    # Correct definition at line 490 returns self.bit_count(self.read_m(0xD356))
