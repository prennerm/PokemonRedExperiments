from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .writers import CsvStatsWriter, JsonStatsWriter, StatsWriter


class StatsCallback(BaseCallback):
    """Collect per-env agent stats and persist in the requested format."""

    def __init__(
        self,
        save_path: Path,
        save_freq: int = 100,
        *,
        output_format: str = "json",
        structured: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.save_freq = max(1, int(save_freq))
        self.structured = structured
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.current_stats: List[Dict[str, Any]] = []
        self.writer = self._build_writer(output_format)

    # ------------------------------------------------------------------
    # Stable-Baselines Hooks
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        stats_lists = self.training_env.get_attr("agent_stats")
        per_env_counts = []
        for env_stats in stats_lists:
            count = len(env_stats) if env_stats else 0
            per_env_counts.append(count)
            if env_stats:
                structured_stats = [self._prepare_stat(stat) for stat in env_stats]
                self.current_stats.extend(structured_stats)

        self.training_env.set_attr("agent_stats", [])

        if self.verbose:
            total_stats = sum(per_env_counts)
            max_env = max(per_env_counts) if per_env_counts else 0
            print(
                f"[StatsCallback] collected={total_stats} max_per_env={max_env} buffer={len(self.current_stats)}"
            )

        if len(self.current_stats) >= self.save_freq:
            self._flush()

    def _on_training_end(self) -> None:
        if self.current_stats:
            self._flush(final=True)
        self.writer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_writer(self, output_format: str) -> StatsWriter:
        fmt = output_format.lower()
        if fmt == "json":
            return JsonStatsWriter(self.save_path)
        if fmt == "csv":
            return CsvStatsWriter(self.save_path)
        raise ValueError(f"Unsupported logging.format: {output_format}")

    def _flush(self, *, final: bool = False) -> None:
        if not self.current_stats:
            return
        buffer = self.current_stats
        self.current_stats = []
        self.writer.write_chunk(buffer, final=final)

    def _prepare_stat(self, raw_stats: Dict[str, Any]) -> Dict[str, Any]:
        normalised = _normalise_stat(raw_stats)
        normalised["total_steps"] = self.num_timesteps

        if not self.structured:
            return normalised

        # If the env already returns structured stats (current BaseRedGymEnv), keep them.
        if isinstance(normalised.get("rewards"), dict):
            return normalised

        reward_components = normalised.get("reward_components", {})

        structured = {
            "step": normalised.get("step", 0),
            "total_steps": self.num_timesteps,
            "position": {
                "x": normalised.get("x", 0),
                "y": normalised.get("y", 0),
                "map": normalised.get("map", 0),
            },
            "rewards": {
                "total": normalised.get("reward_total", 0),
                "step": normalised.get("reward_step", 0),
                "components": {
                    "event": reward_components.get("event", normalised.get("reward_event", 0)),
                    "level": reward_components.get("level", normalised.get("reward_level", 0)),
                    "heal": reward_components.get("heal", normalised.get("reward_heal", 0)),
                    "badge": reward_components.get("badge", normalised.get("reward_badge", 0)),
                    "explore": reward_components.get("explore", normalised.get("reward_explore", 0)),
                    "dead": reward_components.get("dead", normalised.get("reward_dead", 0)),
                    "stuck": reward_components.get("stuck", normalised.get("reward_stuck", 0)),
                },
            },
            "player_status": {
                "health": normalised.get("hp", 0),
                "levels": normalised.get("levels", []),
                "levels_sum": normalised.get("levels_sum", 0),
                "badges": normalised.get("badge", 0),
                "pokemon_count": normalised.get("pcount", 0),
                "pokemon_types": normalised.get("ptypes", []),
            },
            "actions": {"last_action": normalised.get("last_action", 0)},
            "statistics": {
                "deaths": normalised.get("deaths", 0),
                "exploration_coords": normalised.get("coord_count", 0),
                "map_progress": normalised.get("max_map_progress", 0),
                "healing_reward": normalised.get("healr", 0),
                "event_progress": normalised.get("event", 0),
            },
        }

        return structured


def _normalise_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    normalised = {}
    for key, value in stat.items():
        if isinstance(value, dict):
            normalised[key] = _normalise_stat(value)
        elif isinstance(value, list):
            normalised[key] = [_convert_value(item) for item in value]
        else:
            normalised[key] = _convert_value(value)
    return normalised


def _convert_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
