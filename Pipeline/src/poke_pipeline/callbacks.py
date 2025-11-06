import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class StatsWriter:
    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class JsonStatsWriter(StatsWriter):
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        if not stats:
            return
        suffix = "final" if final else str(int(time.time()))
        file_path = self.base_path / f"stats_{suffix}.json"
        with file_path.open("w", encoding="utf-8") as fh:
            json.dump(stats, fh, ensure_ascii=False, indent=2)


class CsvStatsWriter(StatsWriter):
    def __init__(self, base_path: Path) -> None:
        self.base_path = base_path

    def write_chunk(self, stats: List[Dict[str, Any]], final: bool = False) -> None:
        if not stats:
            return
        flattened = [flatten_dict(stat) for stat in stats]
        fieldnames = sorted({key for row in flattened for key in row.keys()})
        suffix = "final" if final else str(int(time.time()))
        file_path = self.base_path / f"stats_{suffix}.csv"
        with file_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in flattened:
                writer.writerow({key: row.get(key, "") for key in fieldnames})


def flatten_dict(stats: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in stats.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


class StatsCallback(BaseCallback):
    """Sammelt agent_stats und schreibt sie batched als JSON oder CSV."""

    def __init__(
        self,
        save_freq: int = 100,
        save_path: str = "stats_logs",
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

    def _build_writer(self, output_format: str) -> StatsWriter:
        fmt = output_format.lower()
        if fmt == "json":
            return JsonStatsWriter(self.save_path)
        if fmt == "csv":
            return CsvStatsWriter(self.save_path)
        raise ValueError(f"Unsupported stats output format: {output_format}")

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        stats_lists = self.training_env.get_attr("agent_stats")
        for env_stats in stats_lists:
            if not env_stats:
                continue
            prepared = [self._prepare_stat(stat) for stat in env_stats]
            self.current_stats.extend(prepared)
        self.training_env.set_attr("agent_stats", [])
        if len(self.current_stats) >= self.save_freq:
            self._flush()

    def _on_training_end(self) -> None:
        self._flush(final=True)
        self.writer.close()

    def _prepare_stat(self, raw_stats: Dict[str, Any]) -> Dict[str, Any]:
        normalised = _normalise_stat(raw_stats)
        normalised["total_steps"] = self.num_timesteps
        training_metrics = None
        metrics_source = getattr(self.model, "latest_train_metrics", None)
        if isinstance(metrics_source, dict):
            training_metrics = _normalise_stat(metrics_source)
            if training_metrics:
                normalised["training"] = training_metrics

        if not self.structured:
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
        if training_metrics:
            structured["training"] = training_metrics
        return structured

    def _flush(self, *, final: bool = False) -> None:
        if not self.current_stats:
            return
        self.writer.write_chunk(self.current_stats, final=final)
        if self.verbose:
            label = "final" if final else f"batch ({len(self.current_stats)} entries)"
            print(f"[StatsCallback] wrote {label} to {self.save_path}")
        self.current_stats = []


def _normalise_stat(stat: Dict[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
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
