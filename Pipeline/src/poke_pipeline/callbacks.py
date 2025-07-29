import json
import time
from pathlib import Path
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class StatsCallback(BaseCallback):
    """
    Callback zum Sammeln und Speichern von Trainingsstatistiken.
    Dieser Callback sammelt die in den Environments gespeicherten agent_stats und
    schreibt alle 100 Einträge in eine neue JSON-Datei.
    """
    def __init__(self, save_freq: int = 100, save_path: str = "stats_logs", verbose: int = 0):
        super(StatsCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.current_stats = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        # 1) Alle agent_stats der Sub-Envs einsammeln
        stats_lists = self.training_env.get_attr("agent_stats")

        for env_stats in stats_lists:          # ← alle Env-Listen durchgehen
            if env_stats:                      #   nur wenn nicht leer
                # Strukturiere die Daten gemäß logging_guidelines.md
                structured_stats = [self._restructure_stats(stat) for stat in env_stats]
                self.current_stats.extend(structured_stats)

        # 2) Jetzt einmal global zurücksetzen  (bisher stand das im Loop)
        self.training_env.set_attr("agent_stats", [])

        # 3) Flush-Schwelle prüfen
        if len(self.current_stats) >= self.save_freq:
            file_name = self.save_path / f"stats_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(
                    self.current_stats,
                    f,
                    default=lambda o: int(o) if isinstance(o, np.integer) else o,
                    indent=2
                )
            if self.verbose:
                print(f"Saved {len(self.current_stats)} stats to {file_name}")
            self.current_stats = []

    def _on_training_end(self) -> None:
        if self.current_stats:
            file_name = self.save_path / f"stats_final_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(self.current_stats, f, default=lambda o: int(o) if isinstance(o, np.int64) else o, indent=2)
            if self.verbose:
                print(f"Final stats saved to {file_name}")

    def _restructure_stats(self, raw_stats):
        """
        Strukturiert die rohen agent_stats gemäß den logging_guidelines.md
        in eine übersichtliche, verschachtelte JSON-Struktur um.
        """
        # Extrahiere reward_components falls vorhanden, sonst erstelle leeres dict
        reward_components = raw_stats.get("reward_components", {})
        
        structured = {
            "step": raw_stats.get("step", 0),
            "position": {
                "x": raw_stats.get("x", 0),
                "y": raw_stats.get("y", 0),
                "map": raw_stats.get("map", 0)
            },
            "rewards": {
                "total": raw_stats.get("reward_total", 0),
                "step": raw_stats.get("reward_step", 0),
                "components": {
                    "event": reward_components.get("event", raw_stats.get("reward_event", 0)),
                    "level": reward_components.get("level", raw_stats.get("reward_level", 0)),
                    "heal": reward_components.get("heal", raw_stats.get("reward_heal", 0)),
                    "badge": reward_components.get("badge", raw_stats.get("reward_badge", 0)),
                    "explore": reward_components.get("explore", raw_stats.get("reward_explore", 0)),
                    "dead": reward_components.get("dead", raw_stats.get("reward_dead", 0)),
                    "stuck": reward_components.get("stuck", raw_stats.get("reward_stuck", 0))
                }
            },
            "player_status": {
                "health": raw_stats.get("hp", 0),
                "levels": raw_stats.get("levels", []),
                "levels_sum": raw_stats.get("levels_sum", 0),
                "badges": raw_stats.get("badge", 0),
                "pokemon_count": raw_stats.get("pcount", 0),
                "pokemon_types": raw_stats.get("ptypes", [])
            },
            "actions": {
                "last_action": raw_stats.get("last_action", 0)
            },
            "statistics": {
                "deaths": raw_stats.get("deaths", 0),
                "exploration_coords": raw_stats.get("coord_count", 0),
                "map_progress": raw_stats.get("max_map_progress", 0),
                "healing_reward": raw_stats.get("healr", 0),
                "event_progress": raw_stats.get("event", 0)
            }
        }
        
        return structured
