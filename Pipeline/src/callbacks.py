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
        stats_lists = self.training_env.get_attr("agent_stats")
        for env_stats in stats_lists:
            if env_stats:
                self.current_stats.extend(env_stats)
                self.training_env.set_attr("agent_stats", [])
        if len(self.current_stats) >= self.save_freq:
            file_name = self.save_path / f"stats_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(self.current_stats, f, default=lambda o: int(o) if isinstance(o, np.int64) else o)
            if self.verbose:
                print(f"Saved {len(self.current_stats)} stats to {file_name}")
            self.current_stats = []

    def _on_training_end(self) -> None:
        if self.current_stats:
            file_name = self.save_path / f"stats_final_{int(time.time())}.json"
            with open(file_name, "w") as f:
                json.dump(self.current_stats, f, default=lambda o: int(o) if isinstance(o, np.int64) else o)
            if self.verbose:
                print(f"Final stats saved to {file_name}")
