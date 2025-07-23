# simple_grid.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class OneByFourGrid(gym.Env):
    """
    Sehr einfaches 1×4-Grid:
    - State: One-Hot über 4 Felder (als Dict unter key "state")
    - Action: 0=links, 1=rechts
    - Reward: +1, wenn Feld 3 erreicht, sonst 0
    - Episode endet nach 10 Steps oder auf Feld 3.
    """
    def __init__(self):
        super().__init__()
        # Observation als Dict, damit SB3’s Dict-Feature-Extractor funktioniert
        self.observation_space = spaces.Dict({
            "state": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        })
        self.action_space = spaces.Discrete(2)
        self.max_steps = 10
        # interner Zähler
        self._steps = None
        self._pos = None

    def reset(self, *, seed=None, options=None):
        # Gymnasium erwartet Signatur reset(self, *, seed=None, options=None)
        # Wir ignorieren seed hier
        super().reset(seed=seed)
        self._steps = 0
        self._pos = 0
        obs = self._make_obs()
        info = {}  # Du kannst hier später z.B. {"time": ...} packen
        return obs, info

    def step(self, action):
        self._steps += 1
        # Bewegung
        if action == 1 and self._pos < 3:
            self._pos += 1
        elif action == 0 and self._pos > 0:
            self._pos -= 1
        # Terminal‐Check
        done = (self._pos == 3) or (self._steps >= self.max_steps)
        reward = 1.0 if self._pos == 3 else 0.0
        obs = self._make_obs()
        info = {}
        return obs, reward, done, False, info
        # Für Gymnasium: return obs, reward, done, truncated, info
        # truncated hier False, da wir max_steps als done behandeln

    def _make_obs(self):
        o = np.zeros(4, dtype=np.float32)
        o[self._pos] = 1.0
        return {"state": o}
