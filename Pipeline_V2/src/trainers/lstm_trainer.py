from __future__ import annotations

import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any, Dict, List

from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.callbacks import BaseCallback

from .default_trainer import DefaultTrainer


class _LstmDiagnosticsCallback(BaseCallback):
    """Record basic diagnostics about LSTM episode starts and reset cadence."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    @staticmethod
    def _ensure_sequence(value: Any) -> List[Any]:
        if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
            return list(value)
        return [value]

    def _on_rollout_end(self) -> None:
        episode_flags = self._ensure_sequence(self.training_env.get_attr("episode_start"))
        active_starts = sum(bool(flag) for flag in episode_flags)
        total_envs = len(episode_flags)
        self.logger.record("lstm/episode_start_envs", active_starts)
        if self.verbose:
            print(
                f"[LSTM] episode_start flags currently active: "
                f"{active_starts}/{total_envs}"
            )

        reset_counts = self._ensure_sequence(self.training_env.get_attr("reset_count"))
        max_resets = max(reset_counts) if reset_counts else 0
        self.logger.record("lstm/max_reset_count", max_resets)


class LSTMTrainer(DefaultTrainer):
    """Trainer for LSTM variants (v3) with automatic safeguards."""

    _EXPECTED_MODEL_TYPE = "RecurrentPPO"
    _EXPECTED_POLICY = MultiInputLstmPolicy
    _EXPECTED_ENV_MODULE = "lstm_env"
    _EXPECTED_ENV_CLASS = "LSTMEnv"

    def __init__(self, args, cfg):
        prepared_cfg = self._prepare_lstm_config(cfg, variant=args.variant)
        super().__init__(args, prepared_cfg)

    def _prepare_lstm_config(self, cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
        cfg_copy = deepcopy(cfg)

        if "model" not in cfg_copy:
            raise ValueError("Config missing 'model' section required for LSTMTrainer")
        if "env" not in cfg_copy:
            raise ValueError("Config missing 'env' section required for LSTMTrainer")

        model_cfg = cfg_copy["model"]
        env_cfg = cfg_copy["env"]

        original_type = model_cfg.get("type")
        if original_type != self._EXPECTED_MODEL_TYPE:
            if original_type is not None:
                warnings.warn(
                    (
                        f"Override model.type={original_type!r} with "
                        f"{self._EXPECTED_MODEL_TYPE!r} for LSTM variant"
                    ),
                    UserWarning,
                )
            model_cfg["type"] = self._EXPECTED_MODEL_TYPE

        policy_value = model_cfg.get("policy")
        if policy_value not in (self._EXPECTED_POLICY, self._EXPECTED_POLICY.__name__, None):
            warnings.warn(
                (
                    f"Ignoring custom policy {policy_value!r}; "
                    f"using {self._EXPECTED_POLICY.__name__}"
                ),
                UserWarning,
            )
        model_cfg["policy"] = self._EXPECTED_POLICY

        if "ld_coef" in model_cfg:
            warnings.warn(
                "ld_coef is not applicable for pure LSTM (v3) training and will be ignored",
                UserWarning,
            )
            model_cfg.pop("ld_coef")

        module_name = env_cfg.get("module")
        if module_name != self._EXPECTED_ENV_MODULE:
            warnings.warn(
                (
                    f"Override env.module={module_name!r} with "
                    f"{self._EXPECTED_ENV_MODULE!r} for LSTM variant"
                ),
                UserWarning,
            )
            env_cfg["module"] = self._EXPECTED_ENV_MODULE

        class_name = env_cfg.get("class")
        if class_name != self._EXPECTED_ENV_CLASS:
            warnings.warn(
                (
                    f"Override env.class={class_name!r} with "
                    f"{self._EXPECTED_ENV_CLASS!r} for LSTM variant"
                ),
                UserWarning,
            )
            env_cfg["class"] = self._EXPECTED_ENV_CLASS

        self._validate_rollout_hyperparameters(cfg_copy, variant)

        return cfg_copy

    def _validate_rollout_hyperparameters(self, cfg: Dict[str, Any], variant: str) -> None:
        num_cpu = cfg.get("num_cpu", 1)
        model_cfg = cfg["model"]
        env_cfg = cfg["env"]

        n_steps = model_cfg.get("n_steps")
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(
                f"LSTMTrainer requires integer model.n_steps > 0 (variant={variant})"
            )

        if n_steps % num_cpu != 0:
            raise ValueError(
                (
                    f"model.n_steps ({n_steps}) must be divisible by num_cpu ({num_cpu}) "
                    f"for LSTM variant {variant}"
                )
            )

        max_steps = env_cfg.get("max_steps")
        if isinstance(max_steps, int) and max_steps > 0 and max_steps % n_steps != 0:
            warnings.warn(
                (
                    f"env.max_steps ({max_steps}) is not divisible by n_steps ({n_steps}); "
                    "rollout alignment may drift over time"
                ),
                UserWarning,
            )

        batch_size = model_cfg.get("batch_size")
        if isinstance(batch_size, int) and batch_size % num_cpu != 0:
            warnings.warn(
                (
                    f"model.batch_size ({batch_size}) is not divisible by num_cpu ({num_cpu}); "
                    "consider adjusting to avoid ragged minibatches"
                ),
                UserWarning,
            )

    def _build_callbacks(self):
        callbacks = super()._build_callbacks()
        verbose_flag = int(bool(self.env_conf.get("debug", 0)))
        callbacks.append(_LstmDiagnosticsCallback(verbose=verbose_flag))
        return callbacks
