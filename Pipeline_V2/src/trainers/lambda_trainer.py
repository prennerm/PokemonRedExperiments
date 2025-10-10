from __future__ import annotations

import warnings
import json
from copy import deepcopy
from pathlib import Path
from typing import IO, Any, Dict, Optional, Type

from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
from stable_baselines3.common.callbacks import BaseCallback

from models import MultiInputLstmPolicyLD, RecurrentPPOLD

from .default_trainer import DefaultTrainer


class LambdaDiagnosticsCallback(BaseCallback):
    """Log λ-discrepancy metrics to JSONL for offline analysis."""

    def __init__(self, log_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_path = log_path
        self._buffer: list[Dict[str, Any]] = []
        self._fh: Optional[IO[str]] = None

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.log_path.open("a", encoding="utf-8")

    def _on_training_end(self) -> None:
        self._flush()
        if self._fh:
            self._fh.close()
            self._fh = None

    def _on_rollout_end(self) -> None:
        self._collect_metrics()

    def _on_training_step(self) -> bool:
        self._collect_metrics()
        return True

    def _on_step(self) -> bool:
        return True

    def _collect_metrics(self) -> None:
        metrics = {
            "ld_loss": self.logger.name_to_value.get("train/ld_loss"),
            "mc_loss": self.logger.name_to_value.get("train/mc_loss"),
            "policy_loss": self.logger.name_to_value.get("train/policy_loss"),
            "entropy_loss": self.logger.name_to_value.get("train/entropy_loss"),
            "ld_target_abs_mean": self.logger.name_to_value.get("train/ld_target_abs_mean"),
            "bootstrap_vs_mc_ratio": self.logger.name_to_value.get("train/bootstrap_vs_mc_ratio"),
        }
        if all(value is None for value in metrics.values()):
            return
        record = {"total_timesteps": int(self.model.num_timesteps)}
        record.update({k: self._cast(v) for k, v in metrics.items() if v is not None})
        self._buffer.append(record)
        if len(self._buffer) >= 25:
            self._flush()

    def _flush(self) -> None:
        if not self._buffer or not self._fh:
            return
        for entry in self._buffer:
            self._fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._fh.flush()
        self._buffer.clear()

    @staticmethod
    def _cast(value: Any) -> Any:
        try:
            import numpy as np  # local import to avoid mandatory dependency at module import

            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return value


class LambdaTrainer(DefaultTrainer):
    """Trainer for λ-discrepancy experiments (v4)."""

    _EXPECTED_MODEL_TYPE = "RecurrentPPOLD"
    _EXPECTED_POLICY_CLASS: Type[MultiInputLstmPolicy] = MultiInputLstmPolicyLD
    _EXPECTED_ENV_MODULE = "lstm_env"
    _EXPECTED_ENV_CLASS = "LSTMEnv"

    def __init__(self, args, cfg):
        prepared_cfg = self._prepare_lambda_config(cfg, variant=args.variant)
        super().__init__(args, prepared_cfg)

    def _prepare_lambda_config(self, cfg: Dict[str, Any], variant: str) -> Dict[str, Any]:
        cfg_copy = deepcopy(cfg)

        if "model" not in cfg_copy:
            raise ValueError("Config missing 'model' section required for LambdaTrainer")
        if "env" not in cfg_copy:
            raise ValueError("Config missing 'env' section required for LambdaTrainer")

        model_cfg = cfg_copy["model"]
        env_cfg = cfg_copy["env"]

        original_type = model_cfg.get("type")
        if original_type != self._EXPECTED_MODEL_TYPE:
            if original_type is not None:
                warnings.warn(
                    (
                        f"Override model.type={original_type!r} with "
                        f"{self._EXPECTED_MODEL_TYPE!r} for Lambda variant"
                    ),
                    UserWarning,
                )
            model_cfg["type"] = self._EXPECTED_MODEL_TYPE

        policy_value = model_cfg.get("policy")
        expected_name = self._EXPECTED_POLICY_CLASS.__name__
        if policy_value not in (self._EXPECTED_POLICY_CLASS, expected_name, None):
            warnings.warn(
                (
                    f"Ignoring custom policy {policy_value!r}; "
                    f"using {expected_name}"
                ),
                UserWarning,
            )
        model_cfg["policy"] = self._EXPECTED_POLICY_CLASS

        ld_coef = model_cfg.get("ld_coef")
        if ld_coef is None:
            warnings.warn(
                "ld_coef missing for Lambda variant; defaulting to 0.1",
                UserWarning,
            )
            model_cfg["ld_coef"] = 0.1
        elif not isinstance(ld_coef, (int, float)) or ld_coef <= 0:
            raise ValueError(
                f"LambdaTrainer requires positive numeric ld_coef, got {ld_coef!r}"
            )

        module_name = env_cfg.get("module")
        if module_name != self._EXPECTED_ENV_MODULE:
            warnings.warn(
                (
                    f"Override env.module={module_name!r} with "
                    f"{self._EXPECTED_ENV_MODULE!r} for Lambda variant"
                ),
                UserWarning,
            )
            env_cfg["module"] = self._EXPECTED_ENV_MODULE

        class_name = env_cfg.get("class")
        if class_name != self._EXPECTED_ENV_CLASS:
            warnings.warn(
                (
                    f"Override env.class={class_name!r} with "
                    f"{self._EXPECTED_ENV_CLASS!r} for Lambda variant"
                ),
                UserWarning,
            )
            env_cfg["class"] = self._EXPECTED_ENV_CLASS

        self._validate_rollout_settings(cfg_copy, variant)

        return cfg_copy

    def _validate_rollout_settings(self, cfg: Dict[str, Any], variant: str) -> None:
        num_cpu = cfg.get("num_cpu", 1)
        model_cfg = cfg["model"]

        n_steps = model_cfg.get("n_steps")
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(
                f"LambdaTrainer requires integer model.n_steps > 0 (variant={variant})"
            )

        if n_steps % num_cpu != 0:
            raise ValueError(
                (
                    f"model.n_steps ({n_steps}) must be divisible by num_cpu ({num_cpu}) "
                    f"for Lambda variant {variant}"
                )
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
        if not any(isinstance(cb, LambdaDiagnosticsCallback) for cb in callbacks):
            log_file = Path(self.dirs["logs"]) / "lambda_metrics.jsonl"
            callbacks.append(LambdaDiagnosticsCallback(log_file, verbose=int(self.cfg.get("debug", 0))))
        return callbacks
