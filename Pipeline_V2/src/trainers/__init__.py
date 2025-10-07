from __future__ import annotations

from typing import Dict, Type

from .base_trainer import BaseTrainer
from .default_trainer import DefaultTrainer
from .lstm_trainer import LSTMTrainer
from .lambda_trainer import LambdaTrainer

TRAINER_REGISTRY: Dict[str, Type[BaseTrainer]] = {
    "v1": DefaultTrainer,
    "v2": DefaultTrainer,
    "v3": LSTMTrainer,
    "v4": LambdaTrainer,
}
