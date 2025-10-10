"""
Pokemon Red RL Pipeline V2 - Training CLI

Usage:
    python train.py --variant v4 --config configs/v4_ld_025.yaml
    python train.py --variant v1 --config configs/v1.yaml --resume experiments/v1/.../model.zip
"""
import argparse
import logging
import sys
import warnings
from pathlib import Path

# Suppress SDL2 and PyBoy warnings
warnings.filterwarnings("ignore", message="Using SDL2 binaries from pysdl2-dll")
logging.getLogger("pyboy.core.sound").setLevel(logging.ERROR)
logging.getLogger("pyboy.core.mb").setLevel(logging.ERROR)

# Ensure src/ is on sys.path
SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from trainers import TRAINER_REGISTRY
from trainers.base_trainer import BaseTrainer, TrainerArgs
from trainers.default_trainer import DefaultTrainer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RL agent on Pokémon Red environments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py --variant v4 --config configs/v4_ld_025.yaml
  python train.py --variant v1 --config configs/v1.yaml --resume experiments/v1/20241010_120000/checkpoints/model_latest.zip

Variants:
  v1: PPO + Frame Stacking (3 frames)
  v2: PPO + Single Frame
  v3: Recurrent PPO + LSTM
  v4: Recurrent PPO + LSTM + Lambda Discrepancy
        """
    )
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v3", "v4"],
        required=True,
        help="Training variant to use"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to YAML config file (e.g., configs/v4_ld_025.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint (.zip) to resume training"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = BaseTrainer.load_config(args.config)

    trainer_args = TrainerArgs(
        variant=args.variant,
        config_path=args.config,
        resume_path=args.resume,
    )

    trainer_cls = TRAINER_REGISTRY.get(args.variant, DefaultTrainer)
    trainer = trainer_cls(trainer_args, cfg)
    trainer.run()


if __name__ == "__main__":
    main()
