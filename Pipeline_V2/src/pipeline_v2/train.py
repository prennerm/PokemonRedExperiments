import argparse
import yaml
from pathlib import Path

import logging
import warnings

# Suppress SDL2 and PyBoy warnings
warnings.filterwarnings("ignore", message="Using SDL2 binaries from pysdl2-dll")
logging.getLogger("pyboy.core.sound").setLevel(logging.ERROR)
logging.getLogger("pyboy.core.mb").setLevel(logging.ERROR)

from trainers.base_trainer import BaseTrainer, TrainerArgs
from trainers.default_trainer import DefaultTrainer



def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent on Pokémon Red environments")
    parser.add_argument(
        "--variant",
        choices=["v1", "v2", "v3", "v4"],
        required=True,
        help="v1: Baseline (RedGymEnv+StreamWrapper), v2/v3: RedGymEnv, v4: RedGymEnvLSTM",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Pfad zur YAML-Konfig, z.B. configs/v1.yaml",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Pfad zum Checkpoint (.zip) um Training fortzusetzen",
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

    trainer = DefaultTrainer(trainer_args, cfg)
    trainer.run()


if __name__ == "__main__":
    main()
