import os, sys
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, str(Path("src").resolve()))
from trainers.base_trainer import BaseTrainer, TrainerArgs
from trainers.default_trainer import DefaultTrainer
cfg = BaseTrainer.load_config(Path("configs/v1.yaml"))
args = TrainerArgs(variant="v1", config_path=Path("configs/v1.yaml"), resume_path=None)
trainer = DefaultTrainer(args, cfg)
print("init ok")

