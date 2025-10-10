import sys
from pathlib import Path
sys.path.insert(0, 'src')

from trainers.base_trainer import BaseTrainer

for variant in ['v1', 'v2', 'v3', 'v4']:
    print(f"\n{'='*50}")
    print(f"Testing {variant}.yaml")
    print('='*50)
    cfg_path = Path(f'configs/{variant}.yaml')
    try:
        cfg = BaseTrainer.load_config(cfg_path)
        print(f"✓ Config loaded successfully")
        print(f"  model.n_steps: {cfg.get('model', {}).get('n_steps')}")
        print(f"  env.max_steps: {cfg.get('env', {}).get('max_steps')}")
        print(f"  env.module: {cfg.get('env', {}).get('module')}")
        print(f"  model.type: {cfg.get('model', {}).get('type')}")
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
