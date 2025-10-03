#!/usr/bin/env python3
"""
test_poke_viz_setup.py - Test der erweiterten poke_viz Installation
"""

def test_imports():
    """Teste alle wichtigen Imports für Memory Analysis"""
    try:
        # Core ML/RL
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Stable Baselines
        from stable_baselines3.common.vec_env import DummyVecEnv
        from sb3_contrib.ppo_recurrent.ppo_recurrent import RecurrentPPO
        from sb3_contrib.ppo_recurrent.policies import MultiInputLstmPolicy
        
        # Project modules
        from src.poke_pipeline.ppo_lambda_discrepancy import RecurrentPPOLD, MultiInputLstmPolicyLD
        from src.poke_pipeline.red_gym_env_lstm import RedGymEnvLSTM
        
        # Analysis tools
        import sklearn
        import umap
        
        print("✅ Alle Core Imports erfolgreich!")
        
        # Test PyTorch CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA verfügbar: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA nicht verfügbar (CPU-only)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Fehler: {e}")
        return False

def test_file_access():
    """Teste Zugriff auf benötigte Dateien"""
    from pathlib import Path
    
    required_files = [
        "data/PokemonRed.gb",
        "data/init.state", 
        "src/poke_pipeline/__init__.py"
    ]
    
    all_good = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} gefunden")
        else:
            print(f"❌ {file_path} nicht gefunden")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("🧪 Teste poke_viz Setup für Memory Analysis...")
    print("=" * 50)
    
    imports_ok = test_imports()
    files_ok = test_file_access()
    
    print("=" * 50)
    if imports_ok and files_ok:
        print("🎉 Setup erfolgreich! Memory Analysis sollte funktionieren.")
    else:
        print("⚠️  Setup unvollständig. Bitte fehlende Dependencies installieren.")