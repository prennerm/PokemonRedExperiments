# setup.py
from setuptools import setup, find_packages

setup(
    name="poke_pipeline",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={"poke_pipeline": ["data/*.json"]},
    install_requires=[
        # Core RL Libraries
        "stable-baselines3",
        "sb3-contrib",
        
        # Environment and Game Engine
        "pyboy",
        "gymnasium",
        
        # Data Processing
        "numpy",
        "einops",
        "scikit-image",
        "mediapy",
        "pandas",
        "scipy",
        
        # Utilities
        "websockets",
        "imageio",
        "tensorboard",
        "pyyaml",
        
        # NOTE: PyTorch mit CUDA wird über conda/environment.yml installiert
        # um Hardware-spezifische CUDA-Versionen korrekt zu handhaben
    ],
)
