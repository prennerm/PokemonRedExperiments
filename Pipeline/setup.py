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
        "stable-baselines3",
        "sb3-contrib",
        "pyboy",
        "gymnasium",
        "numpy",
        "einops",
        "scikit-image",
        "mediapy",
        "websockets",
        "imageio",
        "pandas",
        "scipy",
        "tensorboard",
        "pyyaml",
    ],
)
