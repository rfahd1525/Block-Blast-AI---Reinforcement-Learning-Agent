"""Setup script for Block Blast AI."""
from setuptools import setup, find_packages

setup(
    name="block-blast-ai",
    version="1.0.0",
    description="Block Blast Reinforcement Learning Agent",
    author="Block Blast AI Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "gymnasium>=0.29.0",
        "tensorboard>=2.14.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bbai-train=scripts.train:main",
            "bbai-evaluate=scripts.evaluate:main",
            "bbai-play=scripts.play:main",
        ],
    },
)
