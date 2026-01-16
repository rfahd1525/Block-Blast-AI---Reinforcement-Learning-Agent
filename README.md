# Block Blast AI - Reinforcement Learning Agent

A deep reinforcement learning agent that learns to play Block Blast, achieving professional-level performance through self-play using **PPO with Action Masking**.

## 📄 Project Report

[**View the Full Project Report (PDF)**](./Block_Blast_RL_Report.pdf)

## About Block Blast

Block Blast is a strategic puzzle game played on an 8×8 grid. Players receive 3 random pieces per turn from a pool of 37 possible shapes. The objective is to place pieces to complete full rows or columns, which then clear and award points. Strategic play involves creating combo setups for maximum score multipliers.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BBML

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## GUI Mode (Recommended)

Launch the visual interface to train and watch the AI play:

```bash
# Start GUI
python run_gui.py
```

The GUI provides:
- **Watch AI Play**: Visual 8x8 game board with real-time statistics
- **Training Mode**: Start/stop training with live progress updates
- **Model Selection**: Choose between best, latest, or final checkpoints
- **Speed Control**: Adjust playback speed when watching AI

## Terminal Mode

### Training

```bash
# Start training with default configuration
python run_train.py

# Train with custom config
python run_train.py --config config/long_train.yaml

# Resume from checkpoint
python run_train.py --resume checkpoints/latest.pt
```

### Evaluation

```bash
# Evaluate a trained model
python scripts/evaluate.py --checkpoint checkpoints/best.pt --episodes 100

# Watch the AI play in terminal
python scripts/play.py --mode watch --checkpoint checkpoints/best.pt

# Play manually
python scripts/play.py --mode manual
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Project Structure

```
BBML/
├── run_gui.py              # GUI launcher (recommended)
├── run_train.py            # Terminal training launcher
├── config/                 # Configuration files
│   ├── default.yaml       # Default hyperparameters
│   └── long_train.yaml    # Extended training config
├── gui/                    # Native GUI
│   └── app.py             # Tkinter application
├── src/                    # Source code
│   ├── game/              # Game engine
│   │   ├── board.py       # Board state management
│   │   ├── pieces.py      # 37 piece definitions
│   │   ├── engine.py      # Game logic
│   │   └── renderer.py    # ASCII visualization
│   ├── environment/       # RL environment
│   │   ├── block_blast_env.py  # Gym environment
│   │   └── wrappers.py    # Environment wrappers
│   ├── models/            # Neural networks
│   │   ├── network.py     # CNN with action masking
│   │   └── utils.py       # Model utilities
│   ├── agents/            # RL agents
│   │   ├── ppo.py         # PPO with action masking
│   │   └── base.py        # Base agent class
│   └── utils/             # Utilities
│       ├── logger.py      # JSONL logging
│       └── device.py      # GPU/CPU detection
├── scripts/               # Executable scripts
│   ├── train.py          # Training script
│   ├── evaluate.py       # Evaluation script
│   ├── play.py           # Interactive play
│   └── benchmark.py      # Performance benchmarking
├── tests/                 # Unit tests
├── checkpoints/           # Saved models
├── logs/                  # Training logs (JSONL format)
└── results/               # Evaluation results
```

## Algorithm

This project uses **Masked PPO (Proximal Policy Optimization)** which:
- **Action Masking**: Invalid moves receive -∞ logits, ensuring the agent only samples valid placements
- **Convolutional Network**: 4-channel input (board + 3 piece masks) processed through residual CNN
- **Generalized Advantage Estimation (GAE)**: Balanced bias-variance tradeoff for advantage computation
- **Vectorized Environments**: 64 parallel games for efficient experience collection
- **Normalized Rewards**: Scaled rewards for stable training convergence

The action masking is implemented at multiple levels:
1. Game engine computes valid placements for each piece
2. Environment provides action mask in observations
3. Network applies mask by adding -∞ to invalid action logits
4. Softmax ensures zero probability for invalid actions

## Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir logs/
```

