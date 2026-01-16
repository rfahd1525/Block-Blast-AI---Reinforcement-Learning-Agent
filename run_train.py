#!/usr/bin/env python3
"""
Block Blast AI - Terminal Training Launcher

Start training from the command line with progress output.
This is the recommended way to train without the GUI.
"""
import sys
import os
import argparse
from pathlib import Path

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import torch
    except ImportError:
        missing.append("torch")

    try:
        import numpy
    except ImportError:
        missing.append("numpy")

    try:
        import gymnasium
    except ImportError:
        missing.append("gymnasium")

    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("\nInstall them with:")
        print(f"  pip install {' '.join(missing)}")
        return False

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Block Blast AI - Terminal Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_train.py                    # Train with default config
  python run_train.py --config config/long_train.yaml  # Use long training config
  python run_train.py --resume checkpoints/latest.pt   # Resume training
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file (default: config/default.yaml)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Print banner
    print("""
    ╔══════════════════════════════════════════╗
    ║      Block Blast AI - Training           ║
    ║   PPO with Action Masking                ║
    ╚══════════════════════════════════════════╝
    """)

    # Import and run training
    from scripts.train import train, load_config

    config_path = project_root / args.config
    if config_path.exists():
        config = load_config(str(config_path))
        print(f"Loaded config from: {config_path}")
    else:
        print(f"Config file not found: {config_path}")
        print("Using default configuration")
        config = {
            'environment': {'board_size': 8},
            'ppo': {
                'learning_rate': 3e-4,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_epsilon': 0.2,
                'entropy_coef': 0.01,
                'value_coef': 0.5,
                'max_grad_norm': 0.5,
            },
            'training': {
                'num_envs': 64,
                'batch_size': 2048,
                'num_epochs': 10,
                'rollout_steps': 128,
                'total_timesteps': 10_000_000,
            },
            'rewards': {
                'line_clear_base': 1.0,
                'block_placed': 0.01,
                'game_over_penalty': -1.0,
            },
            'logging': {
                'log_interval': 10,
                'save_interval': 100,
                'eval_interval': 50,
            },
            'paths': {
                'checkpoint_dir': 'checkpoints',
                'log_dir': 'logs',
                'results_dir': 'results',
            },
        }

    # Show config summary
    train_config = config.get('training', {})
    print(f"Training Configuration:")
    print(f"  Total timesteps: {train_config.get('total_timesteps', 10_000_000):,}")
    print(f"  Parallel envs: {train_config.get('num_envs', 64)}")
    print(f"  Batch size: {train_config.get('batch_size', 2048)}")
    print(f"  Rollout steps: {train_config.get('rollout_steps', 128)}")
    print()

    # Start training
    try:
        train(config, resume_path=args.resume, seed=args.seed)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")


if __name__ == "__main__":
    main()
