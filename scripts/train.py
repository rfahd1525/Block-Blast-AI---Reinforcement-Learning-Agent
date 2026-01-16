"""
Training script for Block Blast AI.

This script handles the complete training loop including:
- Environment management
- PPO training
- Logging and checkpointing
- Evaluation
"""
import argparse
import os
import sys
import time
import signal
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable
import yaml
import numpy as np
import torch
from datetime import datetime

# Only set signal handler in main thread (prevents errors when imported from GUI threads)
if threading.current_thread() is threading.main_thread():
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except ValueError:
        pass  # Signal handling not supported in this context

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.wrappers import VectorizedBlockBlastEnv
from agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from utils.device import get_device, set_seed
from utils.logger import Logger, TensorBoardLogger, MetricsTracker


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_directories(config: Dict[str, Any]) -> Dict[str, Path]:
    """Create necessary directories."""
    paths = config.get('paths', {})
    
    dirs = {
        'checkpoint': Path(paths.get('checkpoint_dir', 'checkpoints')),
        'log': Path(paths.get('log_dir', 'logs')),
        'results': Path(paths.get('results_dir', 'results')),
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def train(
    config: Dict[str, Any],
    resume_path: Optional[str] = None,
    seed: int = 42,
    progress_callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> None:
    """
    Main training function.

    Args:
        config: Training configuration
        resume_path: Path to checkpoint to resume from
        seed: Random seed
        progress_callback: Optional callback function that receives metrics dict
                          and returns False to stop training
    """
    # Setup
    set_seed(seed)
    device = get_device()
    dirs = create_directories(config)
    
    # Get config sections
    env_config = config.get('environment', {})
    ppo_config = config.get('ppo', {})
    train_config = config.get('training', {})
    reward_config = config.get('rewards', {})
    log_config = config.get('logging', {})
    
    # Create experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"ppo_{timestamp}"
    
    # Create loggers
    logger = Logger(str(dirs['log']), experiment_name)
    tb_logger = TensorBoardLogger(str(dirs['log']), experiment_name)
    metrics_tracker = MetricsTracker(window_size=100)
    
    # Create environments
    num_envs = train_config.get('num_envs', 64)
    vec_env = VectorizedBlockBlastEnv(
        num_envs=num_envs,
        seed=seed,
        reward_config=reward_config,
    )
    
    print(f"Created {num_envs} parallel environments")
    
    # Create agent
    agent_config = PPOConfig(
        learning_rate=ppo_config.get('learning_rate', 3e-4),
        gamma=ppo_config.get('gamma', 0.99),
        gae_lambda=ppo_config.get('gae_lambda', 0.95),
        clip_epsilon=ppo_config.get('clip_epsilon', 0.2),
        entropy_coef=ppo_config.get('entropy_coef', 0.01),
        value_coef=ppo_config.get('value_coef', 0.5),
        max_grad_norm=ppo_config.get('max_grad_norm', 0.5),
        num_epochs=ppo_config.get('num_epochs', 10),
        batch_size=train_config.get('batch_size', 2048),
    )
    
    agent = PPOAgent(agent_config, device)
    agent.train()
    
    print(f"Created PPO agent with {sum(p.numel() for p in agent.network.parameters()):,} parameters")
    
    # Resume from checkpoint if provided
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        agent.load(resume_path)
        # Try to extract step from filename
        try:
            start_step = int(Path(resume_path).stem.split('_')[-1])
        except:
            pass
    
    # Create rollout buffer
    rollout_steps = train_config.get('rollout_steps', 128)
    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        num_envs=num_envs,
        device=device,
    )
    
    # Training parameters
    total_timesteps = train_config.get('total_timesteps', 50_000_000)
    log_interval = log_config.get('log_interval', 100)
    save_interval = log_config.get('save_interval', 1000)
    eval_interval = log_config.get('eval_interval', 500)
    
    # Initialize
    obs, _ = vec_env.reset()
    global_step = start_step
    num_updates = 0
    
    best_score = 0
    episode_scores = []
    episode_lengths = []
    
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print(f"  Rollout steps: {rollout_steps}")
    print(f"  Batch size: {agent_config.batch_size}")
    print(f"  Learning rate: {agent_config.learning_rate}")
    print("-" * 60, flush=True)
    
    start_time = time.time()
    
    try:
        while global_step < total_timesteps:
            # Collect rollouts
            buffer.reset()
            
            for step in range(rollout_steps):
                global_step += num_envs
                
                # Select actions
                actions, log_probs, values = agent.select_actions(obs)
                
                # Step environments
                next_obs, rewards, terminated, truncated, infos = vec_env.step(actions)
                dones = np.logical_or(terminated, truncated)
                
                # Store in buffer
                buffer.add(
                    board=obs['board'],
                    pieces=obs['pieces'],
                    action_mask=obs['action_mask'],
                    action=actions,
                    log_prob=log_probs,
                    reward=rewards,
                    done=dones.astype(np.float32),
                    value=values,
                )
                
                # Track episode statistics
                for i, (term, info) in enumerate(zip(terminated, infos)):
                    if term:
                        episode_scores.append(info.get('final_score', info.get('score', 0)))
                        episode_lengths.append(info.get('moves', 0))
                        metrics_tracker.add('episode_score', episode_scores[-1])
                        metrics_tracker.add('episode_length', episode_lengths[-1])
                
                obs = next_obs
            
            # Compute bootstrap value
            last_values = agent.get_values(obs)
            
            # Update agent
            update_metrics = agent.update(buffer, last_values)
            num_updates += 1
            
            # Print progress every update for first 20 updates
            if num_updates <= 20:
                elapsed = time.time() - start_time
                fps = global_step / elapsed if elapsed > 0 else 0
                print(f"Update {num_updates}: step={global_step:,}, FPS={fps:.0f}, policy_loss={update_metrics['policy_loss']:.4f}", flush=True)
            
            # Logging
            if num_updates % log_interval == 0 or num_updates <= 10:
                # Calculate statistics
                elapsed = time.time() - start_time
                fps = global_step / elapsed
                
                avg_score = metrics_tracker.get_mean('episode_score')
                max_score = metrics_tracker.get_max('episode_score')
                avg_length = metrics_tracker.get_mean('episode_length')
                
                if avg_score > best_score:
                    best_score = avg_score
                    # Save best model
                    best_path = dirs['checkpoint'] / "best.pt"
                    agent.save(str(best_path))
                
                # Log to console
                metrics = {
                    'step': global_step,
                    'fps': fps,
                    'avg_score': avg_score,
                    'max_score': max_score,
                    'best_score': best_score,
                    'avg_length': avg_length,
                    **update_metrics,
                }
                
                logger.log(metrics, global_step)
                logger.print_metrics(metrics)
                sys.stdout.flush()  # Force flush output
                
                # Log to TensorBoard
                tb_logger.log_metrics({
                    'performance/avg_score': avg_score,
                    'performance/max_score': max_score,
                    'performance/best_score': best_score,
                    'performance/avg_length': avg_length,
                    'performance/fps': fps,
                    'training/policy_loss': update_metrics['policy_loss'],
                    'training/value_loss': update_metrics['value_loss'],
                    'training/entropy': update_metrics['entropy'],
                    'training/approx_kl': update_metrics['approx_kl'],
                    'training/clip_fraction': update_metrics['clip_fraction'],
                }, global_step)

                # Call progress callback if provided
                if progress_callback is not None:
                    callback_metrics = {
                        'total_steps': global_step,
                        'mean_score': avg_score,
                        'best_score': best_score,
                        'episodes': len(episode_scores),
                        'fps': fps,
                    }
                    if not progress_callback(callback_metrics):
                        print("\nTraining stopped by callback")
                        break
            
            # Save checkpoint
            if num_updates % save_interval == 0:
                checkpoint_path = dirs['checkpoint'] / f"checkpoint_{global_step}.pt"
                agent.save(str(checkpoint_path))
                
                # Also save as latest
                latest_path = dirs['checkpoint'] / "latest.pt"
                agent.save(str(latest_path))
                
                print(f"Saved checkpoint to {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        # Final save
        final_path = dirs['checkpoint'] / "final.pt"
        agent.save(str(final_path))
        print(f"Saved final model to {final_path}")
        
        # Save summary
        logger.save_summary()
        tb_logger.close()
        vec_env.close()
        
        # Print final statistics
        elapsed = time.time() - start_time
        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"  Total steps: {global_step:,}")
        print(f"  Total time: {elapsed / 3600:.2f} hours")
        print(f"  Final FPS: {global_step / elapsed:.0f}")
        print(f"  Best average score: {best_score:.1f}")
        print(f"  Total episodes: {len(episode_scores)}")
        if episode_scores:
            print(f"  Max episode score: {max(episode_scores)}")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Block Blast AI")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/default.yaml",
        help="Path to configuration file"
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
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
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
                'line_clear_base': 100,
                'block_placed': 1,
                'game_over_penalty': -500,
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
    
    # Start training
    train(config, args.resume, args.seed)


if __name__ == "__main__":
    main()
