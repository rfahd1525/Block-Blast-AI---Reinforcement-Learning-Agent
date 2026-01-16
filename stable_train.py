"""Stable extended training script without tqdm."""
import sys
import os
sys.path.insert(0, 'src')

import numpy as np
import torch
import time

from environment.wrappers import VectorizedBlockBlastEnv
from agents.ppo import PPOAgent, PPOConfig, RolloutBuffer
from utils.device import get_device

def stable_train():
    """Run stable extended training loop without tqdm."""
    device = get_device()
    
    # Create environments
    num_envs = 32
    vec_env = VectorizedBlockBlastEnv(num_envs=num_envs)
    print(f"Created {num_envs} environments")
    
    # Create agent
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        num_epochs=4,
        batch_size=512,
    )
    
    agent = PPOAgent(config, device)
    agent.train()
    print(f"Created agent with {sum(p.numel() for p in agent.network.parameters()):,} parameters")
    
    # Try to load best model if it exists
    checkpoint_path = 'checkpoints/quick_trained.pt'
    if os.path.exists(checkpoint_path):
        try:
            agent.load(checkpoint_path)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
    
    # Create buffer
    rollout_steps = 64
    buffer = RolloutBuffer(
        buffer_size=rollout_steps,
        num_envs=num_envs,
        device=device,
    )
    
    # Training settings
    total_timesteps = 2_000_000
    log_every = 10  # Updates
    
    # Initialize
    obs, _ = vec_env.reset()
    global_step = 0
    num_updates = 0
    
    episode_scores = []
    episode_lengths = []
    best_avg = 0
    
    print(f"\nTraining for {total_timesteps:,} timesteps...")
    start_time = time.time()
    last_log_time = start_time
    last_log_step = 0
    
    try:
        while global_step < total_timesteps:
            # Collect rollout
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
                
                # Track episode stats
                for term, info in zip(terminated, infos):
                    if term:
                        episode_scores.append(info.get('final_score', info.get('score', 0)))
                        episode_lengths.append(info.get('moves', 0))
                
                obs = next_obs
            
            # Get bootstrap value
            last_values = agent.get_values(obs)
            
            # Update agent
            metrics = agent.update(buffer, last_values)
            num_updates += 1
            
            # Log progress
            if num_updates % log_every == 0 and len(episode_scores) > 0:
                current_time = time.time()
                elapsed = current_time - start_time
                step_delta = global_step - last_log_step
                time_delta = current_time - last_log_time
                fps = step_delta / time_delta if time_delta > 0 else 0
                
                recent_scores = episode_scores[-100:]
                avg_score = np.mean(recent_scores)
                max_score = np.max(episode_scores)
                
                if avg_score > best_avg:
                    best_avg = avg_score
                
                pct_complete = 100.0 * global_step / total_timesteps
                est_total_time = elapsed / pct_complete * 100 if pct_complete > 0 else 0
                est_remaining = est_total_time - elapsed
                
                print(f"Step {global_step:>8,}/{total_timesteps:,} ({pct_complete:5.1f}%) | "
                      f"Avg: {avg_score:7.0f} | Max: {max_score:7.0f} | Best: {best_avg:7.0f} | "
                      f"FPS: {fps:7.0f} | ETA: {est_remaining/60:6.1f}m")
                
                last_log_time = current_time
                last_log_step = global_step
                
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
    except Exception as e:
        print(f"\n\nTraining error: {e}")
        import traceback
        traceback.print_exc()
    
    # Final stats
    elapsed = time.time() - start_time
    print(f"\n{'='*100}")
    print(f"Training Complete!")
    print(f"  Total steps: {global_step:,} / {total_timesteps:,}")
    print(f"  Total time: {elapsed/60:.1f} min")
    print(f"  FPS: {global_step/elapsed:.0f}")
    print(f"  Episodes: {len(episode_scores)}")
    if episode_scores:
        print(f"  Final avg score: {np.mean(episode_scores[-100:]):.0f}")
        print(f"  Max score: {np.max(episode_scores):.0f}")
        print(f"  Best avg score: {best_avg:.0f}")
    print(f"{'='*100}")
    
    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    agent.save('checkpoints/extended_trained.pt')
    print(f"Saved model to checkpoints/extended_trained.pt")

if __name__ == "__main__":
    stable_train()
