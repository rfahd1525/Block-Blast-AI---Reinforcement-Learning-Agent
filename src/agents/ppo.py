"""
Proximal Policy Optimization (PPO) Agent for Block Blast.

This module implements PPO with:
- Generalized Advantage Estimation (GAE)
- Clipped objective function
- Entropy bonus for exploration
- Action masking for invalid actions
"""
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.base import BaseAgent
from models.network import BlockBlastNetwork


@dataclass
class PPOConfig:
    """Configuration for PPO agent."""
    # Learning rate
    learning_rate: float = 3e-4
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training settings
    num_epochs: int = 10
    batch_size: int = 64
    
    # Network settings
    conv_channels: Tuple[int, ...] = (64, 128, 128)
    fc_hidden: Tuple[int, ...] = (512, 256)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef,
            'max_grad_norm': self.max_grad_norm,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'conv_channels': self.conv_channels,
            'fc_hidden': self.fc_hidden,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PPOConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class RolloutBuffer:
    """
    Buffer for storing rollout data.
    """
    
    def __init__(
        self,
        buffer_size: int,
        num_envs: int,
        board_size: int = 8,
        num_pieces: int = 3,
        action_space_size: int = 192,
        device: torch.device = torch.device("cpu"),
    ):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Number of steps per environment
            num_envs: Number of parallel environments
            board_size: Size of the game board
            num_pieces: Number of pieces per turn
            action_space_size: Total number of actions
            device: Torch device
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        
        # Allocate buffers
        self.boards = np.zeros((buffer_size, num_envs, board_size, board_size), dtype=np.float32)
        self.pieces = np.zeros((buffer_size, num_envs, num_pieces, board_size, board_size), dtype=np.float32)
        self.action_masks = np.zeros((buffer_size, num_envs, action_space_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.values = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        # Computed during finalization
        self.advantages = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_envs), dtype=np.float32)
        
        self.ptr = 0
        self.full = False
    
    def add(
        self,
        board: np.ndarray,
        pieces: np.ndarray,
        action_mask: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Add a step to the buffer."""
        self.boards[self.ptr] = board
        self.pieces[self.ptr] = pieces
        self.action_masks[self.ptr] = action_mask
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        
        self.ptr += 1
        if self.ptr >= self.buffer_size:
            self.full = True
    
    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """
        Compute returns and GAE advantages.
        
        Args:
            last_values: Value estimates for the last state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        last_gae_lam = 0
        
        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_values = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam
        
        self.returns = self.advantages + self.values
    
    def get_samples(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get random samples from the buffer.
        
        Args:
            batch_size: Number of samples per batch
            
        Yields:
            Batches of (boards, pieces, action_masks, actions, old_log_probs, advantages, returns)
        """
        # Flatten across envs and timesteps
        total_size = self.buffer_size * self.num_envs
        
        boards = self.boards.reshape(total_size, *self.boards.shape[2:])
        pieces = self.pieces.reshape(total_size, *self.pieces.shape[2:])
        action_masks = self.action_masks.reshape(total_size, -1)
        actions = self.actions.reshape(total_size)
        log_probs = self.log_probs.reshape(total_size)
        advantages = self.advantages.reshape(total_size)
        returns = self.returns.reshape(total_size)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Generate random indices
        indices = np.random.permutation(total_size)
        
        for start in range(0, total_size, batch_size):
            end = min(start + batch_size, total_size)
            batch_indices = indices[start:end]
            
            yield (
                torch.from_numpy(boards[batch_indices]).to(self.device),
                torch.from_numpy(pieces[batch_indices]).to(self.device),
                torch.from_numpy(action_masks[batch_indices]).to(self.device),
                torch.from_numpy(actions[batch_indices]).to(self.device),
                torch.from_numpy(log_probs[batch_indices]).to(self.device),
                torch.from_numpy(advantages[batch_indices]).to(self.device),
                torch.from_numpy(returns[batch_indices]).to(self.device),
            )
    
    def reset(self) -> None:
        """Reset the buffer."""
        self.ptr = 0
        self.full = False


class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization agent for Block Blast.
    """
    
    def __init__(
        self,
        config: Optional[PPOConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize PPO agent.
        
        Args:
            config: PPO configuration
            device: Torch device
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        super().__init__(device)
        
        self.config = config or PPOConfig()
        
        # Create network
        self.network = BlockBlastNetwork(
            conv_channels=self.config.conv_channels,
            fc_hidden=self.config.fc_hidden,
        ).to(device)
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
        )
        
        # Learning rate scheduler
        self.scheduler = None
    
    def select_action(
        self,
        observation: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        Select an action given an observation.
        
        Args:
            observation: Environment observation with 'board', 'pieces', 'action_mask'
            deterministic: Whether to select action deterministically
            
        Returns:
            Tuple of (action, info_dict)
        """
        with torch.no_grad():
            board = torch.from_numpy(observation['board']).unsqueeze(0).to(self.device)
            pieces = torch.from_numpy(observation['pieces']).unsqueeze(0).to(self.device)
            action_mask = torch.from_numpy(observation['action_mask']).unsqueeze(0).float().to(self.device)
            
            action, log_prob, entropy, value = self.network.get_action_and_value(
                board, pieces, action_mask, deterministic=deterministic
            )
            
            return action.item(), {
                'log_prob': log_prob.item(),
                'entropy': entropy.item(),
                'value': value.item(),
            }
    
    def select_actions(
        self,
        observations: Dict[str, np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Select actions for multiple environments.
        
        Args:
            observations: Batched observations
            deterministic: Whether to select actions deterministically
            
        Returns:
            Tuple of (actions, log_probs, values)
        """
        with torch.no_grad():
            board = torch.from_numpy(observations['board']).to(self.device)
            pieces = torch.from_numpy(observations['pieces']).to(self.device)
            action_mask = torch.from_numpy(observations['action_mask']).float().to(self.device)
            
            action, log_prob, _, value = self.network.get_action_and_value(
                board, pieces, action_mask, deterministic=deterministic
            )
            
            return (
                action.cpu().numpy(),
                log_prob.cpu().numpy(),
                value.cpu().numpy(),
            )
    
    def get_values(self, observations: Dict[str, np.ndarray]) -> np.ndarray:
        """Get value estimates for observations."""
        with torch.no_grad():
            board = torch.from_numpy(observations['board']).to(self.device)
            pieces = torch.from_numpy(observations['pieces']).to(self.device)
            
            values = self.network.get_value(board, pieces)
            return values.cpu().numpy()
    
    def update(
        self, 
        buffer: RolloutBuffer,
        last_values: np.ndarray,
    ) -> Dict[str, float]:
        """
        Update the agent using collected rollout data.
        
        Args:
            buffer: Rollout buffer with collected experience
            last_values: Value estimates for final states
            
        Returns:
            Dictionary of training metrics
        """
        # Compute returns and advantages
        buffer.compute_returns_and_advantages(
            last_values,
            self.config.gamma,
            self.config.gae_lambda,
        )
        
        # Training metrics
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_loss = 0
        total_approx_kl = 0
        total_clip_fraction = 0
        num_updates = 0
        
        # Multiple epochs of PPO updates
        for epoch in range(self.config.num_epochs):
            for batch in buffer.get_samples(self.config.batch_size):
                boards, pieces, action_masks, actions, old_log_probs, advantages, returns = batch
                
                # Get current policy outputs
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    boards, pieces, action_masks, action=actions
                )
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon, 
                    1 + self.config.clip_epsilon
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns)
                
                # Entropy loss (negative because we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    policy_loss 
                    + self.config.value_coef * value_loss 
                    + self.config.entropy_coef * entropy_loss
                )
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.network.parameters(), 
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy.mean().item()
                total_loss += loss.item()
                total_approx_kl += approx_kl.item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1
        
        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy_loss / num_updates,
            'total_loss': total_loss / num_updates,
            'approx_kl': total_approx_kl / num_updates,
            'clip_fraction': total_clip_fraction / num_updates,
        }
    
    def save(self, path: str) -> None:
        """Save agent to disk."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.to_dict(),
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent from disk."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'config' in checkpoint:
            self.config = PPOConfig.from_dict(checkpoint['config'])
    
    def train(self) -> None:
        """Set to training mode."""
        super().train()
        self.network.train()
    
    def eval(self) -> None:
        """Set to evaluation mode."""
        super().eval()
        self.network.eval()


if __name__ == "__main__":
    # Test PPO agent
    print("Testing PPO Agent...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create agent
    config = PPOConfig(learning_rate=1e-4)
    agent = PPOAgent(config, device)
    
    print(f"Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")
    
    # Test action selection
    obs = {
        'board': np.random.rand(8, 8).astype(np.float32),
        'pieces': np.random.rand(3, 8, 8).astype(np.float32),
        'action_mask': np.ones(192, dtype=np.float32),
    }
    
    action, info = agent.select_action(obs)
    print(f"Selected action: {action}")
    print(f"Info: {info}")
    
    # Test batched action selection
    batch_obs = {
        'board': np.random.rand(4, 8, 8).astype(np.float32),
        'pieces': np.random.rand(4, 3, 8, 8).astype(np.float32),
        'action_mask': np.ones((4, 192), dtype=np.float32),
    }
    
    actions, log_probs, values = agent.select_actions(batch_obs)
    print(f"Batched actions: {actions}")
    
    # Test rollout buffer
    print("\nTesting RolloutBuffer...")
    buffer = RolloutBuffer(
        buffer_size=128,
        num_envs=4,
        device=device,
    )
    
    for _ in range(128):
        buffer.add(
            board=np.random.rand(4, 8, 8).astype(np.float32),
            pieces=np.random.rand(4, 3, 8, 8).astype(np.float32),
            action_mask=np.ones((4, 192), dtype=np.float32),
            action=np.random.randint(0, 192, size=4),
            log_prob=np.random.rand(4).astype(np.float32),
            reward=np.random.rand(4).astype(np.float32),
            done=np.zeros(4, dtype=np.float32),
            value=np.random.rand(4).astype(np.float32),
        )
    
    # Test update
    last_values = np.random.rand(4).astype(np.float32)
    metrics = agent.update(buffer, last_values)
    print(f"Update metrics: {metrics}")
    
    # Test save/load
    agent.save("test_agent.pt")
    agent.load("test_agent.pt")
    print("Save/load successful!")
    
    import os
    os.remove("test_agent.pt")
