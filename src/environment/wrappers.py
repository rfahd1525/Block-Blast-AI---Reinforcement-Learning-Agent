"""
Environment wrappers for Block Blast.

Provides vectorized and other useful environment wrappers.
"""
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import gymnasium as gym

from .block_blast_env import BlockBlastEnv


class VectorizedBlockBlastEnv:
    """
    Vectorized environment running multiple Block Blast games in parallel.
    
    This allows for efficient batch collection of experiences for training.
    """
    
    def __init__(
        self,
        num_envs: int,
        seed: Optional[int] = None,
        reward_config: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize vectorized environment.
        
        Args:
            num_envs: Number of parallel environments
            seed: Base random seed (each env gets seed + i)
            reward_config: Custom reward configuration
        """
        self.num_envs = num_envs
        self.reward_config = reward_config
        
        # Create environments
        self.envs = []
        for i in range(num_envs):
            env_seed = seed + i if seed is not None else None
            env = BlockBlastEnv(seed=env_seed, reward_config=reward_config)
            self.envs.append(env)
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self.single_action_space = self.action_space
        
        # Track which envs are done
        self._dones = np.zeros(num_envs, dtype=bool)
    
    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]]]:
        """
        Reset all environments.
        
        Returns:
            Tuple of (stacked observations, list of info dicts)
        """
        observations = []
        infos = []
        
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            observations.append(obs)
            infos.append(info)
        
        self._dones.fill(False)
        return self._stack_observations(observations), infos
    
    def step(
        self, actions: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Step all environments.
        
        Args:
            actions: Array of actions, one per environment
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
        """
        observations = []
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.zeros(self.num_envs, dtype=bool)
        truncated = np.zeros(self.num_envs, dtype=bool)
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, term, trunc, info = env.step(int(action))
            
            # Auto-reset on termination
            if term or trunc:
                # Store final info
                info['terminal_observation'] = obs
                info['final_score'] = info['score']
                # Reset environment
                obs, _ = env.reset()
            
            observations.append(obs)
            rewards[i] = reward
            terminated[i] = term
            truncated[i] = trunc
            infos.append(info)
        
        return (
            self._stack_observations(observations),
            rewards,
            terminated,
            truncated,
            infos,
        )
    
    def _stack_observations(
        self, observations: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        """Stack observations from all environments."""
        return {
            'board': np.stack([obs['board'] for obs in observations]),
            'pieces': np.stack([obs['pieces'] for obs in observations]),
            'action_mask': np.stack([obs['action_mask'] for obs in observations]),
        }
    
    def get_action_masks(self) -> np.ndarray:
        """Get action masks for all environments."""
        masks = [env.get_action_mask() for env in self.envs]
        return np.stack(masks)
    
    def sample_valid_actions(self) -> np.ndarray:
        """Sample valid actions for all environments."""
        actions = np.array([env.sample_valid_action() for env in self.envs])
        return actions
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()


class NormalizedRewardWrapper(gym.Wrapper):
    """
    Wrapper that normalizes rewards using running statistics.
    """
    
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        """
        Initialize reward normalization wrapper.
        
        Args:
            env: Environment to wrap
            gamma: Discount factor for return estimation
            epsilon: Small constant for numerical stability
        """
        super().__init__(env)
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Running statistics
        self.return_rms = RunningMeanStd()
        self.returns = 0.0
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update return estimate
        self.returns = self.returns * self.gamma + reward
        self.return_rms.update(np.array([self.returns]))
        
        # Normalize reward
        normalized_reward = reward / (np.sqrt(self.return_rms.var) + self.epsilon)
        
        if terminated or truncated:
            self.returns = 0.0
        
        info['raw_reward'] = reward
        return obs, normalized_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self.returns = 0.0
        return self.env.reset(**kwargs)


class RunningMeanStd:
    """
    Tracks running mean and standard deviation.
    """
    
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """Initialize running statistics."""
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update statistics with new batch of data."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        """Update from batch statistics."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class FrameStackWrapper(gym.Wrapper):
    """
    Wrapper that stacks observations over multiple frames.
    Useful for capturing temporal information.
    """
    
    def __init__(self, env: gym.Env, num_frames: int = 4):
        """
        Initialize frame stacking wrapper.
        
        Args:
            env: Environment to wrap
            num_frames: Number of frames to stack
        """
        super().__init__(env)
        self.num_frames = num_frames
        self.frames = None
        
        # Update observation space
        old_space = env.observation_space['board']
        new_shape = (num_frames,) + old_space.shape
        self.observation_space = gym.spaces.Dict({
            'board': gym.spaces.Box(
                low=0.0, high=1.0,
                shape=new_shape,
                dtype=np.float32
            ),
            'pieces': env.observation_space['pieces'],
            'action_mask': env.observation_space['action_mask'],
        })
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Initialize frame stack with copies of first observation
        self.frames = [obs['board'].copy() for _ in range(self.num_frames)]
        
        stacked_obs = {
            'board': np.stack(self.frames, axis=0),
            'pieces': obs['pieces'],
            'action_mask': obs['action_mask'],
        }
        return stacked_obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Update frame stack
        self.frames.pop(0)
        self.frames.append(obs['board'].copy())
        
        stacked_obs = {
            'board': np.stack(self.frames, axis=0),
            'pieces': obs['pieces'],
            'action_mask': obs['action_mask'],
        }
        return stacked_obs, reward, terminated, truncated, info


def make_env(
    seed: Optional[int] = None,
    reward_config: Optional[Dict[str, float]] = None,
    normalize_reward: bool = False,
    frame_stack: int = 1,
) -> gym.Env:
    """
    Factory function to create a Block Blast environment with wrappers.
    
    Args:
        seed: Random seed
        reward_config: Custom reward configuration
        normalize_reward: Whether to normalize rewards
        frame_stack: Number of frames to stack (1 = no stacking)
        
    Returns:
        Configured environment
    """
    env = BlockBlastEnv(seed=seed, reward_config=reward_config)
    
    if frame_stack > 1:
        env = FrameStackWrapper(env, num_frames=frame_stack)
    
    if normalize_reward:
        env = NormalizedRewardWrapper(env)
    
    return env


def make_vec_env(
    num_envs: int,
    seed: Optional[int] = None,
    reward_config: Optional[Dict[str, float]] = None,
) -> VectorizedBlockBlastEnv:
    """
    Factory function to create vectorized Block Blast environments.
    
    Args:
        num_envs: Number of parallel environments
        seed: Base random seed
        reward_config: Custom reward configuration
        
    Returns:
        Vectorized environment
    """
    return VectorizedBlockBlastEnv(
        num_envs=num_envs,
        seed=seed,
        reward_config=reward_config,
    )


if __name__ == "__main__":
    # Test vectorized environment
    print("Testing VectorizedBlockBlastEnv...")
    vec_env = VectorizedBlockBlastEnv(num_envs=4, seed=42)
    
    obs, infos = vec_env.reset()
    print(f"Observation shapes:")
    print(f"  board: {obs['board'].shape}")  # (4, 8, 8)
    print(f"  pieces: {obs['pieces'].shape}")  # (4, 3, 8, 8)
    print(f"  action_mask: {obs['action_mask'].shape}")  # (4, 192)
    
    # Run a few steps
    total_rewards = np.zeros(4)
    for step in range(100):
        actions = vec_env.sample_valid_actions()
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        total_rewards += rewards
        
        for i, (term, info) in enumerate(zip(terminated, infos)):
            if term:
                print(f"Env {i} finished with score {info.get('final_score', 'N/A')}")
    
    print(f"\nTotal rewards: {total_rewards}")
    vec_env.close()
