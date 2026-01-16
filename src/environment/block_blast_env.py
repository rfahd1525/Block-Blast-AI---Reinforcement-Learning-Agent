"""
Block Blast Gymnasium Environment.

This module provides a Gymnasium-compatible environment for training
reinforcement learning agents on Block Blast.
"""
from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.engine import GameEngine, GameStatus, MoveResult
from game.pieces import NUM_PIECES, PIECE_LIST


class BlockBlastEnv(gym.Env):
    """
    Gymnasium environment for Block Blast.
    
    Observation Space:
        Dictionary with:
        - 'board': (8, 8) float32 array, 0=empty, 1=filled
        - 'pieces': (3, 8, 8) float32 array, piece masks
        - 'action_mask': (192,) bool array, valid actions
    
    Action Space:
        Discrete(192) - representing (piece_idx, row, col) as flat index
        Action = piece_idx * 64 + row * 8 + col
        piece_idx: 0-2, row: 0-7, col: 0-7
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    BOARD_SIZE = 8
    NUM_PIECES_PER_TURN = 3
    ACTION_SPACE_SIZE = NUM_PIECES_PER_TURN * BOARD_SIZE * BOARD_SIZE  # 3 * 8 * 8 = 192
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        reward_config: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Block Blast environment.
        
        Args:
            render_mode: 'human' for console output, 'ansi' for string return
            reward_config: Custom reward configuration
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.seed_value = seed
        
        # Reward configuration
        # Rewards scaled to roughly [-1, 1] range for stable training
        self.reward_config = {
            'line_clear_base': 1.0,
            'block_placed': 0.01,
            'game_over_penalty': -1.0,
            'hole_penalty': -0.05,
            'center_bonus': 0.02,
            'combo_multiplier_bonus': 0.5,
            'survival_bonus': 0.001,
        }
        if reward_config:
            self.reward_config.update(reward_config)
        
        # Initialize game engine
        self.engine = GameEngine(board_size=self.BOARD_SIZE, seed=seed)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'board': spaces.Box(
                low=0.0, high=1.0, 
                shape=(self.BOARD_SIZE, self.BOARD_SIZE), 
                dtype=np.float32
            ),
            'pieces': spaces.Box(
                low=0.0, high=1.0,
                shape=(self.NUM_PIECES_PER_TURN, self.BOARD_SIZE, self.BOARD_SIZE),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.ACTION_SPACE_SIZE,),
                dtype=np.int8
            ),
        })
        
        # Define action space
        self.action_space = spaces.Discrete(self.ACTION_SPACE_SIZE)
        
        # Track previous state for reward shaping
        self._prev_holes = 0
        self._prev_center_openness = 1.0
    
    def _action_to_move(self, action: int) -> Tuple[int, int, int]:
        """
        Convert flat action index to (piece_idx, row, col).
        
        Args:
            action: Flat action index (0-191)
            
        Returns:
            Tuple of (piece_idx, row, col)
        """
        piece_idx = action // (self.BOARD_SIZE * self.BOARD_SIZE)
        remainder = action % (self.BOARD_SIZE * self.BOARD_SIZE)
        row = remainder // self.BOARD_SIZE
        col = remainder % self.BOARD_SIZE
        return piece_idx, row, col
    
    def _move_to_action(self, piece_idx: int, row: int, col: int) -> int:
        """
        Convert (piece_idx, row, col) to flat action index.
        
        Args:
            piece_idx: Piece index (0-2)
            row: Board row (0-7)
            col: Board column (0-7)
            
        Returns:
            Flat action index
        """
        return piece_idx * self.BOARD_SIZE * self.BOARD_SIZE + row * self.BOARD_SIZE + col
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get the current observation."""
        obs = self.engine.get_observation()
        
        # Flatten action mask
        action_mask_3d = obs['action_mask']
        action_mask = action_mask_3d.flatten().astype(np.int8)
        
        return {
            'board': obs['board'],
            'pieces': obs['pieces'],
            'action_mask': action_mask,
        }
    
    def _calculate_reward(self, result: MoveResult) -> float:
        """
        Calculate reward for a move.
        
        Args:
            result: The result of the move
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base reward for placing blocks
        reward += result.blocks_placed * self.reward_config['block_placed']
        
        # Survival bonus
        reward += self.reward_config['survival_bonus']
        
        # Reward for clearing lines
        if result.lines_cleared > 0:
            line_reward = result.lines_cleared * self.reward_config['line_clear_base']
            line_reward *= result.combo_multiplier
            reward += line_reward
            
            # Extra bonus for combo multiplier
            if result.combo_multiplier > 1:
                reward += (result.combo_multiplier - 1) * self.reward_config['combo_multiplier_bonus']
        
        # Penalty for game over
        if result.game_over:
            reward += self.reward_config['game_over_penalty']
        
        # Reward shaping based on board state
        current_holes = self.engine.board.count_holes()
        hole_delta = current_holes - self._prev_holes
        if hole_delta > 0:
            reward += hole_delta * self.reward_config['hole_penalty']
        self._prev_holes = current_holes
        
        # Center openness bonus
        current_center = self.engine.board.get_center_openness()
        if current_center >= self._prev_center_openness:
            reward += self.reward_config['center_bonus'] * 0.1  # Small bonus for maintaining center
        self._prev_center_openness = current_center
        
        return reward
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed_value = seed
        
        self.engine.reset(seed=self.seed_value)
        self._prev_holes = 0
        self._prev_center_openness = 1.0
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Flat action index (0-191)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to move
        piece_idx, row, col = self._action_to_move(action)
        
        # Check if action is valid
        if not self.engine.can_place_piece(piece_idx, row, col):
            # Invalid action - give penalty and don't change state
            observation = self._get_observation()
            info = self._get_info()
            info['invalid_action'] = True
            return observation, -10.0, False, False, info
        
        # Execute the move
        result = self.engine.make_move(piece_idx, row, col)
        
        # Calculate reward
        reward = self._calculate_reward(result)
        
        # Check termination
        terminated = result.game_over
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        info = self._get_info(result)
        
        if self.render_mode == "human":
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def _get_info(self, result: Optional[MoveResult] = None) -> Dict[str, Any]:
        """Get info dictionary."""
        stats = self.engine.get_statistics()
        info = {
            'score': stats['score'],
            'moves': stats['moves_made'],
            'lines_cleared': stats['total_lines_cleared'],
            'max_combo': stats['max_combo'],
            'blocks_placed': stats['total_blocks_placed'],
            'board_fill': stats['board_fill_ratio'],
            'holes': stats['holes'],
            'invalid_action': False,
        }
        
        if result:
            info['last_move'] = {
                'blocks_placed': result.blocks_placed,
                'lines_cleared': result.lines_cleared,
                'combo_multiplier': result.combo_multiplier,
                'score_gained': result.score_gained,
            }
        
        return info
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get the current action mask.
        
        Returns:
            Boolean array of shape (192,) where True = valid action
        """
        return self._get_observation()['action_mask'].astype(bool)
    
    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode == "ansi":
            return str(self.engine)
        elif self.render_mode == "human":
            print("\033[2J\033[H")  # Clear screen
            print(self.engine)
            return None
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        pass
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid action indices."""
        mask = self.get_action_mask()
        return np.where(mask)[0].tolist()
    
    def sample_valid_action(self) -> int:
        """Sample a random valid action."""
        valid_actions = self.get_valid_actions()
        if not valid_actions:
            return 0  # No valid actions (game should be over)
        return np.random.choice(valid_actions)


class BlockBlastEnvFlat(BlockBlastEnv):
    """
    Block Blast environment with flattened observation for simpler networks.
    
    Observation is a single flat array containing:
    - Board (64 values)
    - Piece encodings (3 * 37 one-hot = 111 values)
    - Pieces used (3 values)
    Total: 190 values
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Redefine observation space
        obs_size = (
            self.BOARD_SIZE * self.BOARD_SIZE +  # Board: 64
            self.NUM_PIECES_PER_TURN * NUM_PIECES +  # Piece one-hots: 3*37=111
            self.NUM_PIECES_PER_TURN  # Pieces used: 3
        )
        
        self.observation_space = spaces.Dict({
            'obs': spaces.Box(
                low=0.0, high=1.0,
                shape=(obs_size,),
                dtype=np.float32
            ),
            'action_mask': spaces.Box(
                low=0, high=1,
                shape=(self.ACTION_SPACE_SIZE,),
                dtype=np.int8
            ),
        })
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get flattened observation."""
        engine_obs = self.engine.get_observation()
        
        # Flatten board
        board_flat = engine_obs['board'].flatten()
        
        # Create piece one-hot encodings
        piece_encodings = []
        for i, piece in enumerate(self.engine.current_pieces):
            one_hot = np.zeros(NUM_PIECES, dtype=np.float32)
            if not self.engine.pieces_used[i]:
                piece_idx = PIECE_LIST.index(piece)
                one_hot[piece_idx] = 1.0
            piece_encodings.append(one_hot)
        pieces_flat = np.concatenate(piece_encodings)
        
        # Pieces used
        used_flat = np.array(self.engine.pieces_used, dtype=np.float32)
        
        # Combine
        obs = np.concatenate([board_flat, pieces_flat, used_flat])
        
        # Action mask
        action_mask = engine_obs['action_mask'].flatten().astype(np.int8)
        
        return {
            'obs': obs,
            'action_mask': action_mask,
        }


# Register environments with Gymnasium
gym.register(
    id='BlockBlast-v0',
    entry_point='environment.block_blast_env:BlockBlastEnv',
    max_episode_steps=10000,
)

gym.register(
    id='BlockBlast-Flat-v0',
    entry_point='environment.block_blast_env:BlockBlastEnvFlat',
    max_episode_steps=10000,
)


if __name__ == "__main__":
    # Test the environment
    print("Testing BlockBlastEnv...")
    env = BlockBlastEnv(render_mode="ansi")
    
    obs, info = env.reset(seed=42)
    print(f"Initial observation shapes:")
    print(f"  board: {obs['board'].shape}")
    print(f"  pieces: {obs['pieces'].shape}")
    print(f"  action_mask: {obs['action_mask'].shape}")
    print(f"  Valid actions: {len(env.get_valid_actions())}")
    
    print("\nPlaying random game...")
    total_reward = 0
    done = False
    steps = 0
    
    while not done and steps < 1000:
        action = env.sample_valid_action()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        steps += 1
        
        if info.get('last_move', {}).get('lines_cleared', 0) > 0:
            print(f"Step {steps}: Cleared {info['last_move']['lines_cleared']} lines! "
                  f"Reward: {reward:.1f}")
    
    print(f"\nGame over after {steps} steps")
    print(f"Final score: {info['score']}")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Lines cleared: {info['lines_cleared']}")
    print(f"Max combo: {info['max_combo']}")
    
    env.close()
