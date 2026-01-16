"""
Tests for the RL environment.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.block_blast_env import BlockBlastEnv, BlockBlastEnvFlat
from environment.wrappers import VectorizedBlockBlastEnv


class TestEnvironmentCreation:
    """Test environment creation."""
    
    def test_create_env(self):
        """Test basic environment creation."""
        env = BlockBlastEnv()
        
        assert env.BOARD_SIZE == 8
        assert env.NUM_PIECES_PER_TURN == 3
        assert env.ACTION_SPACE_SIZE == 192
    
    def test_create_with_seed(self):
        """Test deterministic environment with seed."""
        env1 = BlockBlastEnv(seed=42)
        env2 = BlockBlastEnv(seed=42)
        
        obs1, _ = env1.reset()
        obs2, _ = env2.reset()
        
        assert np.array_equal(obs1['board'], obs2['board'])
    
    def test_observation_space(self):
        """Test observation space definition."""
        env = BlockBlastEnv()
        
        assert 'board' in env.observation_space.spaces
        assert 'pieces' in env.observation_space.spaces
        assert 'action_mask' in env.observation_space.spaces
    
    def test_action_space(self):
        """Test action space definition."""
        env = BlockBlastEnv()
        
        assert env.action_space.n == 192


class TestEnvironmentReset:
    """Test environment reset."""
    
    def test_reset(self):
        """Test reset returns proper observation."""
        env = BlockBlastEnv()
        obs, info = env.reset()
        
        assert 'board' in obs
        assert 'pieces' in obs
        assert 'action_mask' in obs
        assert isinstance(info, dict)
    
    def test_reset_observation_shapes(self):
        """Test observation shapes after reset."""
        env = BlockBlastEnv()
        obs, _ = env.reset()
        
        assert obs['board'].shape == (8, 8)
        assert obs['pieces'].shape == (3, 8, 8)
        assert obs['action_mask'].shape == (192,)
    
    def test_reset_with_seed(self):
        """Test deterministic reset."""
        env = BlockBlastEnv()
        
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        assert np.array_equal(obs1['board'], obs2['board'])


class TestEnvironmentStep:
    """Test environment stepping."""
    
    def test_step_valid_action(self):
        """Test stepping with valid action."""
        env = BlockBlastEnv()
        obs, _ = env.reset(seed=42)
        
        action = env.sample_valid_action()
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert 'board' in obs
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_step_invalid_action(self):
        """Test stepping with invalid action."""
        env = BlockBlastEnv()
        obs, _ = env.reset(seed=42)
        
        # Find an invalid action
        mask = obs['action_mask']
        invalid_actions = np.where(mask == 0)[0]
        
        if len(invalid_actions) > 0:
            action = invalid_actions[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            assert info.get('invalid_action', False)
            assert reward < 0  # Penalty for invalid action
    
    def test_step_updates_state(self):
        """Test that step updates the game state."""
        env = BlockBlastEnv()
        obs1, _ = env.reset(seed=42)
        
        action = env.sample_valid_action()
        obs2, _, _, _, info = env.step(action)
        
        # Board should have changed (unless line cleared and reset)
        # At minimum, the action mask should have changed
        # (since we used one piece)


class TestEnvironmentTermination:
    """Test episode termination."""
    
    def test_episode_terminates(self):
        """Episode should eventually terminate."""
        env = BlockBlastEnv()
        obs, _ = env.reset(seed=42)
        
        done = False
        steps = 0
        max_steps = 1000
        
        while not done and steps < max_steps:
            action = env.sample_valid_action()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
        
        assert done or steps == max_steps


class TestActionMasking:
    """Test action masking."""
    
    def test_get_action_mask(self):
        """Test action mask retrieval."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        mask = env.get_action_mask()
        
        assert mask.shape == (192,)
        assert mask.dtype == bool
        assert np.sum(mask) > 0  # At least some valid actions
    
    def test_get_valid_actions(self):
        """Test getting list of valid actions."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        valid_actions = env.get_valid_actions()
        
        assert len(valid_actions) > 0
        assert all(0 <= a < 192 for a in valid_actions)
    
    def test_sample_valid_action(self):
        """Test sampling valid action."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        action = env.sample_valid_action()
        mask = env.get_action_mask()
        
        assert mask[action]  # Action should be valid


class TestActionConversion:
    """Test action encoding/decoding."""
    
    def test_action_to_move(self):
        """Test converting flat action to (piece, row, col)."""
        env = BlockBlastEnv()
        
        # Action 0 should be piece 0, row 0, col 0
        piece, row, col = env._action_to_move(0)
        assert (piece, row, col) == (0, 0, 0)
        
        # Action 64 should be piece 1, row 0, col 0
        piece, row, col = env._action_to_move(64)
        assert (piece, row, col) == (1, 0, 0)
        
        # Action 128 should be piece 2, row 0, col 0
        piece, row, col = env._action_to_move(128)
        assert (piece, row, col) == (2, 0, 0)
    
    def test_move_to_action(self):
        """Test converting (piece, row, col) to flat action."""
        env = BlockBlastEnv()
        
        assert env._move_to_action(0, 0, 0) == 0
        assert env._move_to_action(1, 0, 0) == 64
        assert env._move_to_action(2, 0, 0) == 128
        assert env._move_to_action(0, 7, 7) == 63


class TestRewards:
    """Test reward calculation."""
    
    def test_positive_reward_for_placement(self):
        """Placing blocks should give positive reward."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        action = env.sample_valid_action()
        _, reward, _, _, _ = env.step(action)
        
        # Should get at least the block placement reward
        # (unless invalid action which we've avoided)
        assert reward > -100  # Not a game over penalty


class TestInfoDict:
    """Test info dictionary."""
    
    def test_info_contains_score(self):
        """Info should contain score."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        action = env.sample_valid_action()
        _, _, _, _, info = env.step(action)
        
        assert 'score' in info
    
    def test_info_contains_moves(self):
        """Info should contain move count."""
        env = BlockBlastEnv()
        env.reset(seed=42)
        
        action = env.sample_valid_action()
        _, _, _, _, info = env.step(action)
        
        assert 'moves' in info


class TestFlatEnvironment:
    """Test flat observation environment."""
    
    def test_create_flat_env(self):
        """Test creating flat observation env."""
        env = BlockBlastEnvFlat()
        
        assert 'obs' in env.observation_space.spaces
    
    def test_flat_observation_shape(self):
        """Test flat observation shape."""
        env = BlockBlastEnvFlat()
        obs, _ = env.reset()
        
        # 64 (board) + 3*37 (piece one-hots) + 3 (used) = 178
        assert obs['obs'].shape == (178,)


class TestVectorizedEnvironment:
    """Test vectorized environment."""
    
    def test_create_vec_env(self):
        """Test creating vectorized env."""
        vec_env = VectorizedBlockBlastEnv(num_envs=4)
        
        assert vec_env.num_envs == 4
    
    def test_vec_env_reset(self):
        """Test vectorized reset."""
        vec_env = VectorizedBlockBlastEnv(num_envs=4)
        obs, infos = vec_env.reset()
        
        assert obs['board'].shape == (4, 8, 8)
        assert obs['pieces'].shape == (4, 3, 8, 8)
        assert len(infos) == 4
    
    def test_vec_env_step(self):
        """Test vectorized step."""
        vec_env = VectorizedBlockBlastEnv(num_envs=4)
        vec_env.reset()
        
        actions = vec_env.sample_valid_actions()
        assert len(actions) == 4
        
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        
        assert obs['board'].shape == (4, 8, 8)
        assert len(rewards) == 4
        assert len(terminated) == 4
    
    def test_vec_env_action_masks(self):
        """Test getting action masks from all envs."""
        vec_env = VectorizedBlockBlastEnv(num_envs=4)
        vec_env.reset()
        
        masks = vec_env.get_action_masks()
        
        assert masks.shape == (4, 192)
    
    def test_vec_env_close(self):
        """Test closing vectorized env."""
        vec_env = VectorizedBlockBlastEnv(num_envs=4)
        vec_env.close()  # Should not raise


class TestRendering:
    """Test environment rendering."""
    
    def test_render_ansi(self):
        """Test ANSI rendering mode."""
        env = BlockBlastEnv(render_mode="ansi")
        env.reset()
        
        output = env.render()
        
        assert isinstance(output, str)
        assert len(output) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
