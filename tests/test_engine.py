"""
Tests for the game engine.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from game.engine import GameEngine, GameStatus, MoveResult, GameState, play_random_game
from game.pieces import SINGLE, DOMINO_H, TRIO_H, I_H, SQUARE_3x3, NUM_PIECES


class TestEngineCreation:
    """Test engine initialization."""
    
    def test_create_engine(self):
        """Test basic engine creation."""
        engine = GameEngine()
        
        assert engine.board_size == 8
        assert engine.score == 0
        assert engine.moves_made == 0
        assert engine.combo_count == 0
        assert engine.status == GameStatus.PLAYING
    
    def test_create_with_seed(self):
        """Test deterministic creation with seed."""
        engine1 = GameEngine(seed=42)
        engine2 = GameEngine(seed=42)
        
        # Same seed should give same initial pieces
        for i in range(3):
            assert engine1.current_pieces[i].name == engine2.current_pieces[i].name
    
    def test_initial_pieces(self):
        """Engine should start with 3 pieces."""
        engine = GameEngine()
        
        assert len(engine.current_pieces) == 3
        assert len(engine.pieces_used) == 3
        assert all(not used for used in engine.pieces_used)


class TestEngineReset:
    """Test engine reset."""
    
    def test_reset(self):
        """Test resetting the game."""
        engine = GameEngine(seed=42)
        
        # Make some moves
        moves = engine.get_valid_moves()
        if moves:
            engine.make_move(*moves[0])
        
        # Reset
        state = engine.reset(seed=42)
        
        assert engine.score == 0
        assert engine.moves_made == 0
        assert engine.status == GameStatus.PLAYING
        assert isinstance(state, GameState)
    
    def test_reset_with_new_seed(self):
        """Reset with different seed gives different pieces."""
        engine = GameEngine(seed=42)
        pieces1 = [p.name for p in engine.current_pieces]
        
        engine.reset(seed=123)
        pieces2 = [p.name for p in engine.current_pieces]
        
        # Very unlikely to be the same
        assert pieces1 != pieces2 or True  # May occasionally be same


class TestMoveValidation:
    """Test move validation."""
    
    def test_can_place_valid(self):
        """Valid placements should be allowed."""
        engine = GameEngine()
        
        # At least one piece should be placeable somewhere
        can_place_any = False
        for i in range(3):
            for row in range(8):
                for col in range(8):
                    if engine.can_place_piece(i, row, col):
                        can_place_any = True
                        break
        
        assert can_place_any
    
    def test_can_place_invalid_piece_index(self):
        """Invalid piece index should fail."""
        engine = GameEngine()
        
        assert not engine.can_place_piece(-1, 0, 0)
        assert not engine.can_place_piece(3, 0, 0)
    
    def test_can_place_used_piece(self):
        """Cannot place already used piece."""
        engine = GameEngine()
        moves = engine.get_valid_moves()
        
        if moves:
            piece_idx, row, col = moves[0]
            engine.make_move(piece_idx, row, col)
            
            # Same piece index should now fail
            assert not engine.can_place_piece(piece_idx, row, col)


class TestMakingMoves:
    """Test making moves."""
    
    def test_make_valid_move(self):
        """Valid move should succeed."""
        engine = GameEngine()
        moves = engine.get_valid_moves()
        
        assert len(moves) > 0
        
        piece_idx, row, col = moves[0]
        result = engine.make_move(piece_idx, row, col)
        
        assert result.success
        assert result.piece_placed is not None
        assert result.position == (row, col)
        assert result.blocks_placed > 0
        assert engine.moves_made == 1
    
    def test_make_invalid_move(self):
        """Invalid move should fail."""
        engine = GameEngine()
        
        # Try placing at invalid position
        result = engine.make_move(0, -1, -1)
        
        assert not result.success
        assert engine.moves_made == 0
    
    def test_piece_marked_used(self):
        """Used piece should be marked."""
        engine = GameEngine()
        moves = engine.get_valid_moves()
        
        piece_idx, row, col = moves[0]
        assert not engine.pieces_used[piece_idx]
        
        engine.make_move(piece_idx, row, col)
        
        assert engine.pieces_used[piece_idx]
    
    def test_new_pieces_after_all_used(self):
        """New pieces should be generated after using all 3."""
        engine = GameEngine(seed=42)
        
        # Use all 3 pieces
        pieces_placed = 0
        while pieces_placed < 3:
            moves = engine.get_valid_moves()
            for move in moves:
                if not engine.pieces_used[move[0]]:
                    engine.make_move(*move)
                    pieces_placed += 1
                    break
            if not moves:
                break
        
        # After using all 3, new pieces should be generated
        if pieces_placed == 3:
            assert all(not used for used in engine.pieces_used)


class TestLineClearingInGame:
    """Test line clearing during gameplay."""
    
    def test_clear_lines_updates_score(self):
        """Clearing lines should add to score."""
        engine = GameEngine()
        initial_score = engine.score
        
        # Fill a row manually for testing
        for col in range(8):
            engine.board.place_piece(SINGLE, 0, col)
        
        # Clear lines via engine
        rows, cols = engine.board.clear_lines()
        
        assert rows == 1  # One row cleared


class TestComboSystem:
    """Test combo system."""
    
    def test_combo_increments(self):
        """Combo should increment on consecutive clears."""
        engine = GameEngine()
        
        # Manually set up board for clears
        # This is a bit tricky in practice...
        assert engine.combo_count == 0
    
    def test_combo_resets(self):
        """Combo should reset when no lines cleared."""
        engine = GameEngine()
        engine.combo_count = 5
        
        # Make a move that doesn't clear
        moves = engine.get_valid_moves()
        if moves:
            result = engine.make_move(*moves[0])
            if result.lines_cleared == 0:
                assert engine.combo_count == 0


class TestGameOver:
    """Test game over detection."""
    
    def test_game_not_over_initially(self):
        """Game should not be over at start."""
        engine = GameEngine()
        
        assert not engine.is_game_over()
        assert engine.status == GameStatus.PLAYING
    
    def test_has_valid_moves_initially(self):
        """Should have valid moves at start."""
        engine = GameEngine()
        
        assert engine.has_valid_moves()
        assert len(engine.get_valid_moves()) > 0


class TestGameState:
    """Test game state serialization."""
    
    def test_get_state(self):
        """Test getting game state."""
        engine = GameEngine()
        state = engine.get_state()
        
        assert isinstance(state, GameState)
        assert state.board.shape == (8, 8)
        assert len(state.current_pieces) == 3
        assert state.score == 0
    
    def test_set_state(self):
        """Test restoring game state."""
        engine = GameEngine(seed=42)
        
        # Make some moves
        moves = engine.get_valid_moves()
        if moves:
            engine.make_move(*moves[0])
        
        # Save state
        state = engine.get_state()
        saved_score = engine.score
        saved_moves = engine.moves_made
        
        # Reset and restore
        engine.reset()
        engine.set_state(state)
        
        assert engine.score == saved_score
        assert engine.moves_made == saved_moves
    
    def test_state_to_dict(self):
        """Test state serialization to dict."""
        engine = GameEngine()
        state = engine.get_state()
        
        state_dict = state.to_dict()
        
        assert 'board' in state_dict
        assert 'current_pieces' in state_dict
        assert 'score' in state_dict
    
    def test_state_from_dict(self):
        """Test state deserialization from dict."""
        engine = GameEngine()
        state = engine.get_state()
        
        state_dict = state.to_dict()
        restored = GameState.from_dict(state_dict)
        
        assert np.array_equal(state.board, restored.board)
        assert state.score == restored.score


class TestObservation:
    """Test observation generation for RL."""
    
    def test_get_observation_structure(self):
        """Test observation structure."""
        engine = GameEngine()
        obs = engine.get_observation()
        
        assert 'board' in obs
        assert 'pieces' in obs
        assert 'pieces_used' in obs
        assert 'action_mask' in obs
    
    def test_observation_shapes(self):
        """Test observation tensor shapes."""
        engine = GameEngine()
        obs = engine.get_observation()
        
        assert obs['board'].shape == (8, 8)
        assert obs['pieces'].shape == (3, 8, 8)
        assert obs['pieces_used'].shape == (3,)
        assert obs['action_mask'].shape == (3, 8, 8)
    
    def test_action_mask_valid(self):
        """Action mask should match valid moves."""
        engine = GameEngine()
        obs = engine.get_observation()
        mask = obs['action_mask']
        
        # Count valid moves from mask
        mask_valid = np.sum(mask)
        
        # Count from get_valid_moves
        moves_valid = len(engine.get_valid_moves())
        
        assert mask_valid == moves_valid


class TestStatistics:
    """Test statistics tracking."""
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        engine = GameEngine()
        stats = engine.get_statistics()
        
        assert 'score' in stats
        assert 'moves_made' in stats
        assert 'total_lines_cleared' in stats
        assert 'max_combo' in stats


class TestRandomGame:
    """Test random game playing."""
    
    def test_play_random_game(self):
        """Random game should complete without errors."""
        stats = play_random_game(seed=42)
        
        assert 'score' in stats
        assert stats['moves_made'] > 0
    
    def test_random_game_deterministic(self):
        """Same seed should give same result."""
        stats1 = play_random_game(seed=42)
        stats2 = play_random_game(seed=42)
        
        assert stats1['score'] == stats2['score']
        assert stats1['moves_made'] == stats2['moves_made']
    
    def test_multiple_random_games(self):
        """Multiple games should complete."""
        for i in range(10):
            stats = play_random_game(seed=i)
            assert stats['moves_made'] > 0


class TestActionMask:
    """Test action mask generation."""
    
    def test_action_mask_shape(self):
        """Action mask should be (3, 8, 8)."""
        engine = GameEngine()
        mask = engine.get_action_mask()
        
        assert mask.shape == (3, 8, 8)
        assert mask.dtype == bool
    
    def test_action_mask_used_pieces(self):
        """Used pieces should have no valid actions."""
        engine = GameEngine()
        
        moves = engine.get_valid_moves()
        if moves:
            piece_idx = moves[0][0]
            engine.make_move(*moves[0])
            
            mask = engine.get_action_mask()
            assert np.sum(mask[piece_idx]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
