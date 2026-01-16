"""
Tests for the game board.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from game.board import Board
from game.pieces import (
    SINGLE, DOMINO_H, DOMINO_V, TRIO_H, TRIO_V,
    I_H, I_V, I5_H, O, SQUARE_3x3, L_1, T_UP
)


class TestBoardBasics:
    """Test basic board operations."""
    
    def test_board_creation(self):
        """Test board initialization."""
        board = Board()
        assert board.size == 8
        assert board.total_blocks == 0
        assert board.empty_cells == 64
    
    def test_board_custom_size(self):
        """Test board with custom size."""
        board = Board(size=4)
        assert board.size == 4
        assert board.empty_cells == 16
    
    def test_board_initially_empty(self):
        """New board should be all zeros."""
        board = Board()
        assert np.all(board.grid == 0)
    
    def test_board_reset(self):
        """Test board reset."""
        board = Board()
        board.place_piece(SINGLE, 0, 0)
        assert board.total_blocks == 1
        
        board.reset()
        assert board.total_blocks == 0
        assert np.all(board.grid == 0)
    
    def test_board_copy(self):
        """Test board copying."""
        board = Board()
        board.place_piece(SINGLE, 0, 0)
        
        copy = board.copy()
        assert np.array_equal(board.grid, copy.grid)
        
        # Modifying copy shouldn't affect original
        copy.place_piece(SINGLE, 1, 1)
        assert board.total_blocks == 1
        assert copy.total_blocks == 2


class TestBoardCells:
    """Test cell operations."""
    
    def test_in_bounds(self):
        """Test bounds checking."""
        board = Board()
        
        assert board.in_bounds(0, 0)
        assert board.in_bounds(7, 7)
        assert board.in_bounds(4, 4)
        
        assert not board.in_bounds(-1, 0)
        assert not board.in_bounds(0, -1)
        assert not board.in_bounds(8, 0)
        assert not board.in_bounds(0, 8)
    
    def test_is_empty_filled(self):
        """Test empty/filled cell checking."""
        board = Board()
        
        assert board.is_empty(0, 0)
        assert not board.is_filled(0, 0)
        
        board.place_piece(SINGLE, 0, 0)
        
        assert not board.is_empty(0, 0)
        assert board.is_filled(0, 0)
    
    def test_get_cell(self):
        """Test getting cell values."""
        board = Board()
        assert board.get_cell(0, 0) == 0
        
        board.place_piece(SINGLE, 0, 0)
        assert board.get_cell(0, 0) == 1


class TestPiecePlacement:
    """Test piece placement."""
    
    def test_place_single(self):
        """Test placing single block."""
        board = Board()
        result = board.place_piece(SINGLE, 0, 0)
        
        assert result is True
        assert board.total_blocks == 1
        assert board.is_filled(0, 0)
    
    def test_place_domino_h(self):
        """Test placing horizontal domino."""
        board = Board()
        result = board.place_piece(DOMINO_H, 0, 0)
        
        assert result is True
        assert board.total_blocks == 2
        assert board.is_filled(0, 0)
        assert board.is_filled(0, 1)
    
    def test_place_domino_v(self):
        """Test placing vertical domino."""
        board = Board()
        result = board.place_piece(DOMINO_V, 0, 0)
        
        assert result is True
        assert board.total_blocks == 2
        assert board.is_filled(0, 0)
        assert board.is_filled(1, 0)
    
    def test_place_large_piece(self):
        """Test placing 3x3 square."""
        board = Board()
        result = board.place_piece(SQUARE_3x3, 0, 0)
        
        assert result is True
        assert board.total_blocks == 9
        
        for r in range(3):
            for c in range(3):
                assert board.is_filled(r, c)
    
    def test_place_at_corner(self):
        """Test placing at corners."""
        board = Board()
        
        # Top-left
        board.place_piece(SINGLE, 0, 0)
        assert board.is_filled(0, 0)
        
        # Top-right
        board.place_piece(SINGLE, 0, 7)
        assert board.is_filled(0, 7)
        
        # Bottom-left
        board.place_piece(SINGLE, 7, 0)
        assert board.is_filled(7, 0)
        
        # Bottom-right
        board.place_piece(SINGLE, 7, 7)
        assert board.is_filled(7, 7)
    
    def test_place_at_edges(self):
        """Test placing at board edges."""
        board = Board()
        
        # Place I_H at right edge
        result = board.place_piece(I_H, 0, 4)
        assert result is True  # Should fit (positions 4,5,6,7)
        
        board.reset()
        
        # Place I_V at bottom edge
        result = board.place_piece(I_V, 4, 0)
        assert result is True  # Should fit (positions 4,5,6,7)


class TestPlacementValidation:
    """Test placement validation."""
    
    def test_can_place_empty_board(self):
        """Any piece can be placed on empty board (in bounds)."""
        board = Board()
        
        assert board.can_place(SINGLE, 0, 0)
        assert board.can_place(SINGLE, 7, 7)
        assert board.can_place(SQUARE_3x3, 0, 0)
        assert board.can_place(SQUARE_3x3, 5, 5)
    
    def test_cannot_place_out_of_bounds(self):
        """Pieces cannot be placed out of bounds."""
        board = Board()
        
        # I_H (4 blocks wide) at column 6 would overflow
        assert not board.can_place(I_H, 0, 6)
        
        # SQUARE_3x3 at (6, 6) would overflow
        assert not board.can_place(SQUARE_3x3, 6, 6)
        
        # Negative positions
        assert not board.can_place(SINGLE, -1, 0)
        assert not board.can_place(SINGLE, 0, -1)
    
    def test_cannot_place_collision(self):
        """Pieces cannot overlap existing blocks."""
        board = Board()
        board.place_piece(SINGLE, 4, 4)
        
        # Try to place on same spot
        assert not board.can_place(SINGLE, 4, 4)
        
        # Try to place overlapping piece
        assert not board.can_place(SQUARE_3x3, 3, 3)  # Would cover (4, 4)
    
    def test_failed_placement_returns_false(self):
        """Invalid placements should return False."""
        board = Board()
        board.place_piece(SINGLE, 0, 0)
        
        result = board.place_piece(SINGLE, 0, 0)
        assert result is False
        assert board.total_blocks == 1  # No change


class TestValidPlacements:
    """Test finding valid placements."""
    
    def test_get_valid_placements_empty(self):
        """SINGLE can be placed anywhere on empty board."""
        board = Board()
        placements = board.get_valid_placements(SINGLE)
        assert len(placements) == 64
    
    def test_get_valid_placements_i_h(self):
        """I_H can only be placed in positions 0-4."""
        board = Board()
        placements = board.get_valid_placements(I_H)
        # 5 columns (0-4) * 8 rows = 40 positions
        assert len(placements) == 40
    
    def test_get_valid_placements_with_blocks(self):
        """Placing blocks reduces valid placements."""
        board = Board()
        initial = len(board.get_valid_placements(SINGLE))
        
        board.place_piece(SINGLE, 4, 4)
        after = len(board.get_valid_placements(SINGLE))
        
        assert after == initial - 1
    
    def test_has_valid_placement(self):
        """Test checking if any placement exists."""
        board = Board()
        assert board.has_valid_placement(SINGLE)
        assert board.has_valid_placement(SQUARE_3x3)


class TestLineClearingRows:
    """Test row clearing."""
    
    def test_clear_single_row(self):
        """Test clearing one complete row."""
        board = Board()
        
        # Fill row 0 completely
        for col in range(8):
            board.place_piece(SINGLE, 0, col)
        
        assert board.total_blocks == 8
        
        complete_rows, complete_cols = board.find_complete_lines()
        assert 0 in complete_rows
        assert len(complete_cols) == 0
        
        rows, cols = board.clear_lines()
        assert rows == 1
        assert cols == 0
        assert board.total_blocks == 0
    
    def test_clear_multiple_rows(self):
        """Test clearing multiple complete rows."""
        board = Board()
        
        # Fill rows 0 and 1 completely
        for row in range(2):
            for col in range(8):
                board.place_piece(SINGLE, row, col)
        
        rows, cols = board.clear_lines()
        assert rows == 2
        assert board.total_blocks == 0
    
    def test_partial_row_not_cleared(self):
        """Incomplete rows should not be cleared."""
        board = Board()
        
        # Fill 7 of 8 cells in row 0
        for col in range(7):
            board.place_piece(SINGLE, 0, col)
        
        rows, cols = board.clear_lines()
        assert rows == 0
        assert board.total_blocks == 7


class TestLineClearingCols:
    """Test column clearing."""
    
    def test_clear_single_column(self):
        """Test clearing one complete column."""
        board = Board()
        
        # Fill column 0 completely
        for row in range(8):
            board.place_piece(SINGLE, row, 0)
        
        complete_rows, complete_cols = board.find_complete_lines()
        assert 0 in complete_cols
        
        rows, cols = board.clear_lines()
        assert cols == 1
        assert rows == 0
        assert board.total_blocks == 0
    
    def test_clear_multiple_columns(self):
        """Test clearing multiple complete columns."""
        board = Board()
        
        # Fill columns 0 and 1
        for col in range(2):
            for row in range(8):
                board.place_piece(SINGLE, row, col)
        
        rows, cols = board.clear_lines()
        assert cols == 2


class TestLineClearingBoth:
    """Test simultaneous row and column clearing."""
    
    def test_clear_row_and_column(self):
        """Test clearing a row and column at once."""
        board = Board()
        
        # Fill row 4
        for col in range(8):
            board.place_piece(SINGLE, 4, col)
        
        # Fill column 4 (except intersection already filled)
        for row in range(8):
            if row != 4:
                board.place_piece(SINGLE, row, 4)
        
        assert board.total_blocks == 15  # 8 + 7
        
        rows, cols = board.clear_lines()
        assert rows == 1
        assert cols == 1
        assert board.total_blocks == 0
    
    def test_full_board_clear(self):
        """Filling entire board should clear everything."""
        board = Board()
        
        # Fill entire board
        for row in range(8):
            for col in range(8):
                board.place_piece(SINGLE, row, col)
        
        assert board.total_blocks == 64
        
        rows, cols = board.clear_lines()
        assert rows == 8
        assert cols == 8
        assert board.total_blocks == 0


class TestBoardHeuristics:
    """Test board analysis heuristics."""
    
    def test_count_holes_empty(self):
        """Empty board has no holes."""
        board = Board()
        assert board.count_holes() == 0
    
    def test_count_holes_single(self):
        """Surrounded empty cell is a hole."""
        board = Board()
        
        # Create a hole at (1, 1)
        board.place_piece(SINGLE, 0, 1)  # Above
        board.place_piece(SINGLE, 2, 1)  # Below
        board.place_piece(SINGLE, 1, 0)  # Left
        board.place_piece(SINGLE, 1, 2)  # Right
        
        # This creates 2 holes: (0,0) and (1,1)
        # (0,0) has neighbors: OOB, filled(1,0), OOB, filled(0,1) = 4 blocked
        # (1,1) has neighbors: filled(0,1), filled(2,1), filled(1,0), filled(1,2) = 4 blocked
        assert board.count_holes() == 2
    
    def test_center_openness_empty(self):
        """Empty board has center openness of 1.0."""
        board = Board()
        assert board.get_center_openness() == 1.0
    
    def test_center_openness_full_center(self):
        """Full center has openness of 0.0."""
        board = Board()
        
        # Fill center 4x4
        for row in range(2, 6):
            for col in range(2, 6):
                board.place_piece(SINGLE, row, col)
        
        assert board.get_center_openness() == 0.0
    
    def test_get_height_map(self):
        """Test height map calculation."""
        board = Board()
        
        # Place pieces in column 0
        board.place_piece(SINGLE, 7, 0)
        board.place_piece(SINGLE, 6, 0)
        
        heights = board.get_height_map()
        assert heights[0] == 2  # Two blocks high
        assert heights[1] == 0  # Empty column


class TestBoardSerialization:
    """Test board state serialization."""
    
    def test_get_set_state(self):
        """Test state save/restore."""
        board = Board()
        board.place_piece(SQUARE_3x3, 0, 0)
        
        state = board.get_state()
        
        new_board = Board()
        new_board.set_state(state)
        
        assert np.array_equal(board.grid, new_board.grid)
        assert board.total_blocks == new_board.total_blocks
    
    def test_to_tensor(self):
        """Test tensor conversion."""
        board = Board()
        board.place_piece(SINGLE, 0, 0)
        
        tensor = board.to_tensor()
        
        assert tensor.dtype == np.float32
        assert tensor.shape == (8, 8)
        assert tensor[0, 0] == 1.0


class TestBoardEquality:
    """Test board equality and hashing."""
    
    def test_boards_equal(self):
        """Identical boards should be equal."""
        board1 = Board()
        board1.place_piece(SINGLE, 0, 0)
        
        board2 = Board()
        board2.place_piece(SINGLE, 0, 0)
        
        assert board1 == board2
    
    def test_boards_not_equal(self):
        """Different boards should not be equal."""
        board1 = Board()
        board1.place_piece(SINGLE, 0, 0)
        
        board2 = Board()
        board2.place_piece(SINGLE, 1, 1)
        
        assert board1 != board2
    
    def test_board_hashable(self):
        """Boards should be hashable."""
        board = Board()
        board.place_piece(SINGLE, 0, 0)
        
        hash_value = hash(board)
        assert isinstance(hash_value, int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
