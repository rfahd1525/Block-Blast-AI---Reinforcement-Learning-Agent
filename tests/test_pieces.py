"""
Tests for piece definitions.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from game.pieces import (
    Piece, PIECES, PIECE_LIST, NUM_PIECES,
    get_piece_by_name, get_piece_by_index, get_all_pieces,
    get_random_pieces, piece_to_one_hot, visualize_piece,
    # Individual pieces
    SINGLE, DOMINO_H, DOMINO_V, DIAG2_TL_BR, DIAG2_TR_BL,
    TRIO_H, TRIO_V, DIAG3_TL_BR, DIAG3_TR_BL,
    TRIO_L1, TRIO_L2, TRIO_L3, TRIO_L4,
    I_H, I_V, I5_H, I5_V, O,
    T_UP, T_DOWN, T_LEFT, T_RIGHT,
    S_H, S_V, Z_H, Z_V,
    L_1, L_2, L_3, L_4,
    J_1, J_2, J_3, J_4,
    RECT_2x3_H, RECT_2x3_V, SQUARE_3x3,
)


class TestPieceCount:
    """Test that we have exactly 37 pieces."""
    
    def test_total_piece_count(self):
        """Verify exactly 37 pieces are defined."""
        assert NUM_PIECES == 37
        assert len(PIECES) == 37
        assert len(PIECE_LIST) == 37
    
    def test_piece_categories(self):
        """Verify piece counts by category."""
        # Single: 1
        assert SINGLE.num_blocks == 1
        
        # Dominos: 2
        assert DOMINO_H.num_blocks == 2
        assert DOMINO_V.num_blocks == 2
        
        # Diagonal 2-block: 2
        assert DIAG2_TL_BR.num_blocks == 2
        assert DIAG2_TR_BL.num_blocks == 2
        
        # Triomino straight: 2
        assert TRIO_H.num_blocks == 3
        assert TRIO_V.num_blocks == 3
        
        # Diagonal 3-block: 2
        assert DIAG3_TL_BR.num_blocks == 3
        assert DIAG3_TR_BL.num_blocks == 3
        
        # Triomino L-shapes: 4
        for piece in [TRIO_L1, TRIO_L2, TRIO_L3, TRIO_L4]:
            assert piece.num_blocks == 3
        
        # I-pieces 4-block: 2
        assert I_H.num_blocks == 4
        assert I_V.num_blocks == 4
        
        # I-pieces 5-block: 2
        assert I5_H.num_blocks == 5
        assert I5_V.num_blocks == 5
        
        # O-piece: 1
        assert O.num_blocks == 4
        
        # T-pieces: 4
        for piece in [T_UP, T_DOWN, T_LEFT, T_RIGHT]:
            assert piece.num_blocks == 4
        
        # S-pieces: 2
        assert S_H.num_blocks == 4
        assert S_V.num_blocks == 4
        
        # Z-pieces: 2
        assert Z_H.num_blocks == 4
        assert Z_V.num_blocks == 4
        
        # L-pieces: 4
        for piece in [L_1, L_2, L_3, L_4]:
            assert piece.num_blocks == 4
        
        # J-pieces: 4
        for piece in [J_1, J_2, J_3, J_4]:
            assert piece.num_blocks == 4
        
        # Rectangles: 2
        assert RECT_2x3_H.num_blocks == 6
        assert RECT_2x3_V.num_blocks == 6
        
        # Large square: 1
        assert SQUARE_3x3.num_blocks == 9


class TestPieceProperties:
    """Test piece properties."""
    
    def test_piece_immutable(self):
        """Pieces should be immutable (frozen dataclass)."""
        with pytest.raises(Exception):
            SINGLE.name = "modified"
    
    def test_piece_blocks_normalized(self):
        """All piece blocks should start from (0, 0)."""
        for piece in PIECE_LIST:
            min_row = min(r for r, _ in piece.blocks)
            min_col = min(c for _, c in piece.blocks)
            assert min_row == 0, f"{piece.name} not normalized: min_row={min_row}"
            assert min_col == 0, f"{piece.name} not normalized: min_col={min_col}"
    
    def test_piece_no_duplicates(self):
        """No piece should have duplicate blocks."""
        for piece in PIECE_LIST:
            blocks = list(piece.blocks)
            unique_blocks = set(piece.blocks)
            assert len(blocks) == len(unique_blocks), f"{piece.name} has duplicate blocks"
    
    def test_piece_dimensions(self):
        """Test piece width and height calculations."""
        assert SINGLE.width == 1 and SINGLE.height == 1
        assert DOMINO_H.width == 2 and DOMINO_H.height == 1
        assert DOMINO_V.width == 1 and DOMINO_V.height == 2
        assert I_H.width == 4 and I_H.height == 1
        assert I_V.width == 1 and I_V.height == 4
        assert O.width == 2 and O.height == 2
        assert SQUARE_3x3.width == 3 and SQUARE_3x3.height == 3


class TestPieceShapes:
    """Test specific piece shapes."""
    
    def test_single_shape(self):
        """SINGLE should be just (0, 0)."""
        assert SINGLE.blocks == ((0, 0),)
    
    def test_domino_h_shape(self):
        """DOMINO_H should be horizontal."""
        assert set(DOMINO_H.blocks) == {(0, 0), (0, 1)}
    
    def test_domino_v_shape(self):
        """DOMINO_V should be vertical."""
        assert set(DOMINO_V.blocks) == {(0, 0), (1, 0)}
    
    def test_trio_h_shape(self):
        """TRIO_H should be 3 blocks horizontal."""
        assert set(TRIO_H.blocks) == {(0, 0), (0, 1), (0, 2)}
    
    def test_trio_v_shape(self):
        """TRIO_V should be 3 blocks vertical."""
        assert set(TRIO_V.blocks) == {(0, 0), (1, 0), (2, 0)}
    
    def test_i_h_shape(self):
        """I_H should be 4 blocks horizontal."""
        assert set(I_H.blocks) == {(0, 0), (0, 1), (0, 2), (0, 3)}
    
    def test_i5_v_shape(self):
        """I5_V should be 5 blocks vertical."""
        assert set(I5_V.blocks) == {(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)}
    
    def test_o_shape(self):
        """O should be 2x2 square."""
        assert set(O.blocks) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    
    def test_square_3x3_shape(self):
        """SQUARE_3x3 should be 3x3 filled square."""
        expected = {(r, c) for r in range(3) for c in range(3)}
        assert set(SQUARE_3x3.blocks) == expected
    
    def test_t_up_shape(self):
        """T_UP should have T pointing up."""
        # Should have center column at top, three blocks below
        assert set(T_UP.blocks) == {(0, 1), (1, 0), (1, 1), (1, 2)}
    
    def test_l_shape(self):
        """L_1 should be vertical L."""
        assert set(L_1.blocks) == {(0, 0), (1, 0), (2, 0), (2, 1)}


class TestPieceHelpers:
    """Test helper functions."""
    
    def test_get_piece_by_name(self):
        """Test getting pieces by name."""
        assert get_piece_by_name("SINGLE") == SINGLE
        assert get_piece_by_name("O") == O
        assert get_piece_by_name("SQUARE_3x3") == SQUARE_3x3
    
    def test_get_piece_by_name_invalid(self):
        """Invalid names should raise ValueError."""
        with pytest.raises(ValueError):
            get_piece_by_name("INVALID")
    
    def test_get_piece_by_index(self):
        """Test getting pieces by index."""
        for i in range(NUM_PIECES):
            piece = get_piece_by_index(i)
            assert piece == PIECE_LIST[i]
    
    def test_get_piece_by_index_invalid(self):
        """Invalid indices should raise ValueError."""
        with pytest.raises(ValueError):
            get_piece_by_index(-1)
        with pytest.raises(ValueError):
            get_piece_by_index(NUM_PIECES)
    
    def test_get_all_pieces(self):
        """get_all_pieces should return a copy."""
        pieces = get_all_pieces()
        assert len(pieces) == NUM_PIECES
        assert pieces is not PIECE_LIST  # Should be a copy
    
    def test_get_random_pieces(self):
        """Test random piece generation."""
        pieces = get_random_pieces(3)
        assert len(pieces) == 3
        for piece in pieces:
            assert piece in PIECE_LIST
    
    def test_get_random_pieces_with_seed(self):
        """Random pieces with same seed should be identical."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        
        pieces1 = get_random_pieces(3, rng1)
        pieces2 = get_random_pieces(3, rng2)
        
        assert pieces1 == pieces2
    
    def test_piece_to_one_hot(self):
        """Test one-hot encoding."""
        one_hot = piece_to_one_hot(SINGLE)
        assert one_hot.shape == (NUM_PIECES,)
        assert np.sum(one_hot) == 1
        assert one_hot[0] == 1  # SINGLE is first
    
    def test_piece_to_mask(self):
        """Test piece mask generation."""
        mask = SINGLE.to_mask(8)
        assert mask.shape == (8, 8)
        assert mask[0, 0] == 1.0
        assert np.sum(mask) == 1.0
        
        mask = O.to_mask(8)
        assert np.sum(mask) == 4.0
    
    def test_visualize_piece(self):
        """Test piece visualization."""
        vis = visualize_piece(SINGLE)
        assert "□" in vis
        
        vis = visualize_piece(DOMINO_H)
        assert "□□" in vis


class TestPieceArrays:
    """Test piece array representations."""
    
    def test_get_shape_array(self):
        """Test shape array generation."""
        arr = SINGLE.get_shape_array()
        assert arr.shape == (1, 1)
        assert arr[0, 0] == 1
        
        arr = DOMINO_H.get_shape_array()
        assert arr.shape == (1, 2)
        assert np.array_equal(arr, [[1, 1]])
        
        arr = O.get_shape_array()
        assert arr.shape == (2, 2)
        assert np.array_equal(arr, [[1, 1], [1, 1]])
        
        arr = SQUARE_3x3.get_shape_array()
        assert arr.shape == (3, 3)
        assert np.all(arr == 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
