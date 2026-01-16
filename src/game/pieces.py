"""
Block Blast Piece Definitions.

This module defines all 37 unique pieces in the Block Blast game.
Each piece is represented as a list of (row, col) offsets from the top-left anchor point.
"""
from dataclasses import dataclass
from typing import List, Tuple, Dict
import numpy as np


@dataclass(frozen=True)
class Piece:
    """Represents a Block Blast piece."""
    name: str
    blocks: Tuple[Tuple[int, int], ...]  # Immutable tuple of (row, col) offsets
    
    @property
    def num_blocks(self) -> int:
        """Return the number of blocks in this piece."""
        return len(self.blocks)
    
    @property
    def width(self) -> int:
        """Return the width of the bounding box."""
        if not self.blocks:
            return 0
        cols = [c for _, c in self.blocks]
        return max(cols) - min(cols) + 1
    
    @property
    def height(self) -> int:
        """Return the height of the bounding box."""
        if not self.blocks:
            return 0
        rows = [r for r, _ in self.blocks]
        return max(rows) - min(rows) + 1
    
    def to_mask(self, board_size: int = 8) -> np.ndarray:
        """Convert piece to a binary mask on an 8x8 grid (placed at origin)."""
        mask = np.zeros((board_size, board_size), dtype=np.float32)
        for r, c in self.blocks:
            if 0 <= r < board_size and 0 <= c < board_size:
                mask[r, c] = 1.0
        return mask
    
    def get_shape_array(self) -> np.ndarray:
        """Get a minimal bounding box array for this piece."""
        arr = np.zeros((self.height, self.width), dtype=np.int8)
        min_row = min(r for r, _ in self.blocks)
        min_col = min(c for _, c in self.blocks)
        for r, c in self.blocks:
            arr[r - min_row, c - min_col] = 1
        return arr
    
    def __repr__(self) -> str:
        return f"Piece({self.name}, {self.num_blocks} blocks)"


def _make_piece(name: str, blocks: List[Tuple[int, int]]) -> Piece:
    """Helper to create a Piece with normalized coordinates."""
    # Normalize so top-left is at (0, 0)
    if not blocks:
        return Piece(name, tuple())
    min_r = min(r for r, _ in blocks)
    min_c = min(c for _, c in blocks)
    normalized = tuple((r - min_r, c - min_c) for r, c in blocks)
    return Piece(name, normalized)


# =============================================================================
# ALL 37 PIECES DEFINED EXACTLY AS SPECIFIED
# =============================================================================

# -----------------------------------------------------------------------------
# Single Block (1 piece)
# -----------------------------------------------------------------------------
SINGLE = _make_piece("SINGLE", [(0, 0)])

# -----------------------------------------------------------------------------
# Domino Pieces (2 pieces)
# -----------------------------------------------------------------------------
DOMINO_H = _make_piece("DOMINO_H", [(0, 0), (0, 1)])  # □□
DOMINO_V = _make_piece("DOMINO_V", [(0, 0), (1, 0)])  # □
                                                       # □

# -----------------------------------------------------------------------------
# Diagonal 2-Block Pieces (2 pieces)
# -----------------------------------------------------------------------------
DIAG2_TL_BR = _make_piece("DIAG2_TL_BR", [(0, 0), (1, 1)])  # □
                                                             #  □
DIAG2_TR_BL = _make_piece("DIAG2_TR_BL", [(0, 1), (1, 0)])  #  □
                                                             # □

# -----------------------------------------------------------------------------
# Triomino Straight Pieces (2 pieces)
# -----------------------------------------------------------------------------
TRIO_H = _make_piece("TRIO_H", [(0, 0), (0, 1), (0, 2)])  # □□□
TRIO_V = _make_piece("TRIO_V", [(0, 0), (1, 0), (2, 0)])  # □
                                                           # □
                                                           # □

# -----------------------------------------------------------------------------
# Diagonal 3-Block Pieces (2 pieces)
# -----------------------------------------------------------------------------
DIAG3_TL_BR = _make_piece("DIAG3_TL_BR", [(0, 0), (1, 1), (2, 2)])  # □
                                                                     #  □
                                                                     #   □
DIAG3_TR_BL = _make_piece("DIAG3_TR_BL", [(0, 2), (1, 1), (2, 0)])  #   □
                                                                     #  □
                                                                     # □

# -----------------------------------------------------------------------------
# Triomino L-Shapes (4 pieces)
# -----------------------------------------------------------------------------
TRIO_L1 = _make_piece("TRIO_L1", [(0, 0), (1, 0), (1, 1)])  # □
                                                             # □□
TRIO_L2 = _make_piece("TRIO_L2", [(0, 0), (0, 1), (1, 1)])  # □□
                                                             #  □
TRIO_L3 = _make_piece("TRIO_L3", [(0, 0), (0, 1), (1, 0)])  # □□
                                                             # □
TRIO_L4 = _make_piece("TRIO_L4", [(0, 1), (1, 0), (1, 1)])  #  □
                                                             # □□

# -----------------------------------------------------------------------------
# I-Pieces 4-Block (2 pieces)
# -----------------------------------------------------------------------------
I_H = _make_piece("I_H", [(0, 0), (0, 1), (0, 2), (0, 3)])  # □□□□
I_V = _make_piece("I_V", [(0, 0), (1, 0), (2, 0), (3, 0)])  # □
                                                             # □
                                                             # □
                                                             # □

# -----------------------------------------------------------------------------
# I-Pieces 5-Block (2 pieces)
# -----------------------------------------------------------------------------
I5_H = _make_piece("I5_H", [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)])  # □□□□□
I5_V = _make_piece("I5_V", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)])  # □
                                                                       # □
                                                                       # □
                                                                       # □
                                                                       # □

# -----------------------------------------------------------------------------
# O-Piece (1 piece) - 2x2 square
# -----------------------------------------------------------------------------
O = _make_piece("O", [(0, 0), (0, 1), (1, 0), (1, 1)])  # □□
                                                         # □□

# -----------------------------------------------------------------------------
# T-Pieces (4 pieces)
# -----------------------------------------------------------------------------
T_UP = _make_piece("T_UP", [(0, 1), (1, 0), (1, 1), (1, 2)])    #  □
                                                                 # □□□
T_DOWN = _make_piece("T_DOWN", [(0, 0), (0, 1), (0, 2), (1, 1)])  # □□□
                                                                   #  □
T_LEFT = _make_piece("T_LEFT", [(0, 0), (1, 0), (1, 1), (2, 0)])  # □
                                                                   # □□
                                                                   # □
T_RIGHT = _make_piece("T_RIGHT", [(0, 1), (1, 0), (1, 1), (2, 1)])  #  □
                                                                     # □□
                                                                     #  □

# -----------------------------------------------------------------------------
# S-Pieces (2 pieces)
# -----------------------------------------------------------------------------
S_H = _make_piece("S_H", [(0, 1), (0, 2), (1, 0), (1, 1)])  #  □□
                                                             # □□
S_V = _make_piece("S_V", [(0, 0), (1, 0), (1, 1), (2, 1)])  # □
                                                             # □□
                                                             #  □

# -----------------------------------------------------------------------------
# Z-Pieces (2 pieces)
# -----------------------------------------------------------------------------
Z_H = _make_piece("Z_H", [(0, 0), (0, 1), (1, 1), (1, 2)])  # □□
                                                             #  □□
Z_V = _make_piece("Z_V", [(0, 1), (1, 0), (1, 1), (2, 0)])  #  □
                                                             # □□
                                                             # □

# -----------------------------------------------------------------------------
# L-Pieces (4 pieces)
# -----------------------------------------------------------------------------
L_1 = _make_piece("L_1", [(0, 0), (1, 0), (2, 0), (2, 1)])  # □
                                                             # □
                                                             # □□
L_2 = _make_piece("L_2", [(0, 0), (0, 1), (0, 2), (1, 0)])  # □□□
                                                             # □
L_3 = _make_piece("L_3", [(0, 0), (0, 1), (1, 1), (2, 1)])  # □□
                                                             #  □
                                                             #  □
L_4 = _make_piece("L_4", [(0, 2), (1, 0), (1, 1), (1, 2)])  #   □
                                                             # □□□

# -----------------------------------------------------------------------------
# J-Pieces (4 pieces)
# -----------------------------------------------------------------------------
J_1 = _make_piece("J_1", [(0, 1), (1, 1), (2, 0), (2, 1)])  #  □
                                                             #  □
                                                             # □□
J_2 = _make_piece("J_2", [(0, 0), (1, 0), (1, 1), (1, 2)])  # □
                                                             # □□□
J_3 = _make_piece("J_3", [(0, 0), (0, 1), (1, 0), (2, 0)])  # □□
                                                             # □
                                                             # □
J_4 = _make_piece("J_4", [(0, 0), (0, 1), (0, 2), (1, 2)])  # □□□
                                                             #   □

# -----------------------------------------------------------------------------
# Rectangles (2 pieces)
# -----------------------------------------------------------------------------
RECT_2x3_H = _make_piece("RECT_2x3_H", [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2)
])  # □□□
    # □□□

RECT_2x3_V = _make_piece("RECT_2x3_V", [
    (0, 0), (0, 1),
    (1, 0), (1, 1),
    (2, 0), (2, 1)
])  # □□
    # □□
    # □□

# -----------------------------------------------------------------------------
# Large Square (1 piece) - 3x3 square
# -----------------------------------------------------------------------------
SQUARE_3x3 = _make_piece("SQUARE_3x3", [
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1), (1, 2),
    (2, 0), (2, 1), (2, 2)
])  # □□□
    # □□□
    # □□□


# =============================================================================
# PIECES DICTIONARY AND HELPER FUNCTIONS
# =============================================================================

# Master list of all 37 pieces
PIECES: Dict[str, Piece] = {
    # Single (1)
    "SINGLE": SINGLE,
    
    # Dominos (2)
    "DOMINO_H": DOMINO_H,
    "DOMINO_V": DOMINO_V,
    
    # Diagonal 2-block (2)
    "DIAG2_TL_BR": DIAG2_TL_BR,
    "DIAG2_TR_BL": DIAG2_TR_BL,
    
    # Triomino straight (2)
    "TRIO_H": TRIO_H,
    "TRIO_V": TRIO_V,
    
    # Diagonal 3-block (2)
    "DIAG3_TL_BR": DIAG3_TL_BR,
    "DIAG3_TR_BL": DIAG3_TR_BL,
    
    # Triomino L-shapes (4)
    "TRIO_L1": TRIO_L1,
    "TRIO_L2": TRIO_L2,
    "TRIO_L3": TRIO_L3,
    "TRIO_L4": TRIO_L4,
    
    # I-pieces 4-block (2)
    "I_H": I_H,
    "I_V": I_V,
    
    # I-pieces 5-block (2)
    "I5_H": I5_H,
    "I5_V": I5_V,
    
    # O-piece (1)
    "O": O,
    
    # T-pieces (4)
    "T_UP": T_UP,
    "T_DOWN": T_DOWN,
    "T_LEFT": T_LEFT,
    "T_RIGHT": T_RIGHT,
    
    # S-pieces (2)
    "S_H": S_H,
    "S_V": S_V,
    
    # Z-pieces (2)
    "Z_H": Z_H,
    "Z_V": Z_V,
    
    # L-pieces (4)
    "L_1": L_1,
    "L_2": L_2,
    "L_3": L_3,
    "L_4": L_4,
    
    # J-pieces (4)
    "J_1": J_1,
    "J_2": J_2,
    "J_3": J_3,
    "J_4": J_4,
    
    # Rectangles (2)
    "RECT_2x3_H": RECT_2x3_H,
    "RECT_2x3_V": RECT_2x3_V,
    
    # Large square (1)
    "SQUARE_3x3": SQUARE_3x3,
}

# List for indexed access (useful for random selection)
PIECE_LIST: List[Piece] = list(PIECES.values())
PIECE_NAMES: List[str] = list(PIECES.keys())
NUM_PIECES: int = len(PIECES)

# Verify we have exactly 37 pieces
# Categories: Single(1) + Domino(2) + Diag2(2) + Trio(2) + Diag3(2) + TrioL(4) + 
#             I4(2) + I5(2) + O(1) + T(4) + S(2) + Z(2) + L(4) + J(4) + Rect(2) + Square(1) = 37
assert NUM_PIECES == 37, f"Expected 37 pieces, got {NUM_PIECES}"


def get_piece_by_name(name: str) -> Piece:
    """Get a piece by its name."""
    if name not in PIECES:
        raise ValueError(f"Unknown piece: {name}. Valid pieces: {list(PIECES.keys())}")
    return PIECES[name]


def get_piece_by_index(index: int) -> Piece:
    """Get a piece by its index (0-40)."""
    if not 0 <= index < NUM_PIECES:
        raise ValueError(f"Piece index must be 0-{NUM_PIECES-1}, got {index}")
    return PIECE_LIST[index]


def get_piece_index(piece: Piece) -> int:
    """Get the index of a piece."""
    return PIECE_LIST.index(piece)


def get_all_pieces() -> List[Piece]:
    """Get a list of all pieces."""
    return PIECE_LIST.copy()


def get_random_pieces(n: int = 3, rng: np.random.Generator = None) -> List[Piece]:
    """Get n random pieces (default 3 as per game rules)."""
    if rng is None:
        rng = np.random.default_rng()
    indices = rng.choice(NUM_PIECES, size=n, replace=True)
    return [PIECE_LIST[i] for i in indices]


def piece_to_one_hot(piece: Piece) -> np.ndarray:
    """Convert a piece to a one-hot encoding."""
    encoding = np.zeros(NUM_PIECES, dtype=np.float32)
    idx = PIECE_LIST.index(piece)
    encoding[idx] = 1.0
    return encoding


def visualize_piece(piece: Piece) -> str:
    """Create a string visualization of a piece."""
    arr = piece.get_shape_array()
    lines = []
    for row in arr:
        line = "".join("□" if cell else " " for cell in row)
        lines.append(line)
    return "\n".join(lines)


if __name__ == "__main__":
    # Print all pieces for verification
    print(f"Total pieces: {NUM_PIECES}")
    print("-" * 40)
    for name, piece in PIECES.items():
        print(f"\n{name} ({piece.num_blocks} blocks, {piece.width}x{piece.height}):")
        print(visualize_piece(piece))
