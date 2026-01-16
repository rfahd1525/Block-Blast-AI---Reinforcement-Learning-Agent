"""
Block Blast Board Module.

This module implements the game board with:
- 8x8 grid representation
- Piece placement validation
- Line clearing logic (rows and columns)
- Combo detection
- Hole detection for reward shaping
"""
from typing import List, Tuple, Set, Optional
import numpy as np
from .pieces import Piece


class Board:
    """
    Represents the 8x8 Block Blast game board.
    
    The board is represented as a 2D numpy array where:
    - 0 = empty cell
    - 1 = filled cell
    """
    
    DEFAULT_SIZE = 8
    
    def __init__(self, size: int = DEFAULT_SIZE):
        """Initialize an empty board."""
        self.size = size
        self.grid = np.zeros((size, size), dtype=np.int8)
        self._total_blocks = 0
    
    def copy(self) -> "Board":
        """Create a deep copy of this board."""
        new_board = Board(self.size)
        new_board.grid = self.grid.copy()
        new_board._total_blocks = self._total_blocks
        return new_board
    
    def reset(self) -> None:
        """Clear the board."""
        self.grid.fill(0)
        self._total_blocks = 0
    
    @property
    def total_blocks(self) -> int:
        """Return total number of filled blocks on the board."""
        return self._total_blocks
    
    @property
    def empty_cells(self) -> int:
        """Return number of empty cells."""
        return self.size * self.size - self._total_blocks
    
    def get_cell(self, row: int, col: int) -> int:
        """Get the value of a cell."""
        return self.grid[row, col]
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if a cell is empty."""
        return self.grid[row, col] == 0
    
    def is_filled(self, row: int, col: int) -> bool:
        """Check if a cell is filled."""
        return self.grid[row, col] == 1
    
    def in_bounds(self, row: int, col: int) -> bool:
        """Check if coordinates are within board bounds."""
        return 0 <= row < self.size and 0 <= col < self.size
    
    def can_place(self, piece: Piece, row: int, col: int) -> bool:
        """
        Check if a piece can be placed at the given position.
        
        Args:
            piece: The piece to place
            row: Row position for the piece's top-left anchor
            col: Column position for the piece's top-left anchor
            
        Returns:
            True if the piece can be placed, False otherwise
        """
        size = self.size
        grid = self.grid
        for dr, dc in piece.blocks:
            r, c = row + dr, col + dc
            # Check bounds (inlined for speed)
            if r < 0 or r >= size or c < 0 or c >= size:
                return False
            # Check collision
            if grid[r, c] != 0:
                return False
        return True
    
    def place_piece(self, piece: Piece, row: int, col: int) -> bool:
        """
        Place a piece on the board.
        
        Args:
            piece: The piece to place
            row: Row position for the piece's top-left anchor
            col: Column position for the piece's top-left anchor
            
        Returns:
            True if successful, False if placement is invalid
        """
        if not self.can_place(piece, row, col):
            return False
        
        for dr, dc in piece.blocks:
            r, c = row + dr, col + dc
            self.grid[r, c] = 1
            self._total_blocks += 1
        
        return True
    
    def get_valid_placements(self, piece: Piece) -> List[Tuple[int, int]]:
        """
        Get all valid placement positions for a piece.
        
        Returns:
            List of (row, col) tuples where the piece can be placed
        """
        valid = []
        # Limit search range based on piece dimensions
        max_row = self.size - piece.height + 1
        max_col = self.size - piece.width + 1
        for row in range(max_row):
            for col in range(max_col):
                if self.can_place(piece, row, col):
                    valid.append((row, col))
        return valid
    
    def has_valid_placement(self, piece: Piece) -> bool:
        """Check if there's at least one valid placement for a piece."""
        max_row = self.size - piece.height + 1
        max_col = self.size - piece.width + 1
        for row in range(max_row):
            for col in range(max_col):
                if self.can_place(piece, row, col):
                    return True
        return False
    
    def find_complete_lines(self) -> Tuple[Set[int], Set[int]]:
        """
        Find all complete rows and columns.
        
        Returns:
            Tuple of (complete_rows, complete_cols) as sets of indices
        """
        complete_rows = set()
        complete_cols = set()
        
        # Check rows
        for row in range(self.size):
            if np.all(self.grid[row, :] == 1):
                complete_rows.add(row)
        
        # Check columns
        for col in range(self.size):
            if np.all(self.grid[:, col] == 1):
                complete_cols.add(col)
        
        return complete_rows, complete_cols
    
    def clear_lines(self) -> Tuple[int, int]:
        """
        Clear all complete rows and columns.
        
        Returns:
            Tuple of (num_rows_cleared, num_cols_cleared)
        """
        complete_rows, complete_cols = self.find_complete_lines()
        
        # Count blocks that will be cleared
        blocks_cleared = 0
        
        # Clear rows
        for row in complete_rows:
            blocks_cleared += np.sum(self.grid[row, :])
            self.grid[row, :] = 0
        
        # Clear columns (may overlap with rows at intersection)
        for col in complete_cols:
            # Only count blocks not already cleared by row clearing
            for row in range(self.size):
                if row not in complete_rows and self.grid[row, col] == 1:
                    blocks_cleared += 1
            self.grid[:, col] = 0
        
        self._total_blocks -= blocks_cleared
        
        return len(complete_rows), len(complete_cols)
    
    def count_holes(self) -> int:
        """
        Count isolated empty cells (holes) on the board.
        
        A hole is an empty cell where all 4 orthogonal neighbors are either:
        - Filled, or
        - Out of bounds
        
        These are difficult or impossible to fill with most pieces.
        """
        holes = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:  # Empty cell
                    neighbors_blocked = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if not self.in_bounds(nr, nc) or self.grid[nr, nc] == 1:
                            neighbors_blocked += 1
                    if neighbors_blocked == 4:
                        holes += 1
        return holes
    
    def count_potential_holes(self) -> int:
        """
        Count empty cells that are mostly surrounded (3 neighbors blocked).
        These are likely to become holes.
        """
        potential_holes = 0
        for row in range(self.size):
            for col in range(self.size):
                if self.grid[row, col] == 0:
                    neighbors_blocked = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = row + dr, col + dc
                        if not self.in_bounds(nr, nc) or self.grid[nr, nc] == 1:
                            neighbors_blocked += 1
                    if neighbors_blocked >= 3:
                        potential_holes += 1
        return potential_holes
    
    def get_center_openness(self) -> float:
        """
        Calculate how open the center of the board is (important for strategy).
        Returns a value from 0.0 (all center cells filled) to 1.0 (all empty).
        """
        # Define center as the inner 4x4 grid (rows 2-5, cols 2-5)
        center = self.grid[2:6, 2:6]
        return 1.0 - (np.sum(center) / 16.0)
    
    def get_height_map(self) -> np.ndarray:
        """
        Get the "height" of each column (topmost filled cell).
        Useful for heuristic evaluation.
        """
        heights = np.zeros(self.size, dtype=np.int32)
        for col in range(self.size):
            for row in range(self.size):
                if self.grid[row, col] == 1:
                    heights[col] = self.size - row
                    break
        return heights
    
    def get_bumpiness(self) -> int:
        """
        Calculate the bumpiness (sum of absolute height differences between adjacent columns).
        Lower bumpiness generally means a more playable board.
        """
        heights = self.get_height_map()
        bumpiness = 0
        for i in range(len(heights) - 1):
            bumpiness += abs(heights[i] - heights[i + 1])
        return bumpiness
    
    def get_state(self) -> np.ndarray:
        """Get the board state as a numpy array."""
        return self.grid.copy()
    
    def set_state(self, state: np.ndarray) -> None:
        """Set the board state from a numpy array."""
        self.grid = state.copy()
        self._total_blocks = int(np.sum(state))
    
    def to_tensor(self) -> np.ndarray:
        """Convert board to a float tensor for neural network input."""
        return self.grid.astype(np.float32)
    
    def __str__(self) -> str:
        """Create a string visualization of the board."""
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(self.size)))
        lines.append("  " + "-" * (self.size * 2 - 1))
        for row in range(self.size):
            row_str = f"{row}|"
            for col in range(self.size):
                cell = "█" if self.grid[row, col] == 1 else "·"
                row_str += cell + " "
            lines.append(row_str.rstrip())
        lines.append("  " + "-" * (self.size * 2 - 1))
        lines.append(f"Blocks: {self._total_blocks}, Empty: {self.empty_cells}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"Board(size={self.size}, blocks={self._total_blocks})"
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Board):
            return False
        return np.array_equal(self.grid, other.grid)
    
    def __hash__(self) -> int:
        return hash(self.grid.tobytes())


if __name__ == "__main__":
    # Test the board
    from .pieces import SINGLE, DOMINO_H, TRIO_H, I_H, SQUARE_3x3
    
    board = Board()
    print("Empty board:")
    print(board)
    print()
    
    # Place some pieces
    board.place_piece(TRIO_H, 0, 0)
    board.place_piece(TRIO_H, 0, 3)
    board.place_piece(DOMINO_H, 0, 6)
    print("After placing pieces to fill row 0:")
    print(board)
    
    # Clear lines
    rows, cols = board.clear_lines()
    print(f"\nCleared {rows} rows and {cols} columns")
    print(board)
