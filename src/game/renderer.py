"""
Block Blast Board Renderer.

Provides visualization utilities for the game board.
"""
from typing import Optional, List
import numpy as np

from .board import Board
from .pieces import Piece


class Renderer:
    """
    ASCII renderer for Block Blast game.
    
    Can be extended for graphical rendering if needed.
    """
    
    # ASCII characters for rendering
    EMPTY = "·"
    FILLED = "█"
    PIECE_PREVIEW = "□"
    
    def __init__(self, board_size: int = 8):
        """Initialize renderer."""
        self.board_size = board_size
    
    def render_board(self, board: Board, show_coords: bool = True) -> str:
        """
        Render the board as ASCII art.
        
        Args:
            board: The board to render
            show_coords: Whether to show row/column numbers
            
        Returns:
            String representation of the board
        """
        lines = []
        
        if show_coords:
            # Column headers
            header = "  " + " ".join(str(i) for i in range(board.size))
            lines.append(header)
            lines.append("  " + "-" * (board.size * 2 - 1))
        
        for row in range(board.size):
            if show_coords:
                row_str = f"{row}|"
            else:
                row_str = ""
            
            for col in range(board.size):
                cell = self.FILLED if board.is_filled(row, col) else self.EMPTY
                row_str += cell
                if col < board.size - 1:
                    row_str += " "
            
            lines.append(row_str)
        
        if show_coords:
            lines.append("  " + "-" * (board.size * 2 - 1))
        
        return "\n".join(lines)
    
    def render_piece(self, piece: Piece) -> str:
        """
        Render a piece as ASCII art.
        
        Args:
            piece: The piece to render
            
        Returns:
            String representation of the piece
        """
        arr = piece.get_shape_array()
        lines = []
        for row in arr:
            line = " ".join(self.PIECE_PREVIEW if cell else " " for cell in row)
            lines.append(line)
        return "\n".join(lines)
    
    def render_pieces(self, pieces: List[Piece], used: List[bool] = None) -> str:
        """
        Render multiple pieces side by side.
        
        Args:
            pieces: List of pieces to render
            used: Optional list indicating which pieces have been used
            
        Returns:
            String representation of all pieces
        """
        if used is None:
            used = [False] * len(pieces)
        
        # Get piece arrays and find max height
        arrays = [p.get_shape_array() for p in pieces]
        max_height = max(arr.shape[0] for arr in arrays)
        max_width = max(arr.shape[1] for arr in arrays)
        
        lines = []
        
        # Headers
        headers = []
        for i, piece in enumerate(pieces):
            status = "(used)" if used[i] else f"[{i}]"
            name = piece.name[:max_width*2]
            headers.append(f"{status} {name}".ljust(max_width * 2 + 4))
        lines.append("  ".join(headers))
        
        # Piece shapes
        for row in range(max_height):
            row_parts = []
            for i, arr in enumerate(arrays):
                if row < arr.shape[0]:
                    part = " ".join(
                        self.PIECE_PREVIEW if arr[row, c] else " " 
                        for c in range(arr.shape[1])
                    )
                else:
                    part = " " * (arr.shape[1] * 2 - 1)
                row_parts.append(part.ljust(max_width * 2 + 4))
            lines.append("  ".join(row_parts))
        
        return "\n".join(lines)
    
    def render_board_with_placement(
        self, 
        board: Board, 
        piece: Piece, 
        row: int, 
        col: int,
        valid: bool = True
    ) -> str:
        """
        Render the board with a piece placement preview.
        
        Args:
            board: The current board
            piece: The piece to preview
            row: Row position for placement
            col: Column position for placement
            valid: Whether the placement is valid
            
        Returns:
            String with board and placement preview
        """
        lines = []
        lines.append("  " + " ".join(str(i) for i in range(board.size)))
        lines.append("  " + "-" * (board.size * 2 - 1))
        
        # Create preview grid
        preview = board.grid.copy()
        preview_marker = 2 if valid else 3  # Different marker for valid/invalid
        
        for dr, dc in piece.blocks:
            r, c = row + dr, col + dc
            if 0 <= r < board.size and 0 <= c < board.size:
                if preview[r, c] == 0:
                    preview[r, c] = preview_marker
                else:
                    preview[r, c] = 3  # Collision
        
        for row_idx in range(board.size):
            row_str = f"{row_idx}|"
            for col_idx in range(board.size):
                cell_val = preview[row_idx, col_idx]
                if cell_val == 0:
                    cell = self.EMPTY
                elif cell_val == 1:
                    cell = self.FILLED
                elif cell_val == 2:
                    cell = "○"  # Valid placement preview
                else:
                    cell = "✗"  # Collision
                row_str += cell + " "
            lines.append(row_str.rstrip())
        
        lines.append("  " + "-" * (board.size * 2 - 1))
        
        return "\n".join(lines)
    
    def render_game_state(
        self,
        board: Board,
        pieces: List[Piece],
        pieces_used: List[bool],
        score: int,
        combo: int,
        moves: int
    ) -> str:
        """
        Render complete game state.
        
        Args:
            board: The game board
            pieces: Current pieces
            pieces_used: Which pieces have been used
            score: Current score
            combo: Current combo count
            moves: Number of moves made
            
        Returns:
            Complete game state visualization
        """
        lines = []
        lines.append("=" * 40)
        lines.append(f"Score: {score:,}  |  Moves: {moves}  |  Combo: x{combo + 1}")
        lines.append("=" * 40)
        lines.append("")
        lines.append(self.render_board(board))
        lines.append("")
        lines.append("Available Pieces:")
        lines.append(self.render_pieces(pieces, pieces_used))
        lines.append("=" * 40)
        
        return "\n".join(lines)


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":
    from .board import Board
    from .pieces import TRIO_H, L_1, SQUARE_3x3
    
    renderer = Renderer()
    board = Board()
    
    # Place some pieces
    board.place_piece(TRIO_H, 0, 0)
    board.place_piece(L_1, 2, 2)
    
    print("Board:")
    print(renderer.render_board(board))
    print()
    
    print("Pieces:")
    pieces = [TRIO_H, L_1, SQUARE_3x3]
    used = [True, True, False]
    print(renderer.render_pieces(pieces, used))
    print()
    
    print("Placement preview:")
    print(renderer.render_board_with_placement(board, SQUARE_3x3, 4, 4, True))
