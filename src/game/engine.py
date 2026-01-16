"""
Block Blast Game Engine.

This module implements the complete game logic including:
- Game state management
- Piece generation (3 random pieces per turn)
- Scoring with combo system
- Game over detection
- Move validation
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np

from .board import Board
from .pieces import (
    Piece, PIECE_LIST, NUM_PIECES, 
    get_random_pieces, get_piece_by_index, get_piece_index
)


class GameStatus(Enum):
    """Game status enumeration."""
    PLAYING = "playing"
    GAME_OVER = "game_over"


@dataclass
class MoveResult:
    """Result of a move action."""
    success: bool
    piece_placed: Optional[Piece] = None
    position: Optional[Tuple[int, int]] = None
    blocks_placed: int = 0
    rows_cleared: int = 0
    cols_cleared: int = 0
    lines_cleared: int = 0
    combo_multiplier: int = 1
    score_gained: int = 0
    game_over: bool = False


@dataclass
class GameState:
    """Complete game state for serialization and RL."""
    board: np.ndarray
    current_pieces: List[int]  # Indices of available pieces
    pieces_used: List[bool]  # Which pieces have been used this turn
    score: int
    combo_count: int
    moves_made: int
    status: GameStatus
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "board": self.board.tolist(),
            "current_pieces": self.current_pieces,
            "pieces_used": self.pieces_used,
            "score": self.score,
            "combo_count": self.combo_count,
            "moves_made": self.moves_made,
            "status": self.status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        """Create from dictionary."""
        return cls(
            board=np.array(data["board"], dtype=np.int8),
            current_pieces=data["current_pieces"],
            pieces_used=data["pieces_used"],
            score=data["score"],
            combo_count=data["combo_count"],
            moves_made=data["moves_made"],
            status=GameStatus(data["status"]),
        )


class GameEngine:
    """
    Block Blast game engine.
    
    Manages the complete game state and provides methods for:
    - Making moves
    - Checking valid actions
    - Computing scores
    - Detecting game over
    """
    
    PIECES_PER_TURN = 3
    # Real Block Blast scoring constants
    PLACEMENT_SCORE_PER_BLOCK = 1  # +1 point per block placed
    BASE_SCORE_PER_BLOCK = 10  # 10 points per block cleared in lines
    MAX_COMBO_MULTIPLIER = 4  # Simultaneous line clear multiplier cap (1x, 2x, 3x, 4x)
    MAX_STREAK_MULTIPLIER = 8  # Consecutive line clear multiplier cap (1x to 8x)
    
    def __init__(self, board_size: int = 8, seed: Optional[int] = None):
        """
        Initialize a new game.
        
        Args:
            board_size: Size of the square board (default 8)
            seed: Random seed for reproducibility
        """
        self.board_size = board_size
        self.board = Board(board_size)
        self.rng = np.random.default_rng(seed)
        
        # Game state
        self.current_pieces: List[Piece] = []
        self.pieces_used: List[bool] = [False, False, False]
        self.score = 0
        self.combo_count = 0  # Tracks consecutive line clears (streak)
        self.moves_made = 0
        self.total_lines_cleared = 0
        self.status = GameStatus.PLAYING
        
        # Statistics
        self.max_combo = 0
        self.total_blocks_placed = 0
        
        # Generate initial pieces
        self._generate_new_pieces()
    
    def reset(self, seed: Optional[int] = None) -> GameState:
        """
        Reset the game to initial state.
        
        Args:
            seed: New random seed (optional)
            
        Returns:
            Initial game state
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        
        self.board.reset()
        self.current_pieces = []
        self.pieces_used = [False, False, False]
        self.score = 0
        self.combo_count = 0
        self.moves_made = 0
        self.total_lines_cleared = 0
        self.status = GameStatus.PLAYING
        self.max_combo = 0
        self.total_blocks_placed = 0
        
        self._generate_new_pieces()
        
        return self.get_state()
    
    def _generate_new_pieces(self) -> None:
        """Generate 3 new random pieces for the turn.
        
        Continues regenerating until ALL pieces can be placed in some order,
        ensuring the player can always place every piece if they play correctly.
        """
        max_attempts = 100  # Prevent infinite loops
        
        for attempt in range(max_attempts):
            self.current_pieces = get_random_pieces(self.PIECES_PER_TURN, self.rng)
            self.pieces_used = [False, False, False]
            
            # Check if all pieces can be placed in some order
            if self._can_place_all_pieces():
                return  # Valid pieces found
        
        # If we exhausted attempts, the board is too full for 3 pieces
        # Keep the last generated pieces (game will end naturally)
    
    def _can_place_all_pieces(self) -> bool:
        """Check if all current pieces can be placed in some order."""
        return self._can_place_remaining(
            self.board.grid.copy(),
            [False, False, False]
        )
    
    def _can_place_remaining(self, board_state: np.ndarray, used: list) -> bool:
        """Recursively check if remaining pieces can all be placed."""
        # If all pieces are used, we succeeded
        if all(used):
            return True
        
        # Try each unused piece
        for idx in range(self.PIECES_PER_TURN):
            if used[idx]:
                continue
            
            piece = self.current_pieces[idx]
            
            # Try all positions for this piece
            for row in range(self.board_size - piece.height + 1):
                for col in range(self.board_size - piece.width + 1):
                    # Check if piece can be placed here
                    can_place = True
                    for dr, dc in piece.blocks:
                        r, c = row + dr, col + dc
                        if r < 0 or r >= self.board_size or c < 0 or c >= self.board_size:
                            can_place = False
                            break
                        if board_state[r, c] != 0:
                            can_place = False
                            break
                    
                    if can_place:
                        # Simulate placing the piece
                        new_board = board_state.copy()
                        for dr, dc in piece.blocks:
                            new_board[row + dr, col + dc] = 1
                        
                        # Simulate line clears
                        new_board = self._simulate_line_clears(new_board)
                        
                        # Mark piece as used and recurse
                        new_used = used.copy()
                        new_used[idx] = True
                        
                        if self._can_place_remaining(new_board, new_used):
                            return True
        
        return False
    
    def _simulate_line_clears(self, board: np.ndarray) -> np.ndarray:
        """Simulate line clears on a board state."""
        # Find complete rows and columns
        complete_rows = [r for r in range(self.board_size) if np.all(board[r, :] == 1)]
        complete_cols = [c for c in range(self.board_size) if np.all(board[:, c] == 1)]
        
        # Clear them
        for r in complete_rows:
            board[r, :] = 0
        for c in complete_cols:
            board[:, c] = 0
        
        return board
    
    def _calculate_combo_multiplier(self, lines_cleared: int) -> int:
        """
        Calculate the combo multiplier based on simultaneous lines cleared.
        
        This multiplier rewards clearing multiple lines in a single move:
        - 1 line: 1x
        - 2 lines: 2x
        - 3 lines: 3x
        - 4+ lines: 4x (capped)
        """
        return min(lines_cleared, self.MAX_COMBO_MULTIPLIER)
    
    def _calculate_streak_multiplier(self) -> int:
        """
        Calculate the streak multiplier based on consecutive line clears.
        
        This multiplier rewards clearing lines on consecutive moves:
        - 1st consecutive clear: 1x
        - 2nd consecutive clear: 2x
        - ... up to 8x maximum
        """
        return min(self.combo_count + 1, self.MAX_STREAK_MULTIPLIER)
    
    def _calculate_placement_score(self, blocks_placed: int) -> int:
        """
        Calculate placement score for placing blocks.
        
        You get points simply for placing a block on the grid,
        regardless of whether you clear a line.
        
        Math: +1 point per block square placed.
        """
        return blocks_placed * self.PLACEMENT_SCORE_PER_BLOCK
    
    def _calculate_score(self, lines_cleared: int, blocks_in_lines: int, blocks_placed: int) -> int:
        """
        Calculate total score for a move using the real Block Blast formula.
        
        Formula:
            Total Score = (Placement Score) + [(Base Line Score) × (Combo Multiplier) × (Streak Multiplier)]
        
        Where:
            - Placement Score: +1 point per block placed (always awarded)
            - Base Line Score: 10 points per block cleared in lines
            - Combo Multiplier: Based on simultaneous lines cleared (1x to 4x)
            - Streak Multiplier: Based on consecutive line clears (1x to 8x)
        
        Example calculations:
            - Placing a 2x2 square (4 blocks), no lines: 4 points
            - Clearing 1 row (8 blocks): 8 × 10 × 1 × streak = 80 × streak points
            - Clearing 2 rows at once (16 blocks): 16 × 10 × 2 × streak = 320 × streak points
        """
        # Placement score is always awarded
        placement_score = self._calculate_placement_score(blocks_placed)
        
        if lines_cleared == 0:
            return placement_score
        
        # Base line score: 10 points per block cleared
        base_line_score = blocks_in_lines * self.BASE_SCORE_PER_BLOCK
        
        # Combo multiplier: based on simultaneous lines cleared (1x, 2x, 3x, 4x)
        combo_multiplier = self._calculate_combo_multiplier(lines_cleared)
        
        # Streak multiplier: based on consecutive line clears (1x to 8x)
        streak_multiplier = self._calculate_streak_multiplier()
        
        # Apply the formula: Placement + (Base × Combo × Streak)
        line_clear_score = base_line_score * combo_multiplier * streak_multiplier
        
        total = placement_score + line_clear_score
        
        return total
    
    def get_available_piece_indices(self) -> List[int]:
        """Get indices of pieces that haven't been used this turn."""
        return [i for i in range(self.PIECES_PER_TURN) if not self.pieces_used[i]]
    
    def get_available_pieces(self) -> List[Tuple[int, Piece]]:
        """Get list of (index, piece) tuples for available pieces."""
        return [
            (i, self.current_pieces[i]) 
            for i in range(self.PIECES_PER_TURN) 
            if not self.pieces_used[i]
        ]
    
    def can_place_piece(self, piece_idx: int, row: int, col: int) -> bool:
        """
        Check if a piece can be placed at the given position.
        
        Args:
            piece_idx: Index of the piece (0, 1, or 2)
            row: Row to place piece at
            col: Column to place piece at
            
        Returns:
            True if the move is valid
        """
        if piece_idx < 0 or piece_idx >= self.PIECES_PER_TURN:
            return False
        if self.pieces_used[piece_idx]:
            return False
        if self.status == GameStatus.GAME_OVER:
            return False
        
        piece = self.current_pieces[piece_idx]
        return self.board.can_place(piece, row, col)
    
    def get_valid_moves(self) -> List[Tuple[int, int, int]]:
        """
        Get all valid moves as (piece_idx, row, col) tuples.
        
        Returns:
            List of all valid moves
        """
        valid_moves = []
        for piece_idx in self.get_available_piece_indices():
            piece = self.current_pieces[piece_idx]
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board.can_place(piece, row, col):
                        valid_moves.append((piece_idx, row, col))
        return valid_moves
    
    def get_action_mask(self) -> np.ndarray:
        """
        Get a mask of valid actions for the neural network.
        
        Returns:
            Boolean array of shape (3, 8, 8) where True = valid action
        """
        mask = np.zeros((self.PIECES_PER_TURN, self.board_size, self.board_size), dtype=bool)
        
        for piece_idx in self.get_available_piece_indices():
            piece = self.current_pieces[piece_idx]
            for row in range(self.board_size):
                for col in range(self.board_size):
                    if self.board.can_place(piece, row, col):
                        mask[piece_idx, row, col] = True
        
        return mask
    
    def has_valid_moves(self) -> bool:
        """Check if any valid moves exist."""
        for piece_idx in self.get_available_piece_indices():
            piece = self.current_pieces[piece_idx]
            if self.board.has_valid_placement(piece):
                return True
        return False
    
    def make_move(self, piece_idx: int, row: int, col: int) -> MoveResult:
        """
        Make a move by placing a piece at the specified position.
        
        Args:
            piece_idx: Index of the piece to place (0, 1, or 2)
            row: Row to place the piece at
            col: Column to place the piece at
            
        Returns:
            MoveResult with details about the move
        """
        # Validate move
        if not self.can_place_piece(piece_idx, row, col):
            return MoveResult(success=False)
        
        piece = self.current_pieces[piece_idx]
        
        # Place the piece
        self.board.place_piece(piece, row, col)
        self.pieces_used[piece_idx] = True
        self.moves_made += 1
        self.total_blocks_placed += piece.num_blocks
        
        # Check for line clears
        rows_cleared, cols_cleared = self.board.clear_lines()
        lines_cleared = rows_cleared + cols_cleared
        
        # Update combo
        if lines_cleared > 0:
            self.combo_count += 1
            self.max_combo = max(self.max_combo, self.combo_count)
            self.total_lines_cleared += lines_cleared
        else:
            self.combo_count = 0
        
        # Calculate score using the real Block Blast formula
        blocks_in_lines = lines_cleared * self.board_size  # Approximate
        score_gained = self._calculate_score(lines_cleared, blocks_in_lines, piece.num_blocks)
        self.score += score_gained
        
        # Check if all pieces are used
        all_used = all(self.pieces_used)
        
        # Check for game over
        if all_used:
            # Generate new pieces
            self._generate_new_pieces()
        
        # Check if any moves are possible
        if not self.has_valid_moves():
            self.status = GameStatus.GAME_OVER
        
        return MoveResult(
            success=True,
            piece_placed=piece,
            position=(row, col),
            blocks_placed=piece.num_blocks,
            rows_cleared=rows_cleared,
            cols_cleared=cols_cleared,
            lines_cleared=lines_cleared,
            combo_multiplier=self._calculate_combo_multiplier(lines_cleared) if lines_cleared > 0 else 1,
            score_gained=score_gained,
            game_over=self.status == GameStatus.GAME_OVER,
        )
    
    def get_state(self) -> GameState:
        """Get the current game state."""
        return GameState(
            board=self.board.get_state(),
            current_pieces=[get_piece_index(p) for p in self.current_pieces],
            pieces_used=self.pieces_used.copy(),
            score=self.score,
            combo_count=self.combo_count,
            moves_made=self.moves_made,
            status=self.status,
        )
    
    def set_state(self, state: GameState) -> None:
        """Restore game from a saved state."""
        self.board.set_state(state.board)
        self.current_pieces = [get_piece_by_index(i) for i in state.current_pieces]
        self.pieces_used = state.pieces_used.copy()
        self.score = state.score
        self.combo_count = state.combo_count
        self.moves_made = state.moves_made
        self.status = state.status
    
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get observation for the neural network.
        
        Returns:
            Dictionary with:
            - 'board': (8, 8) float array of board state
            - 'pieces': (3, 8, 8) float array of piece masks
            - 'pieces_used': (3,) bool array of which pieces are used
            - 'action_mask': (3, 8, 8) bool array of valid actions
        """
        # Board state
        board = self.board.to_tensor()
        
        # Piece masks (each piece rendered on an 8x8 grid)
        pieces = np.zeros((self.PIECES_PER_TURN, self.board_size, self.board_size), 
                         dtype=np.float32)
        for i, piece in enumerate(self.current_pieces):
            if not self.pieces_used[i]:
                pieces[i] = piece.to_mask(self.board_size)
        
        # Action mask
        action_mask = self.get_action_mask()
        
        return {
            'board': board,
            'pieces': pieces,
            'pieces_used': np.array(self.pieces_used, dtype=bool),
            'action_mask': action_mask,
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get game statistics."""
        return {
            'score': self.score,
            'moves_made': self.moves_made,
            'total_lines_cleared': self.total_lines_cleared,
            'max_combo': self.max_combo,
            'total_blocks_placed': self.total_blocks_placed,
            'board_fill_ratio': self.board.total_blocks / (self.board_size ** 2),
            'holes': self.board.count_holes(),
            'center_openness': self.board.get_center_openness(),
        }
    
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.status == GameStatus.GAME_OVER
    
    def __str__(self) -> str:
        """String representation of the game state."""
        lines = [str(self.board)]
        lines.append(f"\nScore: {self.score} | Moves: {self.moves_made} | "
                    f"Combo: {self.combo_count} | Status: {self.status.value}")
        lines.append("\nAvailable pieces:")
        for i, piece in enumerate(self.current_pieces):
            status = "USED" if self.pieces_used[i] else "available"
            lines.append(f"  [{i}] {piece.name} ({status})")
        return "\n".join(lines)


def play_random_game(seed: Optional[int] = None, verbose: bool = False) -> Dict[str, Any]:
    """
    Play a complete game with random moves for testing.
    
    Args:
        seed: Random seed
        verbose: Whether to print game progress
        
    Returns:
        Dictionary with game statistics
    """
    engine = GameEngine(seed=seed)
    
    if verbose:
        print("Starting random game...")
        print(engine)
    
    while not engine.is_game_over():
        valid_moves = engine.get_valid_moves()
        if not valid_moves:
            break
        
        # Choose random move
        move = valid_moves[engine.rng.choice(len(valid_moves))]
        result = engine.make_move(*move)
        
        if verbose and result.lines_cleared > 0:
            print(f"\nCleared {result.lines_cleared} lines! "
                  f"Combo x{result.combo_multiplier}, +{result.score_gained} points")
    
    stats = engine.get_statistics()
    
    if verbose:
        print("\n" + "=" * 40)
        print("GAME OVER!")
        print(engine)
        print(f"\nFinal Statistics: {stats}")
    
    return stats


if __name__ == "__main__":
    # Run a test game
    stats = play_random_game(seed=42, verbose=True)
    print(f"\nFinal score: {stats['score']}")
    
    # Benchmark
    import time
    print("\nRunning performance benchmark...")
    start = time.time()
    num_games = 100
    total_moves = 0
    for i in range(num_games):
        stats = play_random_game(seed=i)
        total_moves += stats['moves_made']
    elapsed = time.time() - start
    print(f"Played {num_games} games with {total_moves} total moves in {elapsed:.2f}s")
    print(f"Performance: {total_moves / elapsed:.0f} moves/second")
