"""
Interactive play script for Block Blast.

Allows watching the AI play or playing manually.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from game.engine import GameEngine
from game.renderer import Renderer
from agents.ppo import PPOAgent
from utils.device import get_device


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def watch_ai_play(
    agent: PPOAgent,
    num_games: int = 1,
    delay: float = 0.5,
    deterministic: bool = True,
    seed: int = 42,
) -> None:
    """
    Watch the AI play Block Blast.
    
    Args:
        agent: Trained agent
        num_games: Number of games to play
        delay: Delay between moves (seconds)
        deterministic: Whether to use deterministic actions
        seed: Random seed
    """
    from environment.block_blast_env import BlockBlastEnv
    
    agent.eval()
    
    for game_num in range(num_games):
        env = BlockBlastEnv(seed=seed + game_num)
        obs, info = env.reset()
        
        print(f"\n{'='*60}")
        print(f"Game {game_num + 1}/{num_games}")
        print(f"{'='*60}\n")
        
        done = False
        total_reward = 0
        step = 0
        
        while not done:
            clear_screen()
            print(f"Game {game_num + 1}/{num_games} | Step {step}")
            print(env.render())
            
            action, action_info = agent.select_action(obs, deterministic=deterministic)
            
            piece_idx = action // 64
            pos = action % 64
            row, col = pos // 8, pos % 8
            print(f"\nAI selects: Piece {piece_idx} at ({row}, {col})")
            print(f"Value estimate: {action_info['value']:.2f}")
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            step += 1
            
            if info.get('last_move', {}).get('lines_cleared', 0) > 0:
                lm = info['last_move']
                print(f"\n*** Cleared {lm['lines_cleared']} lines! "
                      f"Combo x{lm['combo_multiplier']} "
                      f"+{lm['score_gained']} points ***")
                time.sleep(delay * 2)
            
            time.sleep(delay)
        
        clear_screen()
        print(f"\n{'='*60}")
        print("GAME OVER!")
        print(f"{'='*60}")
        print(f"Final Score: {info['score']:,}")
        print(f"Moves: {info['moves']}")
        print(f"Lines Cleared: {info['lines_cleared']}")
        print(f"Max Combo: {info['max_combo']}")
        print(f"Total Reward: {total_reward:.1f}")
        print(f"{'='*60}\n")
        
        env.close()
        
        if game_num < num_games - 1:
            input("Press Enter for next game...")


def play_manual(seed: int = 42) -> None:
    """
    Play Block Blast manually in the terminal.
    
    Args:
        seed: Random seed
    """
    engine = GameEngine(seed=seed)
    renderer = Renderer()
    
    print("\n" + "="*60)
    print("BLOCK BLAST - Manual Play")
    print("="*60)
    print("\nControls:")
    print("  Enter move as: piece_idx row col (e.g., '0 3 4')")
    print("  Type 'q' to quit")
    print("  Type 'r' to restart")
    print("="*60 + "\n")
    
    while True:
        # Show current state
        clear_screen()
        print(renderer.render_game_state(
            engine.board,
            engine.current_pieces,
            engine.pieces_used,
            engine.score,
            engine.combo_count,
            engine.moves_made,
        ))
        
        if engine.is_game_over():
            print("\n*** GAME OVER! ***")
            print(f"Final Score: {engine.score:,}")
            print(f"Moves: {engine.moves_made}")
            print(f"Max Combo: {engine.max_combo}")
            
            action = input("\nPlay again? (y/n): ").strip().lower()
            if action == 'y':
                engine.reset()
                continue
            else:
                break
        
        # Get valid moves
        valid_moves = engine.get_valid_moves()
        if not valid_moves:
            continue
        
        # Show valid placements for each piece
        print("\nValid placements per piece:")
        for i in range(3):
            if not engine.pieces_used[i]:
                placements = [f"({r},{c})" for p, r, c in valid_moves if p == i]
                if placements:
                    print(f"  Piece {i}: {len(placements)} positions")
        
        # Get user input
        try:
            user_input = input("\nEnter move (piece row col): ").strip().lower()
            
            if user_input == 'q':
                print("Thanks for playing!")
                break
            elif user_input == 'r':
                engine.reset()
                continue
            
            parts = user_input.split()
            if len(parts) != 3:
                print("Invalid input. Use format: piece row col")
                time.sleep(1)
                continue
            
            piece_idx, row, col = int(parts[0]), int(parts[1]), int(parts[2])
            
            if not engine.can_place_piece(piece_idx, row, col):
                print("Invalid move! Try again.")
                time.sleep(1)
                continue
            
            result = engine.make_move(piece_idx, row, col)
            
            if result.lines_cleared > 0:
                print(f"\n*** Cleared {result.lines_cleared} lines! "
                      f"Combo x{result.combo_multiplier} "
                      f"+{result.score_gained} points ***")
                time.sleep(1)
        
        except (ValueError, IndexError):
            print("Invalid input. Use format: piece row col")
            time.sleep(1)


def play_random(num_games: int = 10, seed: int = 42) -> None:
    """
    Play random games and show statistics.
    
    Args:
        num_games: Number of games to play
        seed: Random seed
    """
    from game.engine import play_random_game
    
    print(f"\nPlaying {num_games} random games...")
    
    scores = []
    moves = []
    lines = []
    
    for i in range(num_games):
        stats = play_random_game(seed=seed + i)
        scores.append(stats['score'])
        moves.append(stats['moves_made'])
        lines.append(stats['total_lines_cleared'])
        
        print(f"Game {i+1}: Score={stats['score']:,}, "
              f"Moves={stats['moves_made']}, "
              f"Lines={stats['total_lines_cleared']}")
    
    print("\n" + "="*60)
    print("RANDOM AGENT STATISTICS")
    print("="*60)
    print(f"Games: {num_games}")
    print(f"Mean Score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
    print(f"Max Score: {max(scores)}")
    print(f"Mean Moves: {np.mean(moves):.1f}")
    print(f"Mean Lines: {np.mean(lines):.1f}")
    print("="*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Play Block Blast")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["watch", "manual", "random"],
        default="watch",
        help="Play mode: watch AI, play manually, or watch random agent"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for watch mode)"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=1,
        help="Number of games to play"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between moves (seconds) for watch mode"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for AI"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.mode == "watch":
        if args.checkpoint is None:
            print("Error: --checkpoint required for watch mode")
            print("Use --mode random to watch a random agent instead")
            sys.exit(1)
        
        if not os.path.exists(args.checkpoint):
            print(f"Checkpoint not found: {args.checkpoint}")
            sys.exit(1)
        
        device = get_device()
        agent = PPOAgent(device=device)
        agent.load(args.checkpoint)
        print(f"Loaded model from {args.checkpoint}")
        
        watch_ai_play(
            agent,
            num_games=args.games,
            delay=args.delay,
            deterministic=args.deterministic,
            seed=args.seed,
        )
    
    elif args.mode == "manual":
        play_manual(seed=args.seed)
    
    elif args.mode == "random":
        play_random(num_games=args.games, seed=args.seed)


if __name__ == "__main__":
    main()
