"""
Evaluation script for Block Blast AI.

Evaluates trained models and generates performance statistics.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from environment.block_blast_env import BlockBlastEnv
from agents.ppo import PPOAgent
from utils.device import get_device


def evaluate_agent(
    agent: PPOAgent,
    num_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        agent: Trained agent to evaluate
        num_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        render: Whether to render the game
        seed: Random seed
        
    Returns:
        Dictionary of evaluation statistics
    """
    agent.eval()
    
    env = BlockBlastEnv(
        render_mode="human" if render else None,
        seed=seed,
    )
    
    scores = []
    lengths = []
    lines_cleared = []
    max_combos = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=seed + episode)
        episode_score = 0
        episode_length = 0
        done = False
        
        while not done:
            action, _ = agent.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_length += 1
            
            if done:
                episode_score = info.get('score', 0)
        
        scores.append(episode_score)
        lengths.append(episode_length)
        lines_cleared.append(info.get('lines_cleared', 0))
        max_combos.append(info.get('max_combo', 0))
    
    env.close()
    
    return {
        'num_episodes': num_episodes,
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'median_score': np.median(scores),
        'mean_length': np.mean(lengths),
        'std_length': np.std(lengths),
        'mean_lines_cleared': np.mean(lines_cleared),
        'mean_max_combo': np.mean(max_combos),
        'scores': scores,
        'lengths': lengths,
    }


def print_results(results: Dict[str, Any]) -> None:
    """Print evaluation results."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {results['num_episodes']}")
    print()
    print("Score Statistics:")
    print(f"  Mean:   {results['mean_score']:.1f} ± {results['std_score']:.1f}")
    print(f"  Median: {results['median_score']:.1f}")
    print(f"  Min:    {results['min_score']:.1f}")
    print(f"  Max:    {results['max_score']:.1f}")
    print()
    print("Game Statistics:")
    print(f"  Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Mean lines cleared: {results['mean_lines_cleared']:.1f}")
    print(f"  Mean max combo: {results['mean_max_combo']:.1f}")
    print("=" * 60)
    
    # Score distribution
    scores = results['scores']
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nScore Percentiles:")
    for p in percentiles:
        print(f"  {p}th: {np.percentile(scores, p):.1f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Block Blast AI")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load model
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    device = get_device()
    agent = PPOAgent(device=device)
    agent.load(args.checkpoint)
    print(f"Loaded model from {args.checkpoint}")
    
    # Evaluate
    results = evaluate_agent(
        agent,
        num_episodes=args.episodes,
        deterministic=args.deterministic,
        render=args.render,
        seed=args.seed,
    )
    
    # Print results
    print_results(results)
    
    # Save results
    if args.output:
        import json
        # Convert numpy arrays to lists for JSON serialization
        save_results = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                       for k, v in results.items()}
        with open(args.output, 'w') as f:
            json.dump(save_results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
