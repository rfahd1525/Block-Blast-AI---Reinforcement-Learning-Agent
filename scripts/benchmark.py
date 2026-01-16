"""
Performance benchmark script for Block Blast.

Tests the speed of the game engine and environment.
"""
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_engine(num_games: int = 1000, seed: int = 42) -> Dict[str, float]:
    """
    Benchmark the game engine speed.
    
    Args:
        num_games: Number of games to play
        seed: Random seed
        
    Returns:
        Dictionary of benchmark results
    """
    from game.engine import GameEngine
    
    print(f"Benchmarking game engine with {num_games} games...")
    
    total_moves = 0
    total_time = 0
    
    for i in range(num_games):
        engine = GameEngine(seed=seed + i)
        
        start = time.perf_counter()
        while not engine.is_game_over():
            valid_moves = engine.get_valid_moves()
            if not valid_moves:
                break
            move = valid_moves[np.random.randint(len(valid_moves))]
            engine.make_move(*move)
            total_moves += 1
        total_time += time.perf_counter() - start
    
    return {
        'num_games': num_games,
        'total_moves': total_moves,
        'total_time': total_time,
        'moves_per_second': total_moves / total_time,
        'games_per_second': num_games / total_time,
        'avg_moves_per_game': total_moves / num_games,
    }


def benchmark_environment(num_steps: int = 100000, seed: int = 42) -> Dict[str, float]:
    """
    Benchmark the RL environment speed.
    
    Args:
        num_steps: Number of steps to take
        seed: Random seed
        
    Returns:
        Dictionary of benchmark results
    """
    from environment.block_blast_env import BlockBlastEnv
    
    print(f"Benchmarking environment with {num_steps} steps...")
    
    env = BlockBlastEnv(seed=seed)
    obs, _ = env.reset()
    
    start = time.perf_counter()
    steps = 0
    episodes = 0
    
    while steps < num_steps:
        action = env.sample_valid_action()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1
    
    total_time = time.perf_counter() - start
    env.close()
    
    return {
        'num_steps': num_steps,
        'num_episodes': episodes,
        'total_time': total_time,
        'steps_per_second': num_steps / total_time,
        'avg_episode_length': num_steps / episodes if episodes > 0 else 0,
    }


def benchmark_vectorized_env(
    num_envs: int = 64, 
    num_steps: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Benchmark the vectorized environment speed.
    
    Args:
        num_envs: Number of parallel environments
        num_steps: Number of steps per environment
        seed: Random seed
        
    Returns:
        Dictionary of benchmark results
    """
    from environment.wrappers import VectorizedBlockBlastEnv
    
    print(f"Benchmarking {num_envs} vectorized environments with {num_steps} steps each...")
    
    vec_env = VectorizedBlockBlastEnv(num_envs=num_envs, seed=seed)
    obs, _ = vec_env.reset()
    
    start = time.perf_counter()
    total_steps = 0
    episodes = 0
    
    while total_steps < num_steps * num_envs:
        actions = vec_env.sample_valid_actions()
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        total_steps += num_envs
        episodes += np.sum(terminated)
    
    total_time = time.perf_counter() - start
    vec_env.close()
    
    return {
        'num_envs': num_envs,
        'total_steps': total_steps,
        'num_episodes': episodes,
        'total_time': total_time,
        'steps_per_second': total_steps / total_time,
        'episodes_per_second': episodes / total_time,
    }


def benchmark_network(batch_sizes: list = [1, 8, 32, 128, 512], device_name: str = None) -> Dict[str, Any]:
    """
    Benchmark the neural network inference speed.
    
    Args:
        batch_sizes: List of batch sizes to test
        device_name: Device to use ('cpu', 'cuda', 'mps')
        
    Returns:
        Dictionary of benchmark results
    """
    import torch
    from models.network import BlockBlastNetwork
    from utils.device import get_device
    
    if device_name:
        device = torch.device(device_name)
    else:
        device = get_device()
    
    print(f"Benchmarking neural network on {device}...")
    
    network = BlockBlastNetwork().to(device)
    network.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create inputs
        board = torch.rand(batch_size, 8, 8).to(device)
        pieces = torch.rand(batch_size, 3, 8, 8).to(device)
        action_mask = torch.ones(batch_size, 192).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _, _ = network(board, pieces, action_mask)
        
        # Time inference
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        num_iters = 100
        start = time.perf_counter()
        
        with torch.no_grad():
            for _ in range(num_iters):
                _, _ = network(board, pieces, action_mask)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        
        results[batch_size] = {
            'batch_size': batch_size,
            'time_per_batch_ms': (elapsed / num_iters) * 1000,
            'samples_per_second': (batch_size * num_iters) / elapsed,
        }
    
    return results


def print_results(title: str, results: Dict[str, Any]) -> None:
    """Print benchmark results."""
    print(f"\n{'='*60}")
    print(title)
    print('='*60)
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        elif isinstance(value, dict):
            print(f"  Batch {key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")
                else:
                    print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    print('='*60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Block Blast")
    parser.add_argument(
        "--engine",
        action="store_true",
        help="Benchmark game engine"
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Benchmark environment"
    )
    parser.add_argument(
        "--vec-env",
        action="store_true",
        help="Benchmark vectorized environment"
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Benchmark neural network"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all benchmarks"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for network benchmark"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    if args.all or args.engine:
        results = benchmark_engine(num_games=1000, seed=args.seed)
        print_results("GAME ENGINE BENCHMARK", results)
    
    if args.all or args.env:
        results = benchmark_environment(num_steps=100000, seed=args.seed)
        print_results("ENVIRONMENT BENCHMARK", results)
    
    if args.all or args.vec_env:
        results = benchmark_vectorized_env(num_envs=64, num_steps=10000, seed=args.seed)
        print_results("VECTORIZED ENVIRONMENT BENCHMARK", results)
    
    if args.all or args.network:
        results = benchmark_network(device_name=args.device)
        print_results("NEURAL NETWORK BENCHMARK", results)
    
    if not any([args.all, args.engine, args.env, args.vec_env, args.network]):
        print("No benchmark selected. Use --all to run all benchmarks.")
        parser.print_help()


if __name__ == "__main__":
    main()
