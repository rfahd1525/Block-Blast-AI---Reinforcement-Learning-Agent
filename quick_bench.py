"""Quick benchmark for game engine performance."""
import time
import sys
sys.path.insert(0, 'src')

from game.engine import GameEngine, play_random_game
import numpy as np

print("Benchmarking raw game engine performance...")

start = time.perf_counter()
moves = 0

for i in range(1000):
    result = play_random_game(seed=i)
    moves += result['moves_made']

elapsed = time.perf_counter() - start
print(f"Total moves: {moves}")
print(f"Time: {elapsed:.2f}s")
print(f"Moves/sec: {moves/elapsed:.0f}")
print(f"Games/sec: {1000/elapsed:.0f}")

# Requirement is 10,000+ moves/second
if moves/elapsed >= 10000:
    print("\n✓ PASSES 10,000+ moves/second requirement")
else:
    print(f"\n✗ Below target ({moves/elapsed:.0f} < 10,000)")
