# src/algorithms/mazegenerator.py
"""
Maze generation utilities using Depth-First Search (DFS).

Exports:
- generate_maze_dfs(width, height) -> np.ndarray
- get_maze(size, seed=None) -> np.ndarray
- show_maze(maze) -> None
"""

import random
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def get_maze(size: int = 21, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a square DFS maze of roughly `size` x `size`.

    - Ensures the size is odd (DFS maze uses odd indices for paths).
    - Optionally sets a random seed for reproducibility.

    Returns:
        np.ndarray of shape (H, W) with:
            1 = wall
            0 = path
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if size % 2 == 0:
        size += 1  # force odd to avoid indexing issues

    return generate_maze_dfs(size, size)


# --------------------------
# 1. Maze Generation with DFS
# --------------------------
def generate_maze_dfs(width: int, height: int) -> np.ndarray:
    """
    Generate a random maze using Depth-First Search (DFS).

    Args:
        width:  number of columns in the maze (x-axis)
        height: number of rows in the maze (y-axis)

    Returns:
        np.ndarray: Maze grid where:
            1 = wall
            0 = path
    """
    # Initialize maze with all walls (1). Use odd indices for paths to ensure wall borders.
    maze = np.ones((height, width), dtype=int)

    # Stack to track DFS traversal (stores (y, x) coordinates)
    dfs_stack = []

    # Starting point (must be odd indices to avoid border walls)
    start_y, start_x = 1, 1
    maze[start_y, start_x] = 0
    dfs_stack.append((start_y, start_x))

    # 4 possible movement directions (up, down, left, right)
    # Step size 2 to “jump over” the wall between cells
    directions = [(-2, 0), (2, 0), (0, -2), (0, 2)]

    while dfs_stack:
        # Current position (LIFO: Last-In-First-Out for DFS)
        current_y, current_x = dfs_stack.pop()

        # Shuffle directions to ensure random maze generation
        random.shuffle(directions)

        for dy, dx in directions:
            # Next cell position (skipping over the wall)
            next_y = current_y + dy
            next_x = current_x + dx

            # Check bounds and ensure next cell is still a wall
            if (
                1 <= next_y < height - 1
                and 1 <= next_x < width - 1
                and maze[next_y, next_x] == 1
            ):
                # Carve path: mark next cell as path and remove the wall between current and next
                maze[next_y, next_x] = 0
                maze[current_y + dy // 2, current_x + dx // 2] = 0  # remove middle wall

                # Continue DFS from this new cell
                dfs_stack.append((next_y, next_x))

    # Ensure fixed start (top-left inside border) and end (bottom-right inside border) are paths
    maze[1, 1] = 0                 # start
    maze[height - 2, width - 2] = 0  # goal

    return maze


def show_maze(maze: np.ndarray) -> None:
    """Visualize a generated maze using matplotlib."""
    plt.figure(figsize=(5, 5))
    plt.imshow(maze, cmap="binary")  # 1 = wall (black), 0 = path (white)
    plt.axis("off")
    plt.title("Generated Maze (DFS)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple test: generate a 21x21 maze and display it
    test_maze = generate_maze_dfs(21, 21)
    show_maze(test_maze)
