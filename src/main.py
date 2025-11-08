"""
Main entry point for the Pathfinding Algorithms Comparison project.
Author: Binger Yu
"""

from pathlib import Path
import argparse
import time
from datetime import datetime

import numpy as np

from visualizations.charts import (
    plot_maze_with_path,
    plot_steps_curve,
    animate_maze_with_path,
)
from core.utils import log_results, format_path
from algorithms.astar import solve_maze_a_star
from algorithms.dijkstra import solve_maze_dijkstra
from algorithms.dfs import solve_maze_dfs
from algorithms.jps import solve_maze_jps
from algorithms.mazegenerator import generate_maze_dfs

# Project root = parent of src/
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"


def print_grid(grid: np.ndarray) -> None:
    """Pretty-print a 2D NumPy array of 0/1 values."""
    for row in grid:
        print(" ".join(str(int(c)) for c in row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pathfinding Algorithm Comparison Tool"
    )
    parser.add_argument(
        "--algo",
        choices=["astar", "dijkstra", "dfs", "jps"],  # Added JPS
        default="astar",
        help="Algorithm to run (astar, dijkstra, dfs, jps)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=21,
        help="Size of the grid (NxN, odd values recommended)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run on a fixed demo 5x5 grid for testing",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate the path being drawn step-by-step",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all algorithms on the same grid",
    )
    args = parser.parse_args()

    # 1. Choose grid (always NumPy)
    if args.demo:
        # Simple hand-made 5x5 test grid
        grid = np.array(
            [
                [0, 0, 0, 0, 0],
                [1, 1, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
            ],
            dtype=int,
        )
        start, goal = (0, 0), (4, 4)
    else:
        # DFS maze generator returns a NumPy array (1 = wall, 0 = path)
        size = args.size
        if size % 2 == 0:
            size += 1  # DFS maze works best with odd dimensions
        grid = generate_maze_dfs(size, size)
        # Start/goal chosen inside the maze (avoid outer border walls)
        start, goal = (1, 1), (size - 2, size - 2)

    # 2. Show the grid in text
    print("\n Generated Grid:")
    print_grid(grid)
    print(f"\n Start: {start}, Goal: {goal}\n")

    # 3. Run algorithm comparison or single algorithm
    if args.compare:
        # Compare all algorithms on the same grid
        print("🎯 Comparing All Algorithms on Same Grid")
        print("=" * 50)

        algorithms = {
            "A*": lambda: solve_maze_a_star(grid, start, goal),
            "Dijkstra": lambda: (solve_maze_dijkstra(grid, start, goal), []),
            "DFS": lambda: (solve_maze_dfs(grid, start, goal), []),
            "JPS": lambda: solve_maze_jps(grid, start, goal)
        }

        results = {}

        for name, algo_func in algorithms.items():
            print(f"\nRunning {name}...")
            t0 = time.time()

            try:
                result = algo_func()
                if len(result) == 2:
                    path, steps = result
                else:
                    path, steps = result, []
            except Exception as e:
                print(f"Error running {name}: {e}")
                continue

            runtime = time.time() - t0

            results[name] = {
                'path': path,
                'steps': steps,
                'runtime': runtime,
                'path_length': len(path) if path else 0
            }

            # Print results
            if path:
                print(f"  ✅ Path found: {len(path)} steps in {runtime:.6f}s")
            else:
                print(f"  ❌ No path found (took {runtime:.6f}s)")

        # Summary table
        print(f"\n📊 Comparison Summary:")
        print("-" * 60)
        print(f"{'Algorithm':<12} {'Success':<8} {'Path Length':<12} {'Runtime (s)':<12}")
        print("-" * 60)

        for name, result in results.items():
            success = "✅" if result['path'] else "❌"
            path_len = result['path_length'] if result['path'] else "N/A"
            runtime = f"{result['runtime']:.6f}"
            print(f"{name:<12} {success:<8} {path_len:<12} {runtime:<12}")

        print("-" * 60)

        # Save visualization for best performing algorithm
        best_algo = min([name for name, r in results.items() if r['path']],
                        key=lambda x: results[x]['runtime'], default=None)

        if best_algo:
            print(f"\n🏆 Fastest algorithm: {best_algo}")
            path = results[best_algo]['path']

            # Save comparison figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            filename = FIGURES_DIR / f"comparison_{timestamp}.png"

            title = f"Best: {best_algo} (length={len(path)})"
            plot_maze_with_path(
                grid, path, start, goal, title, save_path=str(filename)
            )
            print(f"Comparison figure saved to: {filename}")

    else:
        # Run single algorithm
        print(f"Running {args.algo.upper()} from {start} → {goal}\n")

        # 4. Run selected algorithm and measure runtime
        t0 = time.time()

        if args.algo == "astar":
            path, steps = solve_maze_a_star(grid, start, goal)
        elif args.algo == "dijkstra":
            path = solve_maze_dijkstra(grid, start, goal)
            steps = []
        elif args.algo == "dfs":
            path = solve_maze_dfs(grid, start, goal)
            steps = []
        elif args.algo == "jps":
            path, steps = solve_maze_jps(grid, start, goal)

        runtime = time.time() - t0

        # 5. Print result, log metrics, and visualize
        if path:
            print("Path found:")
            print(format_path(path))
            print(f"Path length: {len(path)} steps")
            print(f"Runtime: {runtime:.6f} seconds")

            # Log metrics to CSV (utils.py handles the path)
            log_results(
                algorithm=args.algo.upper(),
                grid_size=int(grid.shape[0]),
                runtime=runtime,
                path_length=len(path),
                cost=None,
            )

            # Save static figure
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            algo_name = args.algo.lower()

            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            filename = FIGURES_DIR / f"{algo_name}_{timestamp}.png"

            title = f"{args.algo.upper()} (length={len(path)})"
            plot_maze_with_path(
                grid,
                path,
                start,
                goal,
                title,
                save_path=str(filename),  # matplotlib expects a string
            )
            print(f"Figure saved to: {filename}")

            # Running-steps graph (only for algorithms that return steps, i.e. A* and JPS)
            if steps:
                plot_steps_curve(steps, algo_name=args.algo.upper())

            # Optional animation (only if user asks for it)
            if args.animate:
                animate_maze_with_path(
                    grid,
                    path,
                    start,
                    goal,
                    title=f"{args.algo.upper()} – animated path",
                    pause_time=0.05,  # smaller = faster animation
                )

        else:
            print("No path found!")


if __name__ == "__main__":
    main()