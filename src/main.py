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
from algorithms.mazegenerator import get_maze
from algorithms.jps import solve_maze_jps

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
        choices=["astar", "dijkstra", "dfs", "jps"],
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for maze generation (same seed = same maze)",
    )
    args = parser.parse_args()

    # 1. Choose grid (always NumPy)
    if args.demo:
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
        size = args.size
        if size % 2 == 0:
            size += 1
        grid = get_maze(size=size, seed=args.seed)
        start, goal = (1, 1), (size - 2, size - 2)

    print("\n🧩 Generated Grid:")
    print_grid(grid)
    print(f"\nStart: {start}, Goal: {goal}\n")

    # 2. Compare mode: run all algorithms on the same grid
    if args.compare:
        print("🎯 Comparing All Algorithms on Same Grid")
        print("=" * 50)

        algorithms = {
            "A*": lambda: solve_maze_a_star(grid, start, goal),
            "Dijkstra": lambda: (solve_maze_dijkstra(grid, start, goal), []),
            "DFS": lambda: (solve_maze_dfs(grid, start, goal), []),
            "JPS": lambda: solve_maze_jps(grid, start, goal),
        }

        results = {}

        for name, algo_func in algorithms.items():
            print(f"\nRunning {name}...")
            t0 = time.time()

            try:
                result = algo_func()
                if isinstance(result, tuple) and len(result) == 2:
                    path, steps = result
                else:
                    path, steps = result, []
            except Exception as e:
                print(f"Error running {name}: {e}")
                continue

            runtime = time.time() - t0
            results[name] = {
                "path": path,
                "steps": steps,
                "runtime": runtime,
                "path_length": len(path) if path else 0,
            }

            # 📝 Log each algorithm's result to CSV
            log_results(
                algorithm=name,  # e.g. "A*", "Dijkstra"
                grid_size=int(grid.shape[0]),
                runtime=runtime,
                path_length=len(path) if path else 0,
                cost=None,
            )

            if path:
                print(f"  ✅ Path found: {len(path)} steps in {runtime:.6f}s")
            else:
                print(f"  ❌ No path found (took {runtime:.6f}s)")

        # Summary
        print("\n📊 Comparison Summary:")
        print("-" * 60)
        print(f"{'Algorithm':<12} {'Success':<8} {'Path Length':<12} {'Runtime (s)':<12}")
        print("-" * 60)
        for name, r in results.items():
            success = "✅" if r["path"] else "❌"
            path_len = r["path_length"] if r["path"] else "N/A"
            runtime = f"{r['runtime']:.6f}"
            print(f"{name:<12} {success:<8} {path_len:<12} {runtime:<12}")
        print("-" * 60)

        # pick best algorithms among those that actually found a path
        successful = [name for name, r in results.items() if r["path"]]

        if successful:
            # 🔹 fastest by runtime
            fastest = min(successful, key=lambda n: results[n]["runtime"])
            # 🔹 shortest by path length
            shortest = min(successful, key=lambda n: results[n]["path_length"])

            print(f"\n🏆 Fastest algorithm: {fastest} "
                  f"({results[fastest]['runtime']:.6f}s)")
            print(f"🧭 Shortest path: {shortest} "
                  f"({results[shortest]['path_length']} steps)")

            # choose which one to visualize (here: fastest)
            best_algo = fastest
            path = results[best_algo]["path"]

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            FIGURES_DIR.mkdir(parents=True, exist_ok=True)
            filename = FIGURES_DIR / f"comparison_{timestamp}.png"

            title = (f"Best (fastest): {best_algo} "
                     f"(len={len(path)}, time={results[best_algo]['runtime']:.6f}s)")
            plot_maze_with_path(grid, path, start, goal, title, save_path=str(filename))
            print(f"Comparison figure saved to: {filename}")

        return  # done in compare mode

    # 3. Single-algorithm mode
    print(f"🚀 Running {args.algo.upper()} from {start} → {goal}\n")
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
    else:
        raise ValueError("Unknown algorithm")

    runtime = time.time() - t0
    print("Grid checksum:", grid.shape, int(grid.sum()))

    if path:
        print("✅ Path found:")
        print(format_path(path))
        print(f"📏 Path length: {len(path)} steps")
        print(f"⏱  Runtime: {runtime:.6f} seconds")

        log_results(
            algorithm=args.algo.upper(),
            grid_size=int(grid.shape[0]),
            runtime=runtime,
            path_length=len(path),
            cost=None,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_name = args.algo.lower()
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        filename = FIGURES_DIR / f"{algo_name}_{timestamp}.png"

        title = f"{args.algo.upper()} (length={len(path)})"
        plot_maze_with_path(grid, path, start, goal, title, save_path=str(filename))
        print(f"🖼  Figure saved to: {filename}")

        if steps:
            plot_steps_curve(steps, algo_name=args.algo.upper())

        if args.animate:
            animate_maze_with_path(
                grid,
                path,
                start,
                goal,
                title=f"{args.algo.upper()} – animated path",
                pause_time=0.05,
            )
    else:
        print("❌ No path found!")


if __name__ == "__main__":
    main()
