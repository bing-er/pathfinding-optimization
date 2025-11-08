# src/visualizations/charts.py
"""
Author: Binger Yu
"""
import os
import numpy as np
import matplotlib.pyplot as plt


def plot_maze_with_path(maze: np.ndarray, path, start, goal,
                        title="Pathfinding Result", save_path: str | None = None):
    """
    Visualize a maze and a path.
    If save_path is provided, save the figure to that file.
    """
    maze = np.array(maze)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze, cmap="binary")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=4, color="#0000FF", solid_capstyle="round")

    sy, sx = start
    gy, gx = goal
    ax.scatter([sx], [sy], marker="o", s=60)  # start
    ax.scatter([gx], [gy], marker="x", s=60)  # goal

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    # Save to file if requested
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)

    plt.show()


def plot_steps_curve(steps, algo_name: str = "A*"):
    """
    Plot a simple curve showing how many nodes have been expanded over time.
    """
    if not steps:
        print("No step data to plot.")
        return

    plt.figure(figsize=(5, 4))
    plt.plot(range(1, len(steps) + 1), steps,color="#0000FF", linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Nodes expanded")
    plt.title(f"{algo_name} – Search Progress")
    plt.tight_layout()
    plt.show()


def animate_maze_with_path(maze, path, start, goal, title="Path Animation", pause_time: float = 0.05):
    maze = np.array(maze)

    plt.ion()  # interactive mode on

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(maze, cmap="binary")

    sy, sx = start
    gy, gx = goal
    ax.scatter([sx], [sy], marker="o", s=70)  # start
    ax.scatter([gx], [gy], marker="x", s=70)  # goal

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    ys, xs = [], []

    # create a single line object and update it
    line, = ax.plot(
        [], [],
        linewidth=4,
        color="#0000FF",
        solid_capstyle="round"
    )

    for (y, x) in path:
        ys.append(y)
        xs.append(x)
        line.set_data(xs, ys)
        plt.draw()
        plt.pause(pause_time)

    plt.ioff()
    plt.show()
