# src/algorithms/astar.py

import heapq
from typing import List, Tuple, Dict

import numpy as np


def solve_maze_a_star(
    maze: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Solve a maze using the A* algorithm.

    Args:
        maze: 2D NumPy array (1 = wall, 0 = path)
        start: (y, x) start coordinates
        end:   (y, x) goal coordinates

    Returns:
        path:  list of (y, x) coordinates from start to end (empty if no path)
        steps: list where steps[i] = nodes expanded after iteration i
               (used for plotting search progress)
    """
    height, width = maze.shape

    # check if start and end are valid
    if maze[start[0], start[1]] != 0 or maze[end[0], end[1]] != 0:
        return [], []

    # 4-neighbour moves
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def heuristic(y: int, x: int) -> int:
        # Manhattan distance
        return abs(y - end[0]) + abs(x - end[1])

    # Priority queue: (f_score, g_score, y, x)
    open_heap: List[Tuple[float, float, int, int]] = []
    heapq.heappush(open_heap, (heuristic(*start), 0, start[0], start[1]))

    came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

    g_score = np.full((height, width), np.inf)
    g_score[start[0], start[1]] = 0

    f_score = np.full((height, width), np.inf)
    f_score[start[0], start[1]] = heuristic(*start)

    closed_set: set[Tuple[int, int]] = set()

    # 👇 for the search-progress curve
    steps: List[int] = []
    nodes_expanded = 0

    while open_heap:
        current_f, current_g, current_y, current_x = heapq.heappop(open_heap)

        # skip stale entries
        if current_g > g_score[current_y, current_x]:
            continue
        if (current_y, current_x) in closed_set:
            continue

        closed_set.add((current_y, current_x))
        nodes_expanded += 1
        steps.append(nodes_expanded)

        if (current_y, current_x) == end:
            # reconstruct path
            path: List[Tuple[int, int]] = []
            cur = (current_y, current_x)
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path, steps

        # explore neighbours
        for dy, dx in directions:
            ny, nx = current_y + dy, current_x + dx

            if not (0 <= ny < height and 0 <= nx < width):
                continue
            if maze[ny, nx] != 0:
                continue
            if (ny, nx) in closed_set:
                continue
                continue

            tentative_g = g_score[current_y, current_x] + 1

            if tentative_g < g_score[ny, nx]:
                came_from[(ny, nx)] = (current_y, current_x)
                g_score[ny, nx] = tentative_g
                f_score[ny, nx] = tentative_g + heuristic(ny, nx)
                heapq.heappush(open_heap, (f_score[ny, nx], tentative_g, ny, nx))

    # no path found
    return [], steps
