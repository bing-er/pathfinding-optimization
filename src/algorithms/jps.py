"""
Jump Point Search (JPS) algorithm implementation.
Author: Sepehr Mansouri
"""

import numpy as np
from typing import List, Tuple, Optional, Set
import heapq
import math


def solve_maze_jps(grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], List[int]]:
    """
    Solve maze using Jump Point Search algorithm.

    :param grid: 2D numpy array where 0 = walkable, 1 = wall
    :param start: Starting position (row, col)
    :param goal: Goal position (row, col)
    :return: Tuple of (path, steps_list) where path is List of positions from start to goal, or None if no path
        and steps_list: List tracking nodes explored (for visualization)
    """

    if not is_valid_position(grid, start) or not is_valid_position(grid, goal):
        return None, []

    if start == goal:
        return [start], [1]

    rows, cols = grid.shape

    # Priority queue: (f_score, g_score, position, parent)
    open_set = [(0, 0, start, None)]
    closed_set: Set[Tuple[int, int]] = set()
    came_from = {}
    g_score = {start: 0}
    steps = []  # Track number of nodes explored for visualization

    while open_set:
        # Get position with lowest f_score
        current_f, current_g, current_pos, parent = heapq.heappop(open_set)

        if current_pos in closed_set:
            continue

        closed_set.add(current_pos)
        came_from[current_pos] = parent
        steps.append(len(closed_set))

        # Found goal
        if current_pos == goal:
            path = reconstruct_path(came_from, goal)
            return path, steps

        # Get pruned neighbors using JPS rules
        directions = get_neighbors(grid, current_pos, parent)

        for direction in directions:
            # Jump to find next jump point
            jump_point = jump(grid, current_pos, direction, goal)

            if jump_point is None or jump_point in closed_set:
                continue

            # Calculate distances
            tentative_g = current_g + euclidean_distance(current_pos, jump_point)

            if jump_point not in g_score or tentative_g < g_score[jump_point]:
                g_score[jump_point] = tentative_g
                h_score = euclidean_distance(jump_point, goal)
                f_score = tentative_g + h_score

                heapq.heappush(open_set, (f_score, tentative_g, jump_point, current_pos))

    return None, steps


def is_valid_position(grid: np.ndarray, pos: Tuple[int, int]) -> bool:
    """Check if position is within grid bounds and walkable."""
    row, col = pos
    rows, cols = grid.shape
    return (0 <= row < rows and
            0 <= col < cols and
            grid[row, col] == 0)


def euclidean_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def get_neighbors(grid: np.ndarray, pos: Tuple[int, int], parent: Optional[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Get valid neighbors for a position, considering forced neighbors.
    implements JPS neighbor pruning rules.
    """
    row, col = pos

    # 8 directions: N, NE, E, SE, S, SW, W, NW
    directions = [
        (-1, 0),  # North
        (-1, 1),  # Northeast
        (0, 1),   # East
        (1, 1),   # Southeast
        (1, 0),   # South
        (1, -1),  # Southwest
        (0, -1),  # West
        (-1, -1)  # Northwest
    ]

    if parent is None:
        # if no parent - return all valid neighbors
        neighbors = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if is_valid_position(grid, (new_row, new_col)):
                neighbors.append((dr, dc))
        return neighbors

    # Calculate direction from parent to current position
    parent_row, parent_col = parent
    dx = 0 if col == parent_col else (1 if col > parent_col else -1)
    dy = 0 if row == parent_row else (1 if row > parent_row else -1)

    neighbors = []

    # Diagonal movement
    if dx != 0 and dy != 0:
        # Continue in same diagonal direction
        if is_valid_position(grid, (row + dy, col + dx)):
            neighbors.append((dy, dx))

        # Check for forced neighbors (corners)
        if not is_valid_position(grid, (row, col - dx)) and is_valid_position(grid, (row + dy, col)):
            neighbors.append((dy, 0))
        if not is_valid_position(grid, (row - dy, col)) and is_valid_position(grid, (row, col + dx)):
            neighbors.append((0, dx))

    # Horizontal movement
    elif dx != 0:
        if is_valid_position(grid, (row, col + dx)):
            neighbors.append((0, dx))

        # Check for forced neighbors above and below
        if not is_valid_position(grid, (row + 1, col)) and is_valid_position(grid, (row + 1, col + dx)):
            neighbors.append((1, dx))
        if not is_valid_position(grid, (row - 1, col)) and is_valid_position(grid, (row - 1, col + dx)):
            neighbors.append((-1, dx))

    # Vertical movement
    elif dy != 0:
        if is_valid_position(grid, (row + dy, col)):
            neighbors.append((dy, 0))

        # Check for forced neighbors left and right
        if not is_valid_position(grid, (row, col + 1)) and is_valid_position(grid, (row + dy, col + 1)):
            neighbors.append((dy, 1))
        if not is_valid_position(grid, (row, col - 1)) and is_valid_position(grid, (row + dy, col - 1)):
            neighbors.append((dy, -1))

    return neighbors


def has_forced_neighbors(grid: np.ndarray, pos: Tuple[int, int], direction: Tuple[int, int]) -> bool:
    """
    Checks if position has forced neighbors, making it a jump point.
    """
    row, col = pos
    dr, dc = direction

    # Diagonal movement
    if dr != 0 and dc != 0:
        return ((not is_valid_position(grid, (row - dr, col)) and is_valid_position(grid, (row - dr, col + dc))) or
                (not is_valid_position(grid, (row, col - dc)) and is_valid_position(grid, (row + dr, col - dc))))

    # Horizontal movement
    elif dc != 0:
        return ((not is_valid_position(grid, (row + 1, col)) and is_valid_position(grid, (row + 1, col + dc))) or
                (not is_valid_position(grid, (row - 1, col)) and is_valid_position(grid, (row - 1, col + dc))))

    # Vertical movement
    elif dr != 0:
        return ((not is_valid_position(grid, (row, col + 1)) and is_valid_position(grid, (row + dr, col + 1))) or
                (not is_valid_position(grid, (row, col - 1)) and is_valid_position(grid, (row + dr, col - 1))))

    return False


def jump(grid: np.ndarray, pos: Tuple[int, int], direction: Tuple[int, int],
         goal: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """
    Jump in a direction until we find a jump point or hit an obstacle.
    This is the core of JPS optimization.
    """
    row, col = pos
    dr, dc = direction

    # Move one step in the direction
    new_row, new_col = row + dr, col + dc

    # Check bounds and obstacles
    if not is_valid_position(grid, (new_row, new_col)):
        return None

    # Reached goal
    if (new_row, new_col) == goal:
        return (new_row, new_col)

    # Check for forced neighbors (this makes it a jump point)
    if has_forced_neighbors(grid, (new_row, new_col), direction):
        return (new_row, new_col)

    # For diagonal movement, check horizontal and vertical jumps
    if dr != 0 and dc != 0:
        # Try horizontal jump
        if jump(grid, (new_row, new_col), (0, dc), goal) is not None:
            return (new_row, new_col)
        # Try vertical jump
        if jump(grid, (new_row, new_col), (dr, 0), goal) is not None:
            return (new_row, new_col)

    # Continue jumping in same direction
    return jump(grid, (new_row, new_col), direction, goal)


def reconstruct_path(came_from: dict, goal: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Reconstruct the path from start to goal.
    """
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = came_from.get(current)

    return path[::-1]  # Reverse to get start->goal order