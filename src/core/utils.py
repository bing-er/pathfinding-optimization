# src/core/utils.py
"""
Utility helpers for the Pathfinding Optimization project.

Currently includes:
- log_results: append experiment metrics to a CSV file
- format_path: pretty-print a path as string

Author: Binger Yu
"""

import csv
from datetime import datetime
from pathlib import Path
# ...
# utils.py is at: <repo>/src/core/utils.py
# parents[0] = <repo>/src/core
# parents[1] = <repo>/src
# parents[2] = <repo>   project root
BASE_DIR = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = RESULTS_DIR / "logs"


def log_results(
    algorithm: str,
    grid_size: int,
    runtime: float,
    path_length: int,
    cost: float | None = None,
    log_dir: Path | None = None,
    filename: str = "runtime_log.csv",
) -> None:
    """
    Append one experiment result to a CSV file.

    Columns:
    timestamp, algorithm, grid_size, runtime, path_length, cost
    """
    # Use default logs directory if none is provided
    log_root = LOGS_DIR if log_dir is None else Path(log_dir)

    # Ensure result/logs directories exist
    log_root.mkdir(parents=True, exist_ok=True)

    csv_path = log_root / filename
    header = ["timestamp", "algorithm", "grid_size",
              "runtime", "path_length", "cost"]

    file_exists = csv_path.exists()

    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            algorithm,
            grid_size,
            f"{runtime:.6f}",
            path_length,
            "" if cost is None else f"{cost:.6f}",
        ])


def format_path(path, per_line: int = 10) -> str:
    """
    Convert a path into a readable string with line breaks.
    Each element in `path` can be (y, x, ...) – only first two values are used.
    """
    # turn each point into "(y,x)"
    tokens = []
    for p in path:
        y, x = p[0], p[1]
        tokens.append(f"({y},{x})")

    # group into lines of at most `per_line` points
    lines = []
    for i in range(0, len(tokens), per_line):
        chunk = " -> ".join(tokens[i:i + per_line])
        lines.append(chunk)

    # indent lines after the first a bit for readability
    return "\n    ".join(lines)
