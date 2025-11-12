# Grid Pathfinding Benchmark — Professor’s Guide

This document explains **every moving part** of the benchmark notebook `maze_benchmark_corners.ipynb` and how to modify it. It is written for fast orientation and **controlled change** of any component.

---

## 0) What this notebook does

- Generates DFS mazes (perfect trees) and optional **loop openings** (sparsity `p`) to create harder/easier graphs.
- Benchmarks four solvers on **paired, reproducible** conditions: **A\***, **Dijkstra**, **DFS**, **JPS** (with path stitching).
- Forces a **corner-to-corner** pair in every condition to remove start/goal variance. Also samples several interior pairs.
- Logs metrics to CSV, produces plots, and draws **canonical overlays** for corners.
- All randomness is **seeded** for reproducibility.

Artifacts are saved to `./artifacts_all_in_one/`:
- `runs.csv` — raw measurements
- `corner_median_runtime_*` — runtime bars by condition
- `overlay_*` — path overlays per algorithm and condition

---

## 1) File layout and top-level switches

Key parameters (Cell 5 in the notebook):

```python
ALGS   = [("A*", astar_metrics), ("DFS", dfs_metrics), ("Dijkstra", dijkstra_metrics), ("JPS", jps_metrics_stitched)]
SIZES  = [(31,31), (61,61), (91,91)]                     # grid sizes (odd preferred)
P_VALUES = [0.0, 0.05, 0.10]                             # loop-opening probability (0 = perfect maze)
K      = 3                                               # seeds per {size, p} condition
```

Change these to scale the experiment:
- **SIZES**: pick odd numbers (DFS generator carves on a grid with borders). More cells ⇒ more runtime.
- **P_VALUES**: `0.0` gives a perfect maze (single simple path). Increasing `p` injects loops; more alternatives.
- **K**: number of different seeds per condition. Higher = better statistics, longer runtime.
- **ALGS**: add/remove algorithms (see §7).

Runtime hygiene (Cell 1):
- Plots and CSVs land in `ARTIFACTS = './artifacts_all_in_one'`.

---

## 2) Maze generation

### `generate_maze_dfs_seeded(width, height, seed)`
- Uses a **stack-based DFS** to carve a “perfect” maze. 
- Cells hold **1=wall**, **0=free**. Carving moves in **two-cell steps** and removes the middle wall.
- Guarantees `(1,1)` and `(H-2,W-2)` are free.

**Modify**:
- To bias structure, change `dirs = [(-2,0), (2,0), (0,-2), (0,2)]` to reorder or probabilistically prefer directions.

### `add_loops(maze, p, seed)`
- Iterates interior cells; any wall becomes free with probability `p` (seeded). Larger `p` ⇒ more cycles.

**Modify**:
- Replace uniform opening with degree-aware rules; e.g., only open walls that connect two different trees.

### `free_cells(maze)`
- Counts free cells; used for metadata.

---

## 3) Start/Goal pairing (controls geometry variance)

### `sg_pairs_for_grid(maze, seed, k_random=4)`
- Always includes **corner pair** `((1,1), (H-2,W-2))`. This is labeled via `is_corner=1` (see §5).
- Adds up to `k_random` additional interior pairs sampled from free cells with a fixed seed.

**Modify**:
- Only corners: set `k_random=0` in the harness.
- Force specific pairs: return a hard-coded list.
- Diagonal endpoints: compute `(H//2, 1) → (H//2, W-2)`, etc.

**Why corners?**
- Removes placement variance. Keeps seeds and maze topology as the only sources of variation.

---

## 4) Solvers

All solvers return a dict with at least:
`path`, `nodes_expanded`, `nodes_generated`, `peak_frontier`, `peak_closed`, `expanded_order`, `pop_times_ns`.

### DFS — `dfs_metrics`
- Uninformed search. Returns **a** path if one exists. On a tree, it equals the unique simple path.
- Useful as a fast baseline; not optimal under loops.

**Change neighbor order** to alter its traversal:
```python
dirs = [(-1,0), (1,0), (0,-1), (0,1)]
# Try: random.shuffle(dirs) each pop with a fixed seed for repeatable variety.
```

### Dijkstra — `dijkstra_metrics`
- Uniform-cost search; returns optimal shortest path in edges.
- Also used as the **ground-truth baseline** for `suboptimality` (edges / optimal_edges).

### A\* — `astar_metrics`
- Manhattan heuristic (admissible for 4-connected edges). Returns an optimal path.

**Swap heuristic**:
```python
def h(p): return abs(p[0]-end[0]) + abs(p[1]-end[1])     # default
# Try weighted A*: f = g + w*h  (w>1 for greedier behavior).
```

### JPS — `solve_maze_jps` + stitching
- JPS searches **jump points** on an 8-neighbor lattice, then we **stitch** between jump points using 4-neighbor A* (`_astar_4`) to get a comparable step path.
- The wrapper `jps_metrics_stitched` returns stitched paths so overlays are consistent.

**Troubleshooting**:
- If stitching fails for a segment, function falls back to A* for the full pair.

---

## 5) Timing and metrics

### `timed_median(solver, maze, start, end, warmup=1, reps=5)`
- Runs `warmup` unmeasured iterations, then `reps` measured runs, and stores the **median** time in `runtime_ms`.
- Tracks peak RSS delta if `psutil` is available.

**Modify**:
- To stress cache effects, increase `warmup`.
- For higher precision, increase `reps` and report both median and IQR.

### `RunRow` (dataclass)
Fields include:
- `size`, `p`, `map_type`, `algo`, **`runtime_ms`**, **`expanded`**, `generated`, `peak_frontier`, `peak_closed`,
- `edges` (steps in the returned path), **`suboptimality`**, `peak_rss_mb`, `free`,
- `start_*`/`end_*` coordinates,
- **`is_corner`** (1 for corner-to-corner).

### `append_measurement_row(...)`
- Computes `success` (endpoints and 4-connected continuity), `edges`, `suboptimality` (vs Dijkstra), sets `is_corner`.

**Add a metric**:
1. Compute it in the solver or here.
2. Add a new field to `RunRow`.
3. Include it in the appended instance and adapt CSV/plots.

---

## 6) The harness

Main loop (Cell 5):
1. For every `(H, W)` in `SIZES` and `p` in `P_VALUES`  
2. Derive a seeded maze; open loops with probability `p`  
3. Build S/G pairs via `sg_pairs_for_grid`  
4. Cache `optimal_edges` via one Dijkstra per pair  
5. For each `algo` run `timed_median` and `append_measurement_row`.

**Scale up/down**:
- **Larger SIZES** and higher `K` improve confidence; expect superlinear runtime.
- Cut `ALGS` to shorten runs.

---

## 7) Adding a new algorithm

1. Implement `new_algo_metrics(maze, start, end) -> dict` with the standard keys (`path`, `nodes_expanded`, …).
2. Register it:
   ```python
   ALGS = [("A*", astar_metrics), ("Dijkstra", dijkstra_metrics), ("DFS", dfs_metrics),
           ("JPS", jps_metrics_stitched), ("MyAlgo", new_algo_metrics)]
   ```
3. The harness and plots pick it up automatically.

**If your solver uses different costs or 8-neighbors**: document the cost model; suboptimality vs Dijkstra may not be meaningful. Either normalize costs or compute a solver-specific optimal baseline.

---

## 8) Corner-only analysis

Corner-only removes start/goal variance:
```python
def corner_view(df): return df[df["is_corner"] == 1].copy()
```
Use it for headline plots and stats. Keep the full `df` to show robustness on interior pairs.

### Example table
```python
(corner_view(df)
 .groupby(["size","map_type","algo"])[["runtime_ms","expanded"]]
 .median().reset_index())
```

---

## 9) Overlays

`canonical_overlays_corners(df)` regenerates the exact maze per `{size, map_type}`, then runs each algorithm on the **corner pair** and overlays the returned path.

Why do paths sometimes look identical?  
- On **perfect mazes** (`p=0.0`) the graph is a **tree**, so there is only **one** simple path between start and goal. All algorithms will draw the same route.

To visualize differences, use `p>0` so multiple shortest paths exist; DFS will likely diverge due to traversal order.

---

## 10) Reproducibility knobs

- **Seeding**: every condition uses `base_seed = 10_000*H + 100*W + int(p*1000)` and offset `t`.
- **Deterministic S/G pairs**: built with `np.random.default_rng(seed)`.
- **Artifacts**: CSV and figures written with parameterized filenames.

Change any of these if you need different seeding regimes; keep them **pure functions of (H, W, p, t)** for auditability.

---

## 11) Common changes (recipes)

- **Only corners**: set `k_random=0` in `sg_pairs_for_grid` call inside the harness.
- **More loops**: increase `P_VALUES` upper bound (e.g., `0.20`).
- **Larger grids**: add `(121,121)`, `(151,151)` to `SIZES`.
- **More trials**: raise `K` to `10` or `20`.
- **ECDFs and significance tests**: compute from `df` or `corner_view(df)` using SciPy (Friedman/Wilcoxon) if installed.
- **Memory focus**: ensure `psutil` is installed; plot `peak_rss_mb` vs `runtime_ms` Pareto charts.

---

## 12) Validation

- `is_valid_path` checks endpoints, 4-connectivity, and in-bounds free cells.
- `suboptimality` is defined only when Dijkstra’s baseline exists and > 0.
- For A\*/JPS/Dijkstra, path lengths should match on 4-connected grids with uniform cost.

---

## 13) Performance tips

- Use fewer `K` and fewer `SIZES` for fast iterations.
- For heavy runs, turn off overlays or reduce `reps` in `timed_median`.
- Pin matplotlib backend to non-interactive if running headless.

---

## 14) Known limitations

- JPS is implemented on 8-neighbors then **stitched** to 4-neighbors for consistent overlays. This is deliberate to compare like-for-like step paths; pure JPS jump sequences are sparser.
- Perfect mazes (p=0) hide differences between optimal algorithms visually; use `p>0` for more informative overlays.

---

## 15) Quick checklist for newcomers

1. Run the notebook once. Confirm `runs.csv` and a few figures appear in `./artifacts_all_in_one/`.
2. Open `runs.csv` and scan columns; verify `is_corner` exists.
3. Increase `K` to 6–10, rerun. Differences stabilize.
4. Change `P_VALUES` to `[0.0, 0.1, 0.2]`. Re-run overlays and compare.
5. Add a toy algorithm (e.g., Greedy Best-First) to validate pipeline extensibility.

---

## 16) Where to make specific changes

| Goal | File/Cell | What to edit |
|---|---|---|
| Add algorithm | Cell 2 + `ALGS` in Cell 5 | Define `new_algo_metrics`; register in `ALGS` |
| Change sizes | Cell 5 | `SIZES` |
| Change sparsity | Cell 5 | `P_VALUES` |
| More seeds | Cell 5 | `K` |
| Only corners | Cell 5 call site | set `k_random=0` in `sg_pairs_for_grid` |
| Neighbor order | Cell 2 (DFS) | Modify `dirs` or shuffle per pop |
| Heuristic | Cell 2 (A*) | Edit `h(p)` or add weight |
| Stronger timing | Cell 3 | `warmup`, `reps` in `timed_median` |
| Output folder | Cell 0 | `ARTIFACTS` |

---

## 17) Minimal troubleshooting

- **All overlays identical**: You are on `p=0`. This is expected.
- **JPS overlay blank**: The stitcher falls back to A\*; ensure the maze and endpoints are free. Try higher `p` or a different seed.
- **CSV empty**: A cell threw earlier. Re-run from the top and watch for exceptions.
- **Very slow**: Reduce `SIZES`, `K`, and `reps`.
