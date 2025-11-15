<div align="center">

# 🚀 Pathfinding Algorithms Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

</div>

This repository contains our group project for *COMP 9060 – Applied Algorithm Analysis*, comparing classical and optimized pathfinding algorithms: **A\***, **Dijkstra**, **DFS**, and **JPS (Jump Point Search)**.  
The study focuses on performance, path optimality, and efficiency across different grid-based environments.


## 👥 Team Members
| Name | Role |
|------|------|
| **Yansong** | Algorithm Developer – A*, Dijkstra, DFS Implementation |
| **Sepehr** | QA & Testing Lead – JPS Implementation and Integration |
| **Vibhor** | Evaluation Lead – Metrics Analysis and Visualization |
| **Binger** | Project Manager – Documentation, Reporting, and Presentation |


## 🎯 Project Overview
Pathfinding is a fundamental problem in AI, robotics, and game development.
This project aims to:
- Implement **A***, **Dijkstra**, **DFS**, and **JPS** algorithms in a common grid framework.
- Evaluate their performance on various **grid configurations** (sparse vs dense, small vs large).
- Measure **runtime**, **path cost**, and **node expansions** to analyze algorithmic efficiency.
- Visualize algorithm behavior through comparative charts and heatmaps.
**Jump Point Search (JPS)** improves **A*** by **skipping redundant nodes** in uniform-cost grids, reducing runtime while preserving optimal path cost.


## 🗂️ Repository Structure
```
pathfinding-optimization/
├── data/                    # Sample grid maps and test cases
│   └── maps/                # Example .txt or .csv grid files
├── docs/                    # Documentation and reports
│   ├── proposal.pdf         # Submitted project proposal
│   ├── report_draft.docx    # In-progress final report
│   └── slides.pptx          # Presentation slides
├── notebooks/                       # Jupyter notebooks for experiments and demos
│   ├── 01_astar_demo.ipynb          # Interactive A* pathfinding demo
│   ├── 02_dijkstra_runtime.ipynb    # Runtime analysis for Dijkstra’s algorithm
│   └── 03_visualization_tests.ipynb # Prototyping plots/heatmaps before moving to src/
├── results/                 # Experiment outputs, logs, and performance data
│   ├── figures/             # Generated charts and comparison graphs
│   └── logs/                # Raw runtime and node expansion logs
├── src/                     # Source code for all algorithms
│   ├── algorithms/          # Pathfinding algorithm implementations
│   │   ├── astar.py         # A* baseline algorithm
│   │   ├── dfs.py           # Depth-First Search baseline
│   │   ├── dijkstra.py      # Dijkstra baseline algorithm
│   │   ├── jps.py           # Jump Point Search (JPS) implementation
│   │   └── mazegenerator.py # DFS-based random maze generator
│   ├── core/                # Shared components
│   │   ├── grid.py          # Grid representation and movement rules
│   │   ├── heuristics.py    # Heuristic functions (Manhattan, Octile, etc.)
│   │   └── utils.py         # Utility functions (logging, timers, helpers)
│   ├── visualizations/      # Visualization and performance analysis
│   │   ├── charts.py        # Static plots for paths and metrics
│   │   └── runtime_plot.py  # Search-progress / runtime-steps plots
│   └── main.py              # Entry point to run and compare algorithms
└── README.md                # Project overview and usage instructions
```


## ⚙️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/bing-er/pathfinding-optimization.git
cd pathfinding-iptimization
```

### 2. Set Up Environment
```
python3 -m venv venv
source venv/bin/activate        # (Mac/Linux)
venv\Scripts\activate           # (Windows)
pip install -r requirements.txt
```

### 3. Run Algorithms
Run individual algorithms:
```
python src/algorithms/astar.py
python src/algorithms/dijkstra.py
python src/algorithms/dfs.py
python src/algorithms/jps.py
```
Or compare all from the main runner:
```
python src/main.py
```
### 4. Visualize Results
Generated logs and performance visualizations will appear in the results/ folder.
You can adjust grid size, obstacle density, or heuristic type in main.py.

### 📊 Evaluation Metrics

| Metric               | Description                      |
| -------------------- | -------------------------------- |
| **Path Length**      | Total distance of computed route |
| **Computation Time** | Time required to reach goal      |
| **Node Expansions**  | Number of explored nodes         |
| **Scalability**      | Performance on larger grid maps  |

## 🧭 Progress Summary 
### Week 10 – Midterm Status
By Week 10, our team has completed the baseline phase of the project. The core pathfinding algorithms - **A***, **Dijkstra**, and **DFS** - have all been implemented, tested, and merged into the main branch. We also added a maze generator to hlep us create consistent test grids for experiments. 

The repository is now fully organized with a clear folder structure, evaluation metrics, and documentation. Everyone’s roles are defined — **Yansong** handled the baseline algorithms, **Sepehr** is leading the **Jump Point Search (JPS)** development, **Vibhor** is focusing on evaluation and visualization, and **Binger** is managing documentation, scheduling, and overall coordination.

Our next milestone is to integrate and test **JPS**, comparing its performance against the baseline algorithms. The team will also begin logging runtime and node-expansion data and preparing visual outputs for comparison. In the following weeks, we’ll move toward compiling the final report, creating visuals, and getting ready for our presentation in Week **14 (Dec 2)**.

### Week 11 - Transition to Performance Testing
By **Week 11 (Nov 11, 2025)**, our team completed the integration phase and transitioned into the **performance testing and visualization stage** of the Pathfinding Optimization Project.
All four pathfinding algorithms — **A***, **Dijkstra**, **DFS**, and **JPS** — are now unified under a consistent evaluation framework, allowing direct comparison on identical grid environments.

The full testing pipeline for **runtime**, **path length**, and **node-expansion metrics** has been finalized. The visualization layer is now being extended for clearer comparative analysis.

The repository is now fully operational, supporting **reproducible experiments**, **runtime logging**, and **benchmark visualizations**.
### ✅ Highlights (Week 11)
**Algorithm Integration & Framework**<br>
✔️ All algorithms (A*, Dijkstra, DFS, JPS) integrated and verified under main.py --compare.
✔️ Unified output schema established for cross-algorithm comparison.
✔️ Consistent testing environment established using fixed random seeds.

**Benchmark & Testing Pipeline**<br>
* ✔️ Performance testing plan finalized
* → Grid sizes: 10×10 → 101×101
* → Densities: 30%, 50%, 70%
* ✔️ Visualization notebooks updated for runtime and node-expansion comparison.
* ✔️ Benchmark suite (maze_benchmark_corners) integrated for comparative testing.

**Team Collaboration**<br>
* ✔️ Team meeting (Nov 11) to finalize responsibilities for runtime testing and data consolidation.
* ✔️ Visualization and logging pipeline now stable for batch testing.

#### 👥 Team Contributions
**🧠 Yansong**
* Implemented and verified the **A***, **Dijkstra**, and **DFS** baseline algorithms.
* Validated path optimality and ensured consistent output formats.
* Assisted in testing alignment between all algorithm interfaces.

**⚙️ Sepehr**
Finalized the **Jump Point Search (JPS)** algorithm with jump + pruning logic.
* Verified cross-comparison results between JPS and the baseline methods.
* Supported debugging and consistency checks across the benchmarking pipeline.

**📊 Vibhor**
* Developed and pushed **benchmark testing notebooks and scripts** (`maze_benchmark_corners.ipynb`, `.py`, `.html`).
* Designed the **performance testing plan** for grid sizes and obstacle densities.
* Implemented **runtime and node-expansion visualization** in Jupyter Notebook.
* Coordinated with team for data collection and figure generation.

**🧩 Binger**
* Implemented and maintained the unified **main runner** (`main.py`) and `--compare` mode.
* Integrated Vibhor’s visualization branch and validated its functionality.
* Updated the logging and results pipeline for consistent output to
* `results/figures/` and `results/logs/` directories.
* Coordinated Week 11 progress and organized next-phase performance testing tasks.
* Added detailed comments and clarifications inside the benchmark notebook (`final_grid_benchmark.ipynb`) to improve readability, explain logic flow, and support team understanding.

## 📅 Next Milestones
### Pefrformance Testing (Nov 15 - Nov 22)
Conduct batch tests on grid sizes **31×31**, **61×61**, and **91×91**.
* Collect:
  * Runtime measurements
  * Path length results
  * Node-expansion metrics
  * across all algorithms.
* Finalize plotting and comparison results for the final report.
### Final Deliverables (Nov 22 – Dec 2)
* Integrate comparison plots into the final paper.
* Begin drafting:
  * Final Report
  * Presentation Slides (Team 3)

## 📅 Updated Project Timeline

| **Milestone** | **Due Date** | **Status** |
|----------------|--------------|-------------|
| Proposal Submission | Oct 21, 2025 | ✅ Submitted |
| Implementation Phase (A*, Dijkstra, DFS, JPS) | Nov 8, 2025 | ✅ Completed |
| Performance Testing + Visualization | Nov 18, 2025 | ✅ Started (Nov 11 Meeting) |
| Final Report & Presentation | Dec 2, 2025 | ⏳ Upcoming |


## 🧠 Visualization Example
*(Runtime-comparison and search-progress figures will be added after completing batch experiments.)*


## 📜 License

This project is developed for **educational purposes** under the **BCIT COMP 9060 – Applied Algorithm Analysis** course.  
Licensed under the [MIT License](LICENSE).


### 🔗 **Useful Links**

- 📘 [Overleaf Proposal](https://www.overleaf.com/9465635879vhhjjwjkmhzk#37ad93)  
- 📄 [Overleaf Final Report](https://www.overleaf.com/6623247675ghmpxqtkrbhc#20506f)  
- 🗂️ [GitHub Project Board](https://github.com/yourusername/COMP9060-Pathfinding-Optimization/projects)  
- 📊 [Results Dashboard (optional)](https://colab.research.google.com/drive/your-dashboard-link)
