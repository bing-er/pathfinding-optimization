<div align="center">

# 🚀 Pathfinding Algorithms Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

</div>

This repository contains our group project for *COMP 9060 – Advanced Algorithms*, comparing classical and optimized pathfinding algorithms: **A\***, **Dijkstra**, **DFS**, and **JPS (Jump Point Search)**.  
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
By Week 11, the team successfully completed integration and began the **performance testing and visualization phase**. All four algorithms (A*, Dijkstra, DFS, JPS) now run under a unified framework, and the testing plan for runtime and node-expansion benchmarking was finalized.

The repository is now fully organized with a clear folder structure, evaluation metrics, and documentation. Everyone’s roles are defined — **Yansong** handled the baseline algorithms, **Sepehr** is leading the **Jump Point Search (JPS)** development, **Vibhor** is focusing on evaluation and visualization, and **Binger** is managing documentation, scheduling, and overall coordination.

Our next milestone is to integrate and test **JPS**, comparing its performance against the baseline algorithms. The team will also begin logging runtime and node-expansion data and preparing visual outputs for comparison. In the following weeks, we’ll move toward compiling the final report, creating visuals, and getting ready for our presentation in Week **14 (Dec 2)**.

### Week 11 - Post-Integration Update
By **Week 11**, our team successfully completed the **integration phase** of the Pathfinding Optimization project. All four pathfinding algorithms — **A***, **Dijkstra**, **DFS**, and **JPS** — have now been implemented, tested, and unified under a consistent framework.

The repository is fully functional and organized, supporting reproducible experiments, runtime logging, and visualization for performance comparison.

**🧠 Highlights**
* **JPS** algorithm finalized and merged into the main branch.
* **Runtime visualization notebooks** and early performance plots demonstrated in the team meeting.
* **Unified main runner** implemented for all algorithms.
* **Automatic saving of runtime data and figures** into the structured results folders.
* **Enhanced maze generation and visualization** integrated into the main workflow.
* **Performance testing plan finalized** (grid sizes 10×10 → 101×101, sparse vs dense).
* **Visualization notebooks updated** for runtime and node-expansion analysis.
* **Vibhor’s benchmark suite integrated** under notebooks/ for automated comparison.
* **Team reviewed results via meeting (Nov 11, 2025)** — confirmed consistent outputs, JPS shows fewer steps but slightly higher runtime.

#### 👥 Team Contributions
**🧠 Yansong**
* Implemented and verified the **A***, **Dijkstra**, and **DFS** baseline algorithms.
* Ensured algorithm correctness, consistent output, and optimal-path validation.
* Assisted in integration testing and aligning algorithm interfaces.

**⚙️ Sepehr**
* Finalized the **Jump Point Search (JPS)** algorithm with jump + pruning logic.
* Verified performance improvements and integrated JPS into the shared framework.
* Supported testing and comparison with baseline algorithms.

**📊 Vibhor**
* Finalized and pushed **benchmark testing notebooks** (maze_benchmark_corners.ipynb, .py, .html).
* Designed runtime and path-length comparison plan for all algorithms.
* Coordinated visualization data pipeline for performance analysis.

**🧩 Binger**
* Implemented the **main runner** (`main.py`) with unified execution and `--compare` mode for algorithm benchmarking.
* Developed **logging and visualization tools** (`utils.py`, `charts.py`, `runtime_plot.py`) for runtime and path analysis.
* Enhanced **maze generator** (`mazegenerator.py`) to ensure consistent grids across algorithms.
* Configured automatic saving to `results/figures`/ and `results/logs/` for reproducible experiments.
* Organized and maintained the full repository structure for seamless testing and visualization.
* Reviewed and merged visualization branch (viz_vib) into main.
* Organized follow-up testing tasks and documented meeting summary for Week 11.

## 📅 Next Milestones
* Conduct performance testing on larger grid sizes (21×21, 51×51, 101×101).
* Analyze runtime, path length, and node expansions for all algorithms.
* Finalize visualization outputs and integrate comparison figures.
* Begin drafting the **final report** and **presentation slides** for submission (Week 14 – Dec 2, 2025).


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
