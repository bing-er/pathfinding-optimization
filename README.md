<div align="center">

# 🚀 Pathfinding Algorithms Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

</div>

This repository contains our group project for *COMP 9060 – Advanced Algorithms*, comparing classical and optimized pathfinding algorithms: **A\***, **Dijkstra**, **DFS**, and **JPS (Jump Point Search)**.  
The study focuses on performance, path optimality, and efficiency across different grid-based environments.

---

## 👥 Team Members
| Name | Role |
|------|------|
| **a** | Algorithm Developer – A*, Dijkstra, DFS Implementation |
| **b** | QA & Testing Lead – JPS Implementation and Integration |
| **c** | Evaluation Lead – Metrics Analysis and Visualization |
| **d** | Project Manager – Documentation, Reporting, and Presentation |

---

## 🎯 Project Overview
Pathfinding is a fundamental problem in AI, robotics, and game development.
This project aims to:
- Implement **A***, **Dijkstra**, **DFS**, and **JPS** algorithms in a common grid framework.
- Evaluate their performance on various **grid configurations** (sparse vs dense, small vs large).
- Measure **runtime**, **path cost**, and **node expansions** to analyze algorithmic efficiency.
- Visualize algorithm behavior through comparative charts and heatmaps.
**Jump Point Search (JPS)** improves **A*** by **skipping redundant nodes** in uniform-cost grids, reducing runtime while preserving optimal path cost.

---

## 🗂️ Repository Structure
```
project/
├── src/                     # Source code for all algorithms
│   ├── algorithms/          # Pathfinding algorithm implementations
│   │   ├── astar.py         # A* baseline algorithm
│   │   ├── dijkstra.py      # Dijkstra baseline algorithm
│   │   ├── dfs.py           # Depth-First Search baseline
│   │   └── jps.py           # Jump Point Search (JPS) implementation
│   ├── core/                # Shared components
│   │   ├── grid.py          # Grid representation and movement rules
│   │   ├── heuristics.py    # Heuristic functions (Manhattan, Octile, etc.)
│   │   └── utils.py         # Utility functions (logging, timers, helpers)
│   ├── visualizations/      # Visualization and performance analysis
│   │   └── charts.py        # Plots for runtime, node expansions, path cost
│   └── main.py              # Entry point to run and compare algorithms
│
├── data/                    # Sample grid maps and test cases
│   └── maps/                # Example .txt or .csv grid files
│
├── results/                 # Experiment outputs, logs, and performance data
│   ├── logs/                # Raw runtime and node expansion logs
│   └── figures/             # Generated charts and comparison graphs
│
├── docs/                    # Documentation and reports
│   ├── proposal.pdf         # Submitted project proposal
│   ├── report_draft.docx    # In-progress final report
│   └── slides.pptx          # Presentation slides
│
└── README.md                # Project overview and usage instructions
```

---

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


## 🗓️ Project Timeline

| Milestone                                  | Due Date     |
| ------------------------------------------ | ------------ |
| Proposal Submission                        | Oct 21, 2025 |
| Implementation Phase (A*, Dijkstra, JPS)** | Nov 10, 2025 |
| Final Report & Presentation                | Dec 2, 2025  |


### 🧠 Visualization Example

## License

This project is developed for **educational purposes** under the **BCIT COMP 9060 – Applied Algorithm Analysis** course.
Licensed under the MIT License

### 🔗 **Useful Links**

- 📘 [Overleaf Proposal](https://www.overleaf.com/project/68f7c8bb19ca6a542e10049d)
- 📄 [Overleaf Final Report](https://www.overleaf.com/project/68f7ed08691b69b5859774ee)  
- 🗂️ [GitHub Project Board](https://github.com/bing-er/pathfinding-optimization)  
- 📊 [Results Dashboard
