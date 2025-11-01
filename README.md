# 🚀 Pathfinding Algorithms Comparison

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow.svg)]()

This repository contains our group project for **COMP 9060 – Advanced Algorithms**, comparing classical and optimized pathfinding algorithms: **A\***, **Dijkstra**, **DFS**, and **JPS (Jump Point Search)**.  
The study focuses on **performance**, **path optimality**, and **efficiency** across different grid-based environments.

---

## 👥 Team Members
| Name | Role |
|------|------|
| **a** | Research & Implementation |
| **b** | Proposal Writing & Documentation |
| **c** | Algorithm Design & JPS Implementation |
| **d** | Baseline Testing & Performance Evaluation |

---

## 🎯 Project Overview
Pathfinding is a fundamental problem in AI and robotics.  
This project aims to:
- Implement **A\***, **Dijkstra**, **DFS**, and **JPS** algorithms.
- Evaluate their performance on various **grid maps**.
- Measure **computation time**, **path length**, and **scalability**.
- Visualize algorithm behavior for better comparison.

**Jump Point Search (JPS)** improves A\* by skipping redundant nodes in uniform grids, significantly reducing search time while maintaining optimality.

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
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Set Up Environment
```python3 -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Run Algorithms
```python src/a_star.py
python src/dijkstra.py
python src/dfs.py
python src/jps.py
```

### 4. Visualize Results
Generated logs and visualizations will appear in the results/ folder.

### 📊 Evaluation Metrics

| Metric               | Description                      |
| -------------------- | -------------------------------- |
| **Path Length**      | Total distance of computed route |
| **Computation Time** | Time required to reach goal      |
| **Node Expansions**  | Number of explored nodes         |
| **Scalability**      | Performance on larger grid maps  |


## 🗓️ Project Timeline

| Milestone                   | Due Date     |
| --------------------------- | ------------ |
| Proposal Submission         | Oct 21, 2025 |
| Midterm Progress Review     | Nov 10, 2025 |
| Final Report & Presentation | Dec 2, 2025  |


### 🧠 Visualization Example

## License

This project is for educational purposes only under the BCIT course COMP XXXX.
Licensed under the MIT License

### 🔗 Useful Links

📘 Overleaf Proposal

🗂️ GitHub Project Board

📊 Results Dashboard (optional)

