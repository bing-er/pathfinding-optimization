# src/visualizations/runtime_plot.py
"""
Author: Binger Yu
"""
import pandas as pd
import matplotlib.pyplot as plt

def plot_runtime_comparison(csv_path="results/logs/runtime_log.csv"):
    df = pd.read_csv(csv_path)

    # Example: average runtime per algorithm
    grouped = df.groupby("algorithm")["runtime"].mean().reset_index()

    print(grouped)

    plt.figure(figsize=(5, 4))
    plt.bar(grouped["algorithm"], grouped["runtime"])
    plt.xlabel("Algorithm")
    plt.ylabel("Average runtime (s)")
    plt.title("Average Runtime by Algorithm")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_runtime_comparison()
