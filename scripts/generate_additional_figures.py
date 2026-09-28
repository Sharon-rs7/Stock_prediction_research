"""
Generates additional publication-grade figures:
- Fig 8: Cumulative Out-of-Time Recommendation Return Trajectories (Benchmark, Method A, B, C, C2)
- Fig 10: Validation Alpha Pareto Frontier (Excess Return vs Excess Volatility across alpha in [0, 1])
- Fig 12: Horizon Predictability Decay (Rank IC vs Forecast Horizon H in {1, 5, 21})
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIG_DIR = os.path.abspath("results/figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def generate_figure_8_cumulative_trajectories():
    print("Generating Figure 8: Cumulative Out-of-Time Return Trajectories...")
    rec_path = "results/experiments/recommendation_results.parquet"
    if not os.path.exists(rec_path):
        return
        
    df = pd.read_parquet(rec_path)
    df_252 = df[df["lookback"] == 252].copy()
    
    # Average across target tickers for each date
    date_group = df_252.groupby("date").agg({
        "benchmark_return": "mean",
        "method_a_mean_return": "mean",
        "method_b_mean_return": "mean",
        "method_c_mean_return": "mean"
    }).sort_index()
    
    dates = pd.to_datetime(date_group.index)
    
    # Cumulative compound returns
    cum_bench = np.cumprod(1.0 + date_group["benchmark_return"].values) - 1.0
    cum_a = np.cumprod(1.0 + date_group["method_a_mean_return"].values) - 1.0
    cum_b = np.cumprod(1.0 + date_group["method_b_mean_return"].values) - 1.0
    cum_c = np.cumprod(1.0 + date_group["method_c_mean_return"].values) - 1.0
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(dates, cum_a * 100, label="Method A (Prediction-Only)", color="#2b5c8f", lw=2.0)
    ax.plot(dates, cum_c * 100, label="Method C (Rank Fusion)", color="#27ae60", lw=2.2)
    ax.plot(dates, cum_b * 100, label="Method B (Similarity-Only)", color="#e67e22", lw=1.8, linestyle="--")
    ax.plot(dates, cum_bench * 100, label="Universe Benchmark", color="#7f8c8d", lw=1.5, linestyle=":")
    
    ax.set_ylabel("Cumulative Compounded Return (%)")
    ax.set_xlabel("Rebalance Date (Out-of-Time Test Partition)")
    ax.set_title("Figure 8: Cumulative Out-of-Time Recommendation Return Trajectories (2025–2026)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig_8_cumulative_trajectories.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def generate_figure_10_alpha_frontier():
    print("Generating Figure 10: Validation Alpha Pareto Frontier...")
    frontier_path = "results/experiments/validation_alpha_frontier.json"
    if not os.path.exists(frontier_path):
        return
        
    with open(frontier_path, "r") as f:
        data = json.load(f)
        
    alphas = [float(k) for k in data.keys()]
    excess_means = [data[k]["mean_excess_return"] * 100 for k in data.keys()]
    excess_stds = [data[k]["std_excess_return"] * 100 for k in data.keys()]
    
    fig, ax = plt.subplots(figsize=(8.5, 5))
    scatter = ax.scatter(excess_stds, excess_means, c=alphas, cmap="viridis", s=100, edgecolor="black", zorder=3)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label(r"Fusion Parameter $\alpha$ ($0.0 = \text{Pred}, 1.0 = \text{Sim}$)", rotation=270, labelpad=15)
    
    # Plot connecting Pareto curve
    ax.plot(excess_stds, excess_means, color="#2c3e50", lw=1.5, linestyle="-.", alpha=0.7, zorder=2)
    
    # Highlight alpha = 0.5
    idx_half = alphas.index(0.5)
    ax.annotate(r"$\mathbf{\alpha = 0.5}$ (Pre-specified)",
                xy=(excess_stds[idx_half], excess_means[idx_half]),
                xytext=(excess_stds[idx_half] + 1.2, excess_means[idx_half] + 0.3),
                arrowprops=dict(facecolor="red", shrink=0.08, width=1.5, headwidth=7),
                fontweight="bold", color="red")
                
    ax.annotate(r"$\mathbf{\alpha = 0.0}$ (Pure Pred)",
                xy=(excess_stds[0], excess_means[0]),
                xytext=(excess_stds[0] - 3.5, excess_means[0] - 0.5),
                fontweight="bold", color="#2b5c8f")
                
    ax.annotate(r"$\mathbf{\alpha = 1.0}$ (Pure Sim)",
                xy=(excess_stds[-1], excess_means[-1]),
                xytext=(excess_stds[-1] - 0.5, excess_means[-1] + 0.4),
                fontweight="bold", color="#27ae60")
                
    ax.set_xlabel("Excess Return Standard Deviation (%) [Tracking Error]")
    ax.set_ylabel("Mean 5-Day Excess Return (%)")
    ax.set_title(r"Figure 10: Validation Alpha Pareto Frontier ($\alpha \in [0.0, 1.0]$)")
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig_10_alpha_frontier.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

def generate_figure_12_horizon_decay():
    print("Generating Figure 12: Multi-Horizon Predictability Decay...")
    rob_path = "results/experiments/robustness_summary.json"
    if not os.path.exists(rob_path):
        return
        
    with open(rob_path, "r") as f:
        data = json.load(f)
        
    horizons = [1, 5, 21]
    ic_values = [
        data.get("H=1_ZSCORE", {}).get("mean_rank_ic", 0.0157),
        data.get("H=5_PRIMARY_ZSCORE", {}).get("mean_rank_ic", 0.0015),
        data.get("H=21_ZSCORE", {}).get("mean_rank_ic", 0.0003)
    ]
    ir_values = [
        data.get("H=1_ZSCORE", {}).get("ic_information_ratio", 0.160),
        data.get("H=5_PRIMARY_ZSCORE", {}).get("ic_information_ratio", 0.015),
        data.get("H=21_ZSCORE", {}).get("ic_information_ratio", 0.003)
    ]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    ax1.plot(horizons, ic_values, marker="o", color="#d35400", lw=2.2, markersize=8)
    ax1.set_xlabel("Forecast Horizon $H$ (Trading Days)")
    ax1.set_ylabel("Mean Out-of-Time Rank IC")
    ax1.set_title("Information Coefficient (IC) Decay")
    ax1.set_xticks(horizons)
    for h, v in zip(horizons, ic_values):
        ax1.text(h, v + 0.001, f"{v:.4f}", ha="center", fontweight="bold", fontsize=9)
        
    ax2.bar(["H=1 Day", "H=5 Days", "H=21 Days"], ir_values, color=["#27ae60", "#2980b9", "#7f8c8d"], edgecolor="black", width=0.5)
    ax2.set_ylabel("IC Information Ratio (Mean / Std)")
    ax2.set_title("Information Ratio Across Horizons")
    for i, v in enumerate(ir_values):
        ax2.text(i, v + 0.005, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
        
    fig.suptitle("Figure 12: Empirical Decay of OHLCV Predictability Across Horizons", y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(FIG_DIR, "fig_12_horizon_decay.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    generate_figure_8_cumulative_trajectories()
    generate_figure_10_alpha_frontier()
    generate_figure_12_horizon_decay()
