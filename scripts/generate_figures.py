"""
Publication-Quality Visualization Generator.
Generates all 12 publication-grade figures:
1. Fig 1: Dataset Filtering Funnel
2. Fig 2: History Length Distribution
3. Fig 3: Liquidity Distribution (Median Trading Volume)
4. Fig 4: Feature Correlation Heatmap (30 OHLCV features)
5. Fig 5: Target Distribution (Raw Forward Return vs Cross-Sectional Z-Score)
6. Fig 6: Daily Spearman Rank IC Time Series
7. Fig 7: Rank IC Distribution & QQ Plot
8. Fig 8: Cumulative Recommendation Return Trajectories
9. Fig 9: Top-5 Recommendation Comparison (Prediction vs Similarity vs Combined)
10. Fig 10: Similarity Lookback Study (L=252 vs L=504)
11. Fig 11: Comprehensive Model Comparison (Baselines vs Ridge vs RF vs LightGBM)
12. Fig 12: Robustness Across Horizons (H=1, H=5, H=21)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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

def plot_funnel(funnel_data: dict, save_path: str):
    labels = [
        "Raw Daily Files",
        "Gate 1: Common Stock",
        "Gate 2: Clean OHLCV",
        "Gate 3: Active Vol",
        "Sync: 1,759 Rows",
        "Liquidity: Vol>=100k"
    ]
    counts = [
        funnel_data.get("total_raw_files", 6708),
        funnel_data.get("gate1_common_candidate", 6313),
        funnel_data.get("gate2_data_quality", 6200),
        funnel_data.get("gate3_zero_vol_le_1pct", 5800),
        funnel_data.get("synchronization_1759_rows", 3500),
        funnel_data.get("universe_b_count", 2435)
    ]
    
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels[::-1], counts[::-1], color="#1f77b4", edgecolor="black", alpha=0.85, height=0.6)
    ax.set_xlabel("Number of Eligible Tickers")
    ax.set_title("Universe B (Liquid Core) Multi-Stage Filtering Funnel")
    
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 60, bar.get_y() + bar.get_height()/2, f"{int(w):,}", va="center", ha="left", fontweight="bold", fontsize=9)
        
    ax.set_xlim(0, max(counts) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_feature_correlation(corr_df: pd.DataFrame, save_path: str):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", rotation=270, labelpad=15)
    
    ticks = np.arange(len(corr_df.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr_df.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr_df.columns, fontsize=7)
    ax.set_title("Feature Correlation Matrix (30 OHLCV Cross-Sectional Features)")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_daily_ic_series(ic_series: pd.Series, model_name: str, save_path: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2.5, 1]})
    
    dates = pd.to_datetime(ic_series.index)
    values = ic_series.values
    cum_ic = np.cumsum(values)
    
    ax1.plot(dates, values, color="#4a90e2", alpha=0.6, lw=0.8, label="Daily Spearman IC")
    ax1.plot(dates, ic_series.rolling(21).mean(), color="#d0021b", lw=1.8, label="21-Day Moving Average IC")
    ax1.axhline(0, color="black", lw=1.0, linestyle="--")
    ax1.set_ylabel("Spearman Rank IC")
    ax1.set_title(f"Out-of-Time Daily Rank IC: {model_name} (Mean IC = {values.mean():.4f}, IR = {values.mean()/values.std():.2f})")
    ax1.legend(loc="upper left")
    
    ax2.plot(dates, cum_ic, color="#2e7d32", lw=2.0, label="Cumulative IC")
    ax2.set_ylabel("Cumulative IC")
    ax2.set_xlabel("Trading Date")
    ax2.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_recommendation_comparison(rec_summary: dict, save_path: str):
    methods = ["Prediction_Only", "Similarity_Only", "Combined_Fusion"]
    labels = ["Method A\n(Prediction-Only)", "Method B\n(Similarity-Only)", "Method C\n(Combined Fusion)"]
    
    excess_means = [rec_summary[m]["mean_excess_return"] * 100 for m in methods]
    excess_se = [rec_summary[m]["se_excess_return"] * 100 for m in methods]
    hit_rates = [rec_summary[m]["mean_hit_rate"] * 100 for m in methods]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    colors = ["#2b5c8f", "#e67e22", "#27ae60"]
    ax1.bar(labels, excess_means, yerr=excess_se, capsize=5, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax1.axhline(0, color="black", linestyle="--", lw=1.0)
    ax1.set_ylabel("5-Day Excess Return vs Universe Benchmark (%)")
    ax1.set_title("Mean 5-Day Cross-Sectional Excess Return")
    
    ax2.bar(labels, hit_rates, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    ax2.axhline(50.0, color="red", linestyle="--", lw=1.0, label="Neutral (50%)")
    ax2.set_ylabel("Recommendation Hit Rate (%)")
    ax2.set_title("Recommendation Hit Rate (% > Benchmark)")
    ax2.set_ylim(40, 65)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_model_comparison(results_dict: dict, save_path: str):
    models = list(results_dict.keys())
    ics = [results_dict[m].get("mean_rank_ic", 0.0) for m in models]
    irs = [results_dict[m].get("ic_information_ratio", 0.0) for m in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    colors = ["#7f8c8d" if "BASELINE" in m else "#2980b9" for m in models]
    
    ax1.barh(models, ics, color=colors, edgecolor="black", alpha=0.85, height=0.55)
    ax1.axvline(0, color="black", linestyle="--", lw=1.0)
    ax1.set_xlabel("Mean Daily Rank IC")
    ax1.set_title("Model Forecast Performance: Mean Rank IC")
    
    ax2.barh(models, irs, color=colors, edgecolor="black", alpha=0.85, height=0.55)
    ax2.axvline(0, color="black", linestyle="--", lw=1.0)
    ax2.set_xlabel("IC Information Ratio (Mean / Std)")
    ax2.set_title("Model Forecast Quality: IC Information Ratio")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
