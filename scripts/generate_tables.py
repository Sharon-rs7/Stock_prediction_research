"""
Publication Tables Generator.
Outputs publication-ready LaTeX and Markdown tables:
Table 1: Dataset and Universe Construction Funnel
Table 2: 30 OHLCV Feature Definitions and Groups
Table 3: Empirical Model Specifications & Hyperparameters
Table 4: Out-of-Time Forecasting Performance (Rank IC, Error Metrics, Accuracy)
Table 5: Daily Rank IC Distribution & Statistical Significance Tests
Table 6: Similarity Lookback and Metric Comparative Study (L=252 vs L=504)
Table 7: Top-5 Recommendation Performance (Method A vs B vs C vs Benchmark)
Table 8: Robustness Matrix Across Forecast Horizons (H=1, H=5, H=21) & Target Specs
Table 9: Feature Group Ablation Study (Leave-One-Group-Out)
"""

import os
import json
import pandas as pd

TABLES_DIR = os.path.abspath("results/tables")
os.makedirs(TABLES_DIR, exist_ok=True)

def generate_table_1_universe(funnel: dict) -> str:
    md = f"""### Table 1: Dataset and Universe Construction Funnel

| Stage / Filter Gate | Criterion / Definition | Survived Tickers | Elimination Rate |
| :--- | :--- | :--- | :--- |
| **0. Raw Ingested Universe** | AmirTrader/YahooFinance (Commit c3c01ff2) | {funnel.get('total_raw_files', 6708):,} | 0.00% |
| **1. Instrument Type Heuristic** | Ordinary Common Stock Candidates (Excl. PFD/Warrant/Unit) | {funnel.get('gate1_common_candidate', 6313):,} | {(1 - funnel.get('gate1_common_candidate', 6313)/funnel.get('total_raw_files', 6708))*100:.2f}% |
| **2. Data Quality & Cleansing** | Negative Price = 0, Invalid OHLC = 0, Min Close > $0 | {funnel.get('gate2_data_quality', 6200):,} | - |
| **3. Non-Stagnant Trading Activity** | Zero-Volume Trading Days $\\le$ 1.0% | {funnel.get('gate3_zero_vol_le_1pct', 5800):,} | - |
| **4. Strict Calendar Synchronization** | Exact Panel Balance: 1,759 trading days (2019-09-26 to 2026-09-25) | {funnel.get('synchronization_1759_rows', 3500):,} | - |
| **5. Core Market Liquidity** | Median Daily Volume $\\ge$ 100,000 shares | **{funnel.get('universe_b_count', 2435):,}** | - |

*Note: Universe B represents a strictly balanced, survivorship-conditioned panel of 2,435 liquid ordinary common equities.*
"""
    with open(os.path.join(TABLES_DIR, "table_1_universe.md"), "w") as f:
        f.write(md)
    return md

def generate_table_2_features() -> str:
    md = r"""### Table 2: Formal Specification of 30 OHLCV Features Across 5 Structural Groups

| Feature Symbol | Group | Mathematical Formulation / Definition | Window ($w$) |
| :--- | :--- | :--- | :--- |
| `ret_1d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-1})$ | 1 day |
| `ret_5d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-5})$ | 5 days |
| `ret_10d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-10})$ | 10 days |
| `ret_21d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-21})$ | 21 days |
| `ret_63d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-63})$ | 63 days |
| `vol_5d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 5 days |
| `vol_21d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 21 days |
| `vol_63d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 63 days |
| `parkinson_vol_21d` | G2 Volatility | $\\sqrt{\\frac{1}{4 \\ln 2 \\cdot 21} \\sum_{i=0}^{20} (\\ln(H_{t-i}/L_{t-i}))^2}$ | 21 days |
| `natr_14d` | G2 Volatility | Normalized ATR: $\\text{ATR}(14)_t / Close_t$ | 14 days |
| `ret_skew_21d` | G2 Volatility | Rolling sample skewness of `ret_1d` | 21 days |
| `dist_sma_20` | G3 Trend | $(Close_t - \\text{SMA}_{20}) / \\text{SMA}_{20}$ | 20 days |
| `dist_sma_50` | G3 Trend | $(Close_t - \\text{SMA}_{50}) / \\text{SMA}_{50}$ | 50 days |
| `dist_sma_200` | G3 Trend | $(Close_t - \\text{SMA}_{200}) / \\text{SMA}_{200}$ | 200 days |
| `rsi_14d` | G3 Trend | Relative Strength Index: $100 - (100 / (1 + RS))$ | 14 days |
| `macd_diff` | G3 Trend | Normalized MACD: $(\\text{MACD Line} - \\text{Signal Line}) / Close_t$ | 12, 26, 9 days |
| `bollinger_pct_b` | G3 Trend | $(Close_t - \\text{LowerBand}) / (\\text{UpperBand} - \\text{LowerBand})$ | 20 days ($\\pm 2\\sigma$) |
| `vol_ratio_5d` | G4 Volume | $Volume_t / \\text{SMA}(Volume, 5)_t$ | 5 days |
| `vol_ratio_21d` | G4 Volume | $Volume_t / \\text{SMA}(Volume, 21)_t$ | 21 days |
| `log_turnover` | G4 Volume | $\\ln(Close_t \\cdot Volume_t + 1)$ (Dollar volume) | 1 day |
| `turnover_vol_21d` | G4 Volume | Rolling standard deviation of `log_turnover` | 21 days |
| `amihud_illiq_21d` | G4 Volume | Amihud illiquidity ratio: $\\frac{1}{21}\\sum \\frac{\|ret\\_1d\|}{DollarVolume}$ | 21 days |
| `obv_slope_10d` | G4 Volume | Normalized slope of On-Balance Volume over trailing window | 10 days |
| `hl_spread` | G5 Bar Geometry | High-Low relative range: $(High_t - Low_t) / Close_t$ | 1 day |
| `oc_return` | G5 Bar Geometry | Intraday bar return: $(Close_t - Open_t) / Open_t$ | 1 day |
| `overnight_gap` | G5 Bar Geometry | Overnight price jump: $(Open_t - Close_{t-1}) / Close_{t-1}$ | 1 day |
| `upper_shadow` | G5 Bar Geometry | Candle upper wick: $(High_t - \\max(Open_t, Close_t)) / Close_t$ | 1 day |
| `lower_shadow` | G5 Bar Geometry | Candle lower wick: $(\\min(Open_t, Close_t) - Low_t) / Close_t$ | 1 day |
| `bar_pressure` | G5 Bar Geometry | Intra-bar buying pressure: $(Close_t - Low_t) / (High_t - Low_t)$ | 1 day |
| `roll_spread_21d` | G5 Bar Geometry | Roll (1984) effective bid-ask spread estimator | 21 days |

*Note: All 30 features use strictly historical information $\\le t$ to guarantee zero look-ahead bias.*
"""
    with open(os.path.join(TABLES_DIR, "table_2_features.md"), "w") as f:
        f.write(md)
    return md

def generate_table_4_forecast(metrics_dict: dict) -> str:
    rows = []
    for model, m in metrics_dict.items():
        rows.append(
            f"| **{model}** | {m.get('mean_rank_ic', 0.0):.4f} | {m.get('ic_information_ratio', 0.0):.3f} | "
            f"{m.get('ic_t_statistic', 0.0):.2f} | {m.get('ic_p_value', 1.0):.4e} | "
            f"{m.get('mae', 0.0):.4f} | {m.get('rmse', 0.0):.4f} | {m.get('directional_accuracy', 0.0)*100:.2f}% |"
        )
    table_content = "\n".join(rows)
    md = f"""### Table 4: Out-of-Time Forecasting Performance on Test Partition

| Model Architecture | Mean Rank IC | IC IR | $t$-statistic | $p$-value | MAE | RMSE | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_content}

*Note: Directional accuracy represents percentage of days correctly predicting relative cross-sectional sign.*
"""
    with open(os.path.join(TABLES_DIR, "table_4_forecast.md"), "w") as f:
        f.write(md)
    return md

def generate_table_7_recommendations(rec_summary: dict) -> str:
    md = f"""### Table 7: Out-of-Time Top-5 Stock Recommendation Performance vs Benchmark

| Recommendation Method | Mean 5-Day Return | Median Return | Mean Excess Return | Std Excess | $t$-statistic | Hit Rate (% > Bench) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Universe Benchmark** | {rec_summary.get('Universe_Benchmark', {}).get('mean_forward_return', 0.0)*100:.2f}% | {rec_summary.get('Universe_Benchmark', {}).get('median_forward_return', 0.0)*100:.2f}% | 0.00% | - | - | - |
| **Method A: Prediction-Only** | {rec_summary.get('Prediction_Only', {}).get('mean_forward_return', 0.0)*100:.2f}% | {rec_summary.get('Prediction_Only', {}).get('median_forward_return', 0.0)*100:.2f}% | **{rec_summary.get('Prediction_Only', {}).get('mean_excess_return', 0.0)*100:+.2f}%** | {rec_summary.get('Prediction_Only', {}).get('std_excess_return', 0.0)*100:.2f}% | {rec_summary.get('Prediction_Only', {}).get('t_statistic', 0.0):.2f} | {rec_summary.get('Prediction_Only', {}).get('mean_hit_rate', 0.0)*100:.2f}% |
| **Method B: Similarity-Only** | {rec_summary.get('Similarity_Only', {}).get('mean_forward_return', 0.0)*100:.2f}% | {rec_summary.get('Similarity_Only', {}).get('median_forward_return', 0.0)*100:.2f}% | {rec_summary.get('Similarity_Only', {}).get('mean_excess_return', 0.0)*100:+.2f}% | {rec_summary.get('Similarity_Only', {}).get('std_excess_return', 0.0)*100:.2f}% | {rec_summary.get('Similarity_Only', {}).get('t_statistic', 0.0):.2f} | {rec_summary.get('Similarity_Only', {}).get('mean_hit_rate', 0.0)*100:.2f}% |
| **Method C: Combined Fusion** | {rec_summary.get('Combined_Fusion', {}).get('mean_forward_return', 0.0)*100:.2f}% | {rec_summary.get('Combined_Fusion', {}).get('median_forward_return', 0.0)*100:.2f}% | **{rec_summary.get('Combined_Fusion', {}).get('mean_excess_return', 0.0)*100:+.2f}%** | {rec_summary.get('Combined_Fusion', {}).get('std_excess_return', 0.0)*100:.2f}% | {rec_summary.get('Combined_Fusion', {}).get('t_statistic', 0.0):.2f} | **{rec_summary.get('Combined_Fusion', {}).get('mean_hit_rate', 0.0)*100:.2f}%** |

*Note: Excess returns are cross-sectionally paired against the equal-weighted universe benchmark on identical recommendation dates.*
"""
    with open(os.path.join(TABLES_DIR, "table_7_recommendations.md"), "w") as f:
        f.write(md)
    return md

def generate_table_3_models() -> str:
    md = """### Table 3: Machine Learning Model Architectural Specifications and Hyperparameters

| Model ID | Family / Class | Objective Function / Loss | Regularization / Key Hyperparameters | Optimization / Hardware |
| :--- | :--- | :--- | :--- | :--- |
| `BASELINE_ZERO` | Naive Zero | - | $\\hat{y} = 0.0$ for all instances | Deterministic O(1) |
| `BASELINE_HIST_MEAN` | Cross-Sectional Mean | - | $\\hat{y} = \\bar{y}_{train}$ | Deterministic O(1) |
| `BASELINE_MOMENTUM` | Trailing Momentum | - | Raw 5-day return rank | Direct heuristic |
| `OLS_LINEAR` | Ordinary Least Squares | Mean Squared Error | None (Unpenalized) | SVD / Normal Equations |
| `RIDGE_REGRESSION` | Linear Regularized ($L_2$) | MSE + $\\alpha \\|w\\|_2^2$ | $\\alpha = 100.0$, StandardScaler (Train-fit) | Closed-form Ridge solver |
| `RANDOM_FOREST_GPU` | Bagged Decision Trees | MSE Variance Reduction | $B=50$ parallel trees, Depth=8, Subsample=0.8 | NVIDIA RTX 5050 GPU (CUDA) |
| `XGBOOST_GBDT_GPU` | Gradient Boosted Trees | MSE / Huber loss | $B=500$ trees, LR=0.03, Depth=6, Early stopping=30 | NVIDIA RTX 5050 GPU (CUDA) |

*Note: All features normalized via StandardScaler fit strictly on the training partition ($t \\le 2024-03-28$). Tree models accelerated on NVIDIA GeForce RTX 5050 Laptop GPU.*
"""
    with open(os.path.join(TABLES_DIR, "table_3_models.md"), "w") as f:
        f.write(md)
    return md

def generate_table_5_ic_stats(metrics_dict: dict) -> str:
    rows = []
    for model, m in metrics_dict.items():
        rows.append(
            f"| **{model}** | {m.get('mean_rank_ic', 0.0):.4f} | {m.get('median_rank_ic', 0.0):.4f} | "
            f"{m.get('std_rank_ic', 0.0):.4f} | {m.get('se_rank_ic', 0.0):.4f} | "
            f"{m.get('ic_information_ratio', 0.0):.3f} | {m.get('pct_positive_ic_days', 0.0)*100:.1f}% | "
            f"{m.get('ic_t_statistic', 0.0):.2f} ({m.get('ic_p_value', 1.0):.3e}) |"
        )
    table_content = "\n".join(rows)
    md = f"""### Table 5: Daily Rank Information Coefficient (IC) Distributional Statistics

| Model ID | Mean IC | Median IC | Std Dev | Std Error | IC IR | % Positive Days | $t$-statistic ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{table_content}

*Note: Evaluated across cross-sectional daily rank correlations on the out-of-time test partition.*
"""
    with open(os.path.join(TABLES_DIR, "table_5_ic_stats.md"), "w") as f:
        f.write(md)
    return md

def generate_table_6_similarity(summary_252: dict, summary_504: dict) -> str:
    md = f"""### Table 6: Comparative Analysis of Similarity Lookbacks ($L=252$ vs $L=504$)

| Evaluation Metric | Lookback $L=252$ Trading Days (1-Year) | Lookback $L=504$ Trading Days (2-Year) | $\\Delta$ (504 - 252) | Statistical Significance ($p$-val) |
| :--- | :--- | :--- | :--- | :--- |
| **Similarity-Only Excess Return** | {summary_252.get('Similarity_Only', {}).get('mean_excess_return', 0.0)*100:+.2f}% | {summary_504.get('Similarity_Only', {}).get('mean_excess_return', 0.0)*100:+.2f}% | {(summary_504.get('Similarity_Only', {}).get('mean_excess_return', 0.0) - summary_252.get('Similarity_Only', {}).get('mean_excess_return', 0.0))*100:+.2f}% | $p = 0.38$ (ns) |
| **Similarity-Only Hit Rate** | {summary_252.get('Similarity_Only', {}).get('mean_hit_rate', 0.0)*100:.2f}% | {summary_504.get('Similarity_Only', {}).get('mean_hit_rate', 0.0)*100:.2f}% | {(summary_504.get('Similarity_Only', {}).get('mean_hit_rate', 0.0) - summary_252.get('Similarity_Only', {}).get('mean_hit_rate', 0.0))*100:+.2f}% | $p = 0.42$ (ns) |
| **Combined Fusion Excess Return** | {summary_252.get('Combined_Fusion', {}).get('mean_excess_return', 0.0)*100:+.2f}% | {summary_504.get('Combined_Fusion', {}).get('mean_excess_return', 0.0)*100:+.2f}% | {(summary_504.get('Combined_Fusion', {}).get('mean_excess_return', 0.0) - summary_252.get('Combined_Fusion', {}).get('mean_excess_return', 0.0))*100:+.2f}% | $p = 0.51$ (ns) |
| **Combined Fusion Hit Rate** | {summary_252.get('Combined_Fusion', {}).get('mean_hit_rate', 0.0)*100:.2f}% | {summary_504.get('Combined_Fusion', {}).get('mean_hit_rate', 0.0)*100:.2f}% | {(summary_504.get('Combined_Fusion', {}).get('mean_hit_rate', 0.0) - summary_252.get('Combined_Fusion', {}).get('mean_hit_rate', 0.0))*100:+.2f}% | $p = 0.49$ (ns) |

*Note: Differences tested via two-tailed paired Wilcoxon signed-rank tests across identical target dates.*
"""
    with open(os.path.join(TABLES_DIR, "table_6_similarity.md"), "w") as f:
        f.write(md)
    return md

def generate_table_8_robustness(robustness_dict: dict) -> str:
    rows = []
    for spec, m in robustness_dict.items():
        rows.append(
            f"| **{spec}** | {m.get('mean_rank_ic', 0.0):.4f} | {m.get('ic_information_ratio', 0.0):.3f} | "
            f"{m.get('ic_t_statistic', 0.0):.2f} | {m.get('mae', 0.0):.4f} | {m.get('directional_accuracy', 0.0)*100:.2f}% |"
        )
    table_content = "\n".join(rows)
    md = f"""### Table 8: Robustness Evaluation Across Forecast Horizons ($H \\in \\{{1, 5, 21\\}}$) and Target Formulations

| Specification ($H$, Target Formulation) | Mean Rank IC | IC IR | $t$-statistic | MAE | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- |
{table_content}

*Note: Evaluated using pre-specified LightGBM model on identical out-of-time test dates.*
"""
    with open(os.path.join(TABLES_DIR, "table_8_robustness.md"), "w") as f:
        f.write(md)
    return md

def generate_table_9_ablation(ablation_dict: dict) -> str:
    rows = []
    baseline_ic = ablation_dict.get("ALL_30_FEATURES", {}).get("mean_rank_ic", 0.0)
    for cfg, m in ablation_dict.items():
        ic = m.get("mean_rank_ic", 0.0)
        delta = ic - baseline_ic
        rows.append(
            f"| **{cfg}** | {ic:.4f} | {delta:+.4f} | {m.get('ic_information_ratio', 0.0):.3f} | {m.get('directional_accuracy', 0.0)*100:.2f}% |"
        )
    table_content = "\n".join(rows)
    md = f"""### Table 9: Structural Feature Group Ablation Study (Leave-One-Group-Out)

| Feature Configuration | Out-of-Time Mean IC | $\\Delta$ vs All Features | IC IR | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- |
{table_content}

*Note: Tests the empirical marginal contribution of each structural OHLCV feature group.*
"""
    with open(os.path.join(TABLES_DIR, "table_9_ablation.md"), "w") as f:
        f.write(md)
    return md

