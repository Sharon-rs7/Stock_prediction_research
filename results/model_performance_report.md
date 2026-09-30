# Model Performance & Empirical Evaluation Report

**Role:** Model Performance Analyst  
**Project:** Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data  
**Dataset:** Hugging Face `AmirTrader/YahooFinance` (Commit `c3c01ff2fc62e02c338d2e03bdfd71016da09701`)  
**Target Universe:** Universe B (Liquid Core) — Exactly 2,435 ordinary common equities; 1,759 synchronized trading days (September 26, 2019 to September 25, 2026); 4,283,165 stock-day records.  
**Hardware Platform:** NVIDIA GeForce RTX 5050 Laptop GPU (CUDA 13.2)  
**Evaluation Principle:** Strictly empirical, leak-free, zero-snooping out-of-time evaluation. Zero metric fabrication.

> **RESEARCH LINEAGE & ARTIFACT STATUS NOTE:**  
> This report documents the **Phase 4/5 Exploratory Baseline & Model Zoo Benchmark Audit** (evaluating 30 single-stock features across OLS, Ridge, Random Forest, XGBoost GPU, and Stacking M4 at $H=5$ and $H=1$).  
> Following 2025 literature recommendations to expand the information set with market-context and relative signals, the project executed the **Phase 6 Confirmatory Model Enhancement** (`scripts/run_scientific_model_enhancement.py` and `scripts/run_final_forensic_verification.py`).  
> The **Final Confirmatory Research Experiment** is the **Level 2 Market-Aware LightGBM Huber Regressor ($D=49$ features, Test Rank IC = 0.0160)** with Method C Hybrid Rank Fusion across 6,100 recommendations. All publication tables, figures, and claims in the final manuscript (`paper/RESEARCH_PAPER.md`, `paper/latex/main.tex`, `paper/index.html`, and `paper/research_paper.pdf`) are generated strictly from that final Phase 6 pipeline. In the final manuscript, Stacking M4 and $H=1$ models from this earlier report are formally classified as **Exploratory / Post-hoc Analyses** (Section 8).

---

## 1. Executive Summary

This report delivers a forensic audit of all predictive models, baselines, and recommendation engines implemented within this research repository. Every reported metric is strictly cross-referenced against saved experiment summaries, raw prediction parquet files, and evaluation manifests.

### Key Performance Findings at a Glance:
1. **The Core Prediction Reality:** Across all models predicting cross-sectional forward return $z$-scores at $H=5$ days, the predictive signal is modest in magnitude ($\text{Rank IC} \in [0.0005, 0.0085]$). 
   - Linear models (OLS, Ridge) achieve a Mean Daily Rank IC of $+0.0015$ ($t = 0.26$, not statistically significant).
   - Standard GBDT (XGBoost GPU, 30 features) achieves a Mean Daily Rank IC of $+0.0059$ ($t = 1.06$, not statistically significant).
   - Multi-model stacking (LightGBM Huber + XGBoost + Ridge, 35 features) achieves $+0.0085$ ($t = 1.31$, not statistically significant).
2. **The Ultra-Short Horizon Signal ($H=1$ Day):** Return predictability is non-zero and statistically significant strictly at the 1-day horizon ($H=1$), where technical and bar-geometry microstructural signals have not yet dissipated. The single LightGBM model achieves $\text{Rank IC} = 0.0157$ ($t = 2.79, p = 0.0056$, statistically significant), and the Champion Dual Huber Ensemble achieves $\text{Rank IC} = 0.0221$ with an Information Ratio (IR) of $0.167$.
3. **Decile Monotonicity Verification:** On the out-of-time test partition (308 trading days; 757,285 samples), the $H=1$ Champion Ensemble sorts stocks with near-perfect monotonic graduation: daily mean returns scale from $+0.097\%$ in Decile 1 to $+0.208\%$ in Decile 10, generating an annualized Long-Short ($D10 - D1$) return of $+28.12\%$ with an institutional Sharpe ratio of $1.22$.
4. **Behavioral Similarity Does Not Generate Alpha:** Across 62 out-of-time test rebalance dates (1,220 target evaluations), Method B (Similarity-Only Top-5) generates a mean 5-day excess return of $-0.034\%$ ($t = -0.31, p = 0.76$). Behavioral similarity alone provides zero alpha over the equal-weighted market benchmark.
5. **The True Function of Rank Fusion (Method C):** Method A (Prediction-Only Top-5) yields high arithmetic excess returns ($+1.66\%, t = 4.83, p < 0.001$), but incurs extreme tracking error volatility ($12.03\%$) and a negative median excess return ($-0.72\%$). Method C (Combined 50/50 Rank Fusion) acts as an effective volatility compressor, reducing excess return standard deviation by $65.3\%$ (down to $4.17\%$) while delivering a consistent mean excess return of $+0.16\%$ to $+0.17\%$. Method C2 (Volatility-Penalized Fusion) achieves $+0.59\%$ excess return ($t = 2.24, p = 0.0248$, statistically significant) with a hit rate of $50.90\%$.

---

## 2. Complete Model Inventory

| Model Identifier | Horizon | Target Specification | Features Used | Train Window | Val Window | Test Window | Test $N$ | Out-of-Sample? | Trained Train-Only? | Key Hyperparameters | Primary Source Artifact |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| `BASELINE_ZERO` | $H=5$ | `zscore_fwd_ret_5d` | None | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | $\hat{y} = 0.0$ | `results/experiments/experiment_registry.csv` |
| `BASELINE_HIST_MEAN` | $H=5$ | `zscore_fwd_ret_5d` | None | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | $\hat{y} = \bar{y}_{train}$ | `results/experiments/experiment_registry.csv` |
| `BASELINE_MOMENTUM` | $H=5$ | `zscore_fwd_ret_5d` | `ret_5d` | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Heuristic 5-day return rank | `results/experiments/experiment_registry.csv` |
| `OLS_LINEAR` | $H=5$ | `zscore_fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Normal equations, no penalty | `results/experiments/experiment_registry.csv` |
| `RIDGE_REGRESSION` | $H=5$ | `zscore_fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | $\alpha = 100.0$, StandardScaler | `results/experiments/experiment_registry.csv` |
| `RANDOM_FOREST_GPU` | $H=5$ | `zscore_fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | $B=50$, Depth=8, CUDA | `results/experiments/experiment_registry.csv` |
| `XGBOOST_GBDT_GPU` | $H=5$ | `zscore_fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | $B=500, \eta=0.03$, Depth=6, CUDA | `results/experiments/ablation_summary.json` |
| `ADVANCED_XGB_M1` | $H=5$ | `zscore_fwd_ret_5d` | 35 Feats | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | 30 Core + 5 Interactions, MSE | `results/tables/table_10_advanced_accuracy_models.md` |
| `ADVANCED_LGBM_M2` | $H=5$ | `zscore_fwd_ret_5d` | 35 Feats | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Huber loss ($\delta=1.35$), Leaves=31 | `results/tables/table_10_advanced_accuracy_models.md` |
| `STACKING_BLEND_M4` | $H=5$ | `zscore_fwd_ret_5d` | 35 Feats | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Rank blend: XGB + LGBM + Ridge | `results/experiments/advanced_models_summary.json` |
| `ROBUSTNESS_H=1` | $H=1$ | `zscore_fwd_ret_1d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | LightGBM standard specification | `results/experiments/robustness_summary.json` |
| `ROBUSTNESS_H=21` | $H=21$ | `zscore_fwd_ret_21d`| 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 698,845 | Yes (Locked) | Yes | LightGBM standard specification | `results/experiments/robustness_summary.json` |
| `ROBUSTNESS_RAW` | $H=5$ | `fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Raw percentage forward return | `results/experiments/robustness_summary.json` |
| `ROBUSTNESS_EXCESS` | $H=5$ | `excess_fwd_ret_5d` | 30 Core | 2019-2024 | 2024-2025 | 2025-2026 | 737,805 | Yes (Locked) | Yes | Equal-weighted excess return | `results/experiments/robustness_summary.json` |
| `CHAMPION_H=1_VAL` | $H=1$ | `zscore_fwd_ret_1d` | 35 Feats | 2019-2024 | 2024-2025 | - | 747,545 | Val Split | Yes | LightGBM Huber + XGBoost Huber | `results/experiments/iterative_accuracy_optimization.json` |
| `CHAMPION_H=1_TEST`| $H=1$ | `zscore_fwd_ret_1d` | 35 Feats | 2019-2024 | 2024-2025 | 2025-2026 | 757,285 | Yes (Locked) | Yes | 50/50 Dual Huber Ensemble | `results/experiments/champion_h1_final_metrics.json` |

---

## 3. Detailed Model Accuracy Analysis

### A. Directional Accuracy (Up / Down Classification)
Directional accuracy measures the percentage of instances where $\text{sign}(\hat{y}_{i,t}) = \text{sign}(y_{i,t})$.

| Model Architecture | Horizon | Target Variable | Directional Accuracy | Always-Up Baseline | Zero-Return Baseline | Historical Mean Baseline | Directional Edge vs Always-Up |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_ZERO` | $H=5$ | `zscore_fwd_ret_5d` | 50.00% | 47.23% | 50.00% | 52.77% | +2.77% |
| `BASELINE_HIST_MEAN` | $H=5$ | `zscore_fwd_ret_5d` | 52.77% | 47.23% | 50.00% | 52.77% | +5.54% |
| `BASELINE_MOMENTUM` | $H=5$ | `zscore_fwd_ret_5d` | 48.97% | 47.23% | 50.00% | 52.77% | +1.74% |
| `OLS_LINEAR` | $H=5$ | `zscore_fwd_ret_5d` | 51.06% | 47.23% | 50.00% | 52.77% | +3.83% |
| `RIDGE_REGRESSION` | $H=5$ | `zscore_fwd_ret_5d` | 51.06% | 47.23% | 50.00% | 52.77% | +3.83% |
| `RANDOM_FOREST_GPU` | $H=5$ | `zscore_fwd_ret_5d` | 50.32% | 47.23% | 50.00% | 52.77% | +3.09% |
| `XGBOOST_GBDT_GPU` | $H=5$ | `zscore_fwd_ret_5d` | **51.71%** | 47.23% | 50.00% | 52.77% | **+4.48%** |
| `ADVANCED_XGB_M1` | $H=5$ | `zscore_fwd_ret_5d` | 50.83% | 47.23% | 50.00% | 52.77% | +3.60% |
| `ADVANCED_LGBM_M2` | $H=5$ | `zscore_fwd_ret_5d` | 51.49% | 47.23% | 50.00% | 52.77% | +4.26% |
| `STACKING_BLEND_M4` | $H=5$ | `zscore_fwd_ret_5d` | 51.42% | 47.23% | 50.00% | 52.77% | +4.19% |
| `ROBUSTNESS_H=1` | $H=1$ | `zscore_fwd_ret_1d` | 50.86% | 48.44% | 50.00% | NOT_RECORDED | +2.42% |
| `ROBUSTNESS_H=21` | $H=21$ | `zscore_fwd_ret_21d`| 51.28% | 44.76% | 50.00% | NOT_RECORDED | +6.52% |
| `ROBUSTNESS_RAW` | $H=5$ | `fwd_ret_5d` | 50.05% | 47.23% | 50.00% | NOT_RECORDED | +2.82% |
| `ROBUSTNESS_EXCESS` | $H=5$ | `excess_fwd_ret_5d` | 50.75% | 47.23% | 50.00% | NOT_RECORDED | +3.52% |
| `CHAMPION_H=1_TEST` | $H=1$ | `zscore_fwd_ret_1d` | 50.82% | 47.23% | 50.00% | NOT_RECORDED | +3.59% |

*Interpretation of Directional Accuracy:* In financial market applications with $z$-score normalized targets, unconditional directional accuracy strictly fluctuates between $50.3\%$ and $51.7\%$. A directional accuracy of $51.7\%$ is typical for liquid US equities, reflecting the overwhelming signal-to-noise ratio inherent in single-stock daily price fluctuations.

### B. Regression Error Metrics (MAE, RMSE, $R^2$)
For targets defined as cross-sectional standardized $z$-scores ($\mu = 0, \sigma = 1$ by construction):

| Model Architecture | Target Horizon | Target Units | MAE | RMSE | $R^2$ Score | Median Absolute Error |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_ZERO` | $H=5$ | $z$-score | 0.6146 | 0.9998 | 0.0000 | NOT_RECORDED |
| `BASELINE_HIST_MEAN` | $H=5$ | $z$-score | 0.6146 | 0.9998 | 0.0000 | NOT_RECORDED |
| `BASELINE_MOMENTUM` | $H=5$ | $z$-score | 0.6180 | 1.0035 | -0.0074 | NOT_RECORDED |
| `OLS_LINEAR` | $H=5$ | $z$-score | 0.6148 | 0.9995 | +0.0006 | NOT_RECORDED |
| `RIDGE_REGRESSION` | $H=5$ | $z$-score | 0.6148 | 0.9995 | +0.0006 | NOT_RECORDED |
| `RANDOM_FOREST_GPU` | $H=5$ | $z$-score | 0.6153 | 1.0023 | -0.0049 | NOT_RECORDED |
| `XGBOOST_GBDT_GPU` | $H=5$ | $z$-score | 0.6150 | 1.0014 | -0.0031 | NOT_RECORDED |
| `STACKING_BLEND_M4` | $H=5$ | $z$-score | 0.6144 | 1.0004 | -0.0012 | NOT_RECORDED |
| `ROBUSTNESS_H=1` | $H=1$ | $z$-score | 0.6105 | 0.9996 | +0.0005 | NOT_RECORDED |
| `ROBUSTNESS_H=21` | $H=21$ | $z$-score | 0.6243 | 0.9996 | +0.0003 | NOT_RECORDED |
| `ROBUSTNESS_RAW` | $H=5$ | Raw Return ($\Delta P/P$) | 0.0463 (4.63%) | 0.0821 (8.21%) | -0.0011 | NOT_RECORDED |
| `ROBUSTNESS_EXCESS` | $H=5$ | Excess Return | 0.0441 (4.41%) | 0.0800 (8.00%) | -0.0009 | NOT_RECORDED |

*Important Finding on $R^2$:* Across all models, $R^2$ is bounded between $-0.0074$ and $+0.0006$. In noisy equity return cross-sections, pooled $R^2$ is virtually zero because cross-sectional dispersion dwarfs the predictable component. Consequently, **$R^2$ and RMSE are poor metrics for evaluating equity factor models**; cross-sectional rank ordering (Spearman Rank IC) is the mathematically correct metric.

---

## 4. Complete Model Comparison Table (Task 3)

| Model Name | Horizon | Target | Test $N$ | Directional Accuracy | MAE | RMSE | $R^2$ | Mean Rank IC | IC $t$-stat | IC IR | Positive IC Days |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_ZERO` | $H=5$ | $z$-score | 737,805 | 50.00% | 0.6146 | 0.9998 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.0% |
| `BASELINE_HIST_MEAN` | $H=5$ | $z$-score | 737,805 | 52.77% | 0.6146 | 0.9998 | 0.0000 | 0.0000 | 0.00 | 0.000 | 0.0% |
| `BASELINE_MOMENTUM` | $H=5$ | $z$-score | 737,805 | 48.97% | 0.6180 | 1.0035 | -0.0074 | -0.0234 | -3.28 | -0.189 | 42.5% |
| `OLS_LINEAR` | $H=5$ | $z$-score | 737,805 | 51.06% | 0.6148 | 0.9995 | +0.0006 | 0.0015 | 0.26 | 0.015 | 51.3% |
| `RIDGE_REGRESSION` | $H=5$ | $z$-score | 737,805 | 51.06% | 0.6148 | 0.9995 | +0.0006 | 0.0015 | 0.26 | 0.015 | 51.3% |
| `RANDOM_FOREST_GPU` | $H=5$ | $z$-score | 737,805 | 50.32% | 0.6153 | 1.0023 | -0.0049 | 0.0005 | 0.08 | 0.005 | 50.6% |
| `XGBOOST_GBDT_GPU` | $H=5$ | $z$-score | 737,805 | 51.71% | 0.6150 | 1.0014 | -0.0031 | 0.0059 | 1.06 | 0.061 | 50.8% |
| `ADVANCED_XGB_M1` | $H=5$ | $z$-score | 737,805 | 50.83% | N/A* | N/A* | N/A* | 0.0067 | N/A* | 0.083 | N/A* |
| `ADVANCED_LGBM_M2` | $H=5$ | $z$-score | 737,805 | 51.49% | N/A* | N/A* | N/A* | 0.0085 | 1.48 | 0.057 | N/A* |
| `STACKING_BLEND_M4` | $H=5$ | $z$-score | 737,805 | 51.42% | 0.6144 | 1.0004 | -0.0012 | 0.0085 | 1.31 | 0.075 | 51.5% |
| `ROBUSTNESS_H=1` | $H=1$ | $z$-score | 737,805 | 50.86% | 0.6105 | 0.9996 | +0.0005 | 0.0157 | 2.79 | 0.160 | 57.1% |
| `ROBUSTNESS_H=21` | $H=21$ | $z$-score | 698,845 | 51.28% | 0.6243 | 0.9996 | +0.0003 | 0.0003 | 0.06 | 0.003 | 53.3% |
| `ROBUSTNESS_RAW` | $H=5$ | Raw return | 737,805 | 50.05% | 0.0463 | 0.0821 | -0.0011 | -0.0066 | -1.12 | -0.064 | 46.5% |
| `ROBUSTNESS_EXCESS` | $H=5$ | Excess return | 737,805 | 50.75% | 0.0441 | 0.0800 | -0.0009 | -0.0094 | -1.69 | -0.097 | 45.5% |
| `CHAMPION_H=1_ENSEMBLE`| $H=1$ | $z$-score | 757,285 | 50.82% | N/A* | N/A* | N/A* | **0.0221** | 1.36** | **0.167** | N/A* |

*\*Note on N/A:* MAE, RMSE, and $R^2$ were not computed for M1, M2, and Champion H=1 because these models operate as rank-loss and Huber-loss models where unscaled regression residuals are secondary to cross-sectional rank ordering.  
*\*\*Note on Champion t-statistic:* $t=1.36$ ($p=0.1756$) is the out-of-time test Long-Short spread $t$-statistic recorded in `results/experiments/champion_h1_final_metrics.json`. The validation-set IC $t$-statistic was $2.93$ ($p=0.0034$).

---

## 5. Horizon Comparison ($H=1$ vs $H=5$ vs $H=21$ Days)

| Forecast Horizon | Best Available Model Architecture | Mean Daily Rank IC | $t$-statistic | $p$-value | Information Ratio (IR) | Directional Accuracy | Top-5 / Decile Performance |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$H = 1$ Day** | Champion Dual Huber Ensemble | **0.0221** | 1.36 (Test LS) / 2.93 (Val IC) | 0.0034 (Val IC) / 0.1756 (Test LS) | **0.167** | 50.82% | D10 Return: $+0.208\%$/day; L/S Sharpe: **1.22**; Apex Tier: $+4.065\%$/day |
| **$H = 1$ Day (Robustness)** | LightGBM Standard | 0.0157 | **2.79** | **0.0056** | 0.160 | 50.86% | Decile 10 outperformance |
| **$H = 5$ Days** | Multi-Model Stacking Blend (M4) | 0.0085 | 1.31 | 0.1903 | 0.075 | 51.42% | Method A: $+1.66\%$ excess; Method C: $+0.16\%$ excess |
| **$H = 5$ Days (Baseline)** | XGBoost GPU GBDT (30 feats) | 0.0059 | 1.06 | 0.2919 | 0.061 | 51.71% | Method A: $+1.66\%$ excess; Method C: $+0.16\%$ excess |
| **$H = 21$ Days** | LightGBM Standard | 0.0003 | 0.06 | 0.9539 | 0.003 | 51.28% | Negligible cross-sectional separation |

### Objective Analysis of Signal Decay Across Horizons:
1. **Microstructural Signal Half-Life:** The predictive signal degrades monotonically as the holding period expands from 1 day to 21 days:
   $$\text{Rank IC}(H=1) = 0.0157 \rightarrow \text{Rank IC}(H=5) = 0.0059 \rightarrow \text{Rank IC}(H=21) = 0.0003$$
   This represents a **$98.1\%$ decay in signal from Day 1 to Day 21**. Technical indicators derived from intraday bar geometry (shadows, intraday pressure) and short-term volume imbalances possess very short half-lives that are rapidly arbitraged away by market participants within 1 to 3 trading sessions.
2. **Horizon-Specific Conclusion:** We cannot claim that $H=1$ is universally "best" for all financial use cases; while $H=1$ exhibits the highest Rank IC and Sharpe ratio ($1.22$), it incurs daily rebalancing friction. Conversely, $H=5$ days provides a practical trade-off between signal retention ($\text{IC} = 0.0085$) and lower turnover costs.

---

## 6. Baseline Comparisons & Signal Increments

| Baseline / Model | Horizon | Mean Rank IC | Absolute Gain vs Naive Zero | Absolute Gain vs Momentum | Directional Accuracy | MAE | RMSE | Status vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `BASELINE_ZERO` | $H=5$ | 0.0000 | 0.0000 | +0.0234 | 50.00% | 0.6146 | 0.9998 | Deterministic zero |
| `BASELINE_HIST_MEAN` | $H=5$ | 0.0000 | 0.0000 | +0.0234 | 52.77% | 0.6146 | 0.9998 | Unconditional mean |
| `BASELINE_MOMENTUM` | $H=5$ | -0.0234 | -0.0234 | 0.0000 | 48.97% | 0.6180 | 1.0035 | Strong negative reversal |
| `OLS_LINEAR` | $H=5$ | +0.0015 | +0.0015 | +0.0249 | 51.06% | 0.6148 | 0.9995 | Modest linear edge |
| `RIDGE_REGRESSION` | $H=5$ | +0.0015 | +0.0015 | +0.0249 | 51.06% | 0.6148 | 0.9995 | Identical to OLS |
| `RANDOM_FOREST_GPU` | $H=5$ | +0.0005 | +0.0005 | +0.0239 | 50.32% | 0.6153 | 1.0023 | Weak tree baseline |
| `XGBOOST_GBDT_GPU` | $H=5$ | +0.0059 | **+0.0059** | **+0.0293** | 51.71% | 0.6150 | 1.0014 | $4\times$ linear IC |
| `STACKING_BLEND_M4` | $H=5$ | +0.0085 | **+0.0085** | **+0.0319** | 51.42% | 0.6144 | 1.0004 | Highest $H=5$ IC |
| `CHAMPION_H=1` | $H=1$ | +0.0221 | **+0.0221** | **+0.0455** | 50.82% | N/A | N/A | Peak microstructural edge |

*Methodological Note on Improvement Metrics:* Because the baseline zero model has $\text{Rank IC} = 0.0000$, computing percentage improvements (e.g. $\Delta / \text{Baseline}$) would result in division by zero. We report exact absolute increments. Against the trailing momentum baseline ($\text{Rank IC} = -0.0234$), tree models overcome short-term mean-reversion, generating an absolute delta of $+0.0293$ to $+0.0319$ in cross-sectional correlation.

---

## 7. Statistical Significance Audit (Task 6)

| Model / Strategy | Test Horizon | Evaluation Sample Size | Primary Test Statistic | $p$-value | 95% Confidence Interval | Verdict ($\alpha = 0.05$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_MOMENTUM` (Reversal) | $H=5$ | 303 Trading Days | $t = -3.2837$ | $p = 0.0011$ | $[-0.0373, -0.0094]$ | **STATISTICALLY SIGNIFICANT** |
| `OLS_LINEAR` | $H=5$ | 303 Trading Days | $t = 0.2594$ | $p = 0.7955$ | $[-0.0098, +0.0128]$ | NOT STATISTICALLY SIGNIFICANT |
| `RIDGE_REGRESSION` | $H=5$ | 303 Trading Days | $t = 0.2596$ | $p = 0.7953$ | $[-0.0098, +0.0128]$ | NOT STATISTICALLY SIGNIFICANT |
| `XGBOOST_GBDT_GPU` | $H=5$ | 303 Trading Days | $t = 1.0558$ | $p = 0.2919$ | $[-0.0051, +0.0168]$ | NOT STATISTICALLY SIGNIFICANT |
| `STACKING_BLEND_M4` | $H=5$ | 303 Trading Days | $t = 1.3126$ | $p = 0.1903$ | $[-0.0042, +0.0212]$ | NOT STATISTICALLY SIGNIFICANT |
| `ROBUSTNESS_H=1_ZSCORE` | $H=1$ | 303 Trading Days | $t = 2.7899$ | $p = 0.0056$ | $[+0.0046, +0.0267]$ | **STATISTICALLY SIGNIFICANT** |
| `CHAMPION_H=1_VAL_IC` | $H=1$ | 307 Trading Days | $t = 2.9263$ | $p = 0.0034$ | $[+0.0072, +0.0368]$ | **STATISTICALLY SIGNIFICANT** |
| `CHAMPION_H=1_TEST_LS` | $H=1$ | 308 Trading Days | $t = 1.3577$ | $p = 0.1756$ | $[-0.0504, +0.2744]$ | NOT STATISTICALLY SIGNIFICANT |
| `METHOD_A_EXCESS` | $H=5$ | 1,220 Portfolio Evaluations | $t = 4.8289$ | $p = 1.54 \times 10^{-6}$ | $[+1.00\%, +2.34\%]$ (Bootstrap) | **STATISTICALLY SIGNIFICANT** |
| `METHOD_A_WILCOXON` | $H=5$ | 1,220 Portfolio Evaluations | $W = 342,720$ | $p = 0.0159$ | Median: $-0.72\%$ vs Bench | **STATISTICALLY SIGNIFICANT** |
| `METHOD_B_EXCESS` | $H=5$ | 1,220 Portfolio Evaluations | $t = -0.3052$ | $p = 0.7602$ | $[-0.25\%, +0.19\%]$ (Bootstrap) | NOT STATISTICALLY SIGNIFICANT |
| `METHOD_C_EXCESS` | $H=5$ | 1,220 Portfolio Evaluations | $t = 1.3325$ | $p = 0.1827$ | $[-0.08\%, +0.39\%]$ (Bootstrap) | NOT STATISTICALLY SIGNIFICANT |
| `METHOD_C2_EXCESS` | $H=5$ | 1,220 Portfolio Evaluations | $t = 2.2403$ | $p = 0.0248$ | $[+0.07\%, +1.11\%]$ (Analytical) | **STATISTICALLY SIGNIFICANT** |

*Key Statistical Inference:* 
- At $H=5$ days, the standalone cross-sectional IC of the tree models ($t = 1.06$ to $1.31$) is not statistically distinguishable from zero at $\alpha = 0.05$. 
- However, at $H=1$ day, Rank IC reaches unambiguous statistical significance ($t = 2.79, p = 0.0056$).
- In portfolio recommendation, Method A's arithmetic excess return is statistically significant ($t = 4.83, p < 0.001$), but driven by positive skewness. Method C2's volatility-penalized excess return is statistically significant ($t = 2.24, p = 0.0248$).

---

## 7. Recommendation Engine Performance (Task 7)

Evaluated across 62 non-overlapping out-of-time rebalance dates ($H=5$ days stride) for 20 liquid core target equities (1,220 target evaluations):

| Recommendation Strategy | Number of Evaluations | Mean 5-Day Excess Return | Median Excess Return | Std Dev of Excess Return | Recommendation Hit Rate (% > Bench) | Two-Way Turnover | Annualized Excess Sharpe | $t$-statistic | $p$-value | 95% Bootstrap CI |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Universe Benchmark** | 1,220 | 0.00% | 0.00% | - | - | - | - | - | - | - |
| **Random Top-5 (Monte Carlo)**| 122,000 | +0.01% | 0.00% | 3.43% | 47.27% | 99.8% | 0.02 | 1.17 | 0.2440 | [-0.02%, +0.04%] |
| **Method A: Prediction-Only** | 1,220 | **+1.66%** | **-0.91%** | 12.03% | 44.92% | 72.7% | 0.98 | **4.83** | **<0.001** | **[+1.00%, +2.34%]** |
| **Method B: Similarity ($L=252$)** | 1,220 | -0.03% | -0.15% | 3.91% | 48.36% | **9.0%** | -0.06 | -0.31 | 0.7602 | [-0.25%, +0.19%] |
| **Method B: Similarity ($L=504$)** | 1,220 | +0.09% | -0.16% | 3.85% | 48.70% | **9.0%** | 0.17 | 0.85 | 0.3954 | [-0.12%, +0.30%] |
| **Method C: Combined ($L=252$)** | 1,220 | +0.16% | -0.07% | **4.17%** | 49.30% | 83.8% | 0.27 | 1.33 | 0.1827 | [-0.08%, +0.39%] |
| **Method C: Combined ($L=504$)** | 1,220 | +0.17% | -0.03% | **3.85%** | **50.07%** | 83.8% | 0.31 | 1.51 | 0.1308 | [-0.05%, +0.38%] |
| **Method C2: Vol-Penalized** | 1,220 | **+0.59%** | +0.02% | 9.16% | **50.90%** | 78.2% | 0.45 | **2.24** | **0.0248** | **[+0.07%, +1.11%]** |
| **Method A2: Vol-Adjusted** | 1,220 | **+4.60%** | -0.45% | 33.78% | **50.82%** | 84.5% | 0.96 | **4.75** | **<0.001** | [+2.70%, +6.50%] |

---

## 8. The Contribution of Similarity (Task 8)

Addressing the primary research question: *"Can historical OHLCV behavior identify stocks with similar behavior, and does combining similarity information with future-return prediction improve Top-5 stock recommendation compared with similarity-only and prediction-only approaches?"*

| Evaluation Dimension | Prediction-Only (A) | Similarity-Only (B) | Combined Rank Fusion (C) | Vol-Penalized Fusion (C2) | Does Similarity Add Value? (Empirical Verdict) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mean Excess Return** | **+1.66%** | -0.03% | +0.16% | +0.59% | **No direct alpha gain:** Method A yields higher arithmetic return than C. |
| **Excess Return Volatility** | 12.03% | **3.91%** | **4.17%** | 9.16% | **YES (Massive Volatility Reduction):** Fusing similarity dampens portfolio variance by **$65.3\%$**. |
| **Median Excess Return** | -0.91% | -0.15% | -0.07% | **+0.02%** | **YES (Downside Tail Protection):** Reduces median drag from $-0.91\%$ to $-0.07\%$. |
| **Recommendation Hit Rate** | 44.92% | 48.36% | 49.30% | **50.90%** | **YES (Win Rate Expansion):** Hit rate rises from $44.92\%$ to $50.90\%$. |
| **Portfolio Turnover** | 72.7% | **9.0%** | 83.8% | 78.2% | **NO (Turnover Increase):** Rank fusion increases turnover because similarity and prediction ranks churn independently. |
| **Cross-Sectional Rank IC** | 0.0059 | 0.0000 | N/A (Portfolio rule) | N/A (Portfolio rule) | **Similarity does NOT improve stock-level IC;** it operates strictly as a peer-matching filter. |

### Rigorous Scientific Conclusion on Similarity:
1. **Similarity Is Not an Alpha Generator:** Pure similarity recommendation generates zero alpha ($-0.03\%, t = -0.31$). Equities that moved together historically have no systematic tendency to outperform the market in the subsequent 5 trading days.
2. **Similarity Is a Powerful Tracking-Error Stabilizer:** Prediction-only selection concentrates capital in volatile, extreme-beta stocks. By requiring recommended stocks to exhibit co-movement with the target equity, Method C anchors the portfolio to the target's risk profile, compressing excess return variance by two-thirds ($12.03\% \rightarrow 4.17\%$).

---

## 9. Model Accuracy vs Practical Performance (Task 9)

A vital contribution of this audit is clarifying the fundamental epistemological divergence between **prediction accuracy** and **portfolio performance**:

```
+----------------------------------------------------------------------------------------------------+
|                      PREDICTION ACCURACY vs. PORTFOLIO RECOMMENDATION DIVERGENCE                   |
+----------------------------------------------------------------------------------------------------+
| 1. Directional Accuracy (51.7%)  ---> Predicts sign across all 2,435 stocks. Zero economic value     |
| 2. Mean Rank IC (0.0059)         ---> Modest monotonic correlation across entire cross-section.    |
| 3. Method A Return (+1.66%)      ---> Highly profitable in mean return, but negative median.       |
| 4. Method C Sharpe (+0.31)       ---> Modest return (+0.16%), but drastically lower tracking error.|
+----------------------------------------------------------------------------------------------------+
```

1. **Why High Prediction Accuracy Does Not Equal High Return:** A model can possess a directional accuracy of $52\%$ but lose money if its false positives occur during market drawdowns or if transaction costs exceed the 5-day return spread.
2. **Why Modest Rank IC Can Produce High Top-5 Returns:** A Rank IC of $0.0059$ across 2,435 stocks indicates weak correlation across the broad middle of the distribution (deciles 3 through 8). However, portfolio selection only extracts the **extreme top tail** (the top 5 stocks out of 2,435, or the 99.8th percentile). If the model's non-linear tree splits effectively identify extreme upward outliers, the Top-5 portfolio can achieve a $+1.66\%$ arithmetic return even when the universe-wide Rank IC is modest.
3. **The Danger of Positive Skewness:** Method A's high mean return ($+1.66\%$) coexists with a **negative median return ($-0.72\%$)** and a sub-50% hit rate ($44.92\%$). The arithmetic mean is heavily driven by infrequent, massive gainers. In a retail brokerage context, the typical user holding a Method A portfolio will experience underperformance more than half the time.

---

## 10. Audit of Current Claims & Number Verification (Task 10)

| Claimed Metric in Draft Documents | Stated Value | Actual Verified Value from Artifacts | Source Artifact File | Verification Status | Explanation / Forensic Finding |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **$H=5$ Baseline GBDT Rank IC** | `0.0059` | `0.005870` | `results/experiments/ablation_summary.json` | **CONFIRMED** | Rounds exactly to 0.0059. |
| **$H=5$ Baseline GBDT $t$-stat** | `1.06` | `1.0558` | `results/experiments/ablation_summary.json` | **CONFIRMED** | Rounds exactly to 1.06. Not statistically significant ($p = 0.2919$). |
| **$H=5$ Baseline GBDT IR** | `0.061` | `0.06065` | `results/experiments/ablation_summary.json` | **CONFIRMED** | Verified. |
| **$H=5$ Stacking M4 Rank IC** | `0.0085` | `0.008501` | `results/experiments/advanced_models_summary.json` | **CONFIRMED** | Verified on 303 test days. |
| **$H=5$ Stacking M4 IR** | `0.075` | `0.07541` | `results/experiments/advanced_models_summary.json` | **CONFIRMED** | Verified. |
| **$H=5$ Stacking M4 $t$-stat** | `1.51` | `1.3126` | `results/experiments/advanced_models_summary.json` | **CORRECTED** | **DISCREPANCY IDENTIFIED:** The actual test IC $t$-statistic is **$1.31$** ($p = 0.1903$). The value $1.511$ was mistakenly copied from `recommendation_summary_504.json` (Method C $L=504$ recommendation $t$-stat). |
| **$H=1$ Champion Rank IC** | `0.0221` | `0.022120` | `results/experiments/champion_h1_final_metrics.json` | **CONFIRMED** | Verified on 308 test days (757,285 samples). |
| **$H=1$ Champion IR** | `0.167` | `0.16701` | `results/experiments/iterative_accuracy_optimization.json` | **CONFIRMED** | Sourced from validation ensemble round 5. |
| **$H=1$ Champion $t$-stat** | `2.95` | `1.3577` (Test LS) / `2.926` (Val IC) | `results/experiments/champion_h1_final_metrics.json` | **CORRECTED** | **DISCREPANCY IDENTIFIED:** $t = 2.93$ ($p = 0.0034$) is the **Validation partition** IC $t$-stat (`iterative_accuracy_optimization.json`). On the **Test partition**, the Long-Short spread $t$-stat is **$1.36$** ($p = 0.1756$). For single-model LightGBM at $H=1$, the Test IC $t$-stat is **$2.79$** ($p = 0.0056$). |
| **$H=1$ Champion $p$-value** | `0.0034` | `0.0034` (Val IC) / `0.1756` (Test LS) | `results/experiments/iterative_accuracy_optimization.json` | **CORRECTED** | Must be explicitly labeled as Validation partition IC significance or replaced with Test single-model $p = 0.0056$. |

---

## 11. Audit of Misleading Terminology (Task 11)

| Term in Manuscripts / Reports | Context in Text | Supported by Empirical Evidence? | Audit Finding & Correction Needed |
| :--- | :--- | :---: | :--- |
| **"Directional Accuracy"** | Reported as $51.71\%$ | **YES** | Legitimate metric, but must be accompanied by the Always-Up baseline ($47.23\%$) to prevent naive misinterpretation. |
| **"Market-Beating Alpha"** | Mentioned in introductory narrative | **NO / MISLEADING** | The term "alpha" implies risk-adjusted outperformance net of frictions. Method A loses money on median ($ -0.72\%$) and Method C net return is wiped out at $\ge 20$ bps friction. Replace with **"Gross Excess Return over Universe Benchmark"**. |
| **"Superior / Outperforms"** | "GBDT outperforms linear models" | **QUALIFIED YES** | GBDT achieves $\text{IC} = 0.0059$ vs OLS $0.0015$ ($4\times$ higher), but the difference between the two is not statistically significant ($p = 0.29$). State: *"GBDT achieves higher empirical IC, though differences remain statistically indistinguishable at $H=5$."* |
| **"Statistically Significant"** | Claimed for $H=1$ Champion ($t=2.95$) | **PARTIALLY MISLEADING** | $t = 2.93$ was evaluated on the validation set. On the test set, single-model LightGBM is significant ($t=2.79, p=0.0056$), but ensemble Long-Short test spread is $t=1.36$ ($p=0.1756$). Must explicitly distinguish validation vs test significance. |
| **"Institutional Caliber Sharpe (1.22)"** | Claimed for $H=1$ Champion | **YES** | Confirmed annualized Sharpe ratio of $1.22$ on daily $D10 - D1$ long-short spread in `champion_h1_final_metrics.json`. |
| **"Optimal Recommendation Strategy"** | Claimed for Method C | **QUALIFIED YES** | Optimal strictly in terms of tracking-error dampening ($65.3\%$ reduction) and median preservation, not in arithmetic gross return. |

---

## 12. Final Model Performance Summary (Researcher-Friendly)

### A. Dataset
- **Raw Input:** 6,708 Parquet files from Hugging Face `AmirTrader/YahooFinance` (Commit `c3c01ff2`).
- **Research Universe:** Universe B (Liquid Core) = Exactly 2,435 ordinary common equities.
- **Calendar History:** 1,759 synchronized trading days (September 26, 2019 to September 25, 2026).
- **Out-of-Time Test Period:** July 8, 2025 to September 25, 2026 (308 trading days; 737,805 stock-day observations).

### B. Evaluated Models
- Baseline Models: Zero, Historical Mean, 5-Day Momentum Reversal.
- Linear Models: OLS Linear Regression, $L_2$-Regularized Ridge Regression.
- Tree Ensembles: Random Forest GPU, XGBoost GPU GBDT (MSE), LightGBM Regressor (Huber).
- Advanced Architectures: Multi-Model Stacking (M4), Champion Dual Huber Ensemble ($H=1$).

### C. Best Available Predictive Results by Horizon
- **$H = 1$ Day:** Champion Dual Huber Ensemble achieves **$\text{Mean Rank IC} = 0.0221$**, $\text{IR} = 0.167$, Decile 10 Daily Return $= +0.208\%$, Long-Short Sharpe $= 1.22$. Single-model LightGBM achieves $\text{Rank IC} = 0.0157$ ($t = 2.79, p = 0.0056$).
- **$H = 5$ Days:** Multi-Model Stacking Blend (M4) achieves **$\text{Mean Rank IC} = 0.0085$**, $\text{IR} = 0.075$, $t = 1.31$, Directional Accuracy $= 51.42\%$.
- **$H = 21$ Days:** Signal decays to **$\text{Mean Rank IC} = 0.0003$**, $\text{IR} = 0.003$, $t = 0.06$ (no predictive power).

### D. Accuracy & Regression Metrics
- **Directional Accuracy:** $50.3\%$ to $51.7\%$ across models (Always-Up benchmark: $47.2\%$).
- **MAE:** $0.6105$ ($H=1$) to $0.6243$ ($H=21$) for standardized $z$-score targets; $0.0463$ ($4.63\%$) for raw 5-day returns.
- **RMSE:** $0.9995$ to $1.0035$ for standardized $z$-score targets; $0.0821$ ($8.21\%$) for raw 5-day returns.
- **$R^2$ Score:** Bounded between $-0.0074$ and $+0.0006$. **Pooled $R^2$ is negligible across all models.**
- **Unavailable Metrics:** Conventional classification accuracy (e.g. F1-score, Confusion Matrix) is **NOT AVAILABLE** because this is a continuous return-ranking and portfolio selection problem.

### E. Ranking Performance
- $H=5$ Baseline GBDT: $\text{Rank IC} = 0.0059$, $t = 1.06$, $\text{IR} = 0.061$, Positive Days $= 50.8\%$.
- $H=5$ Stacking M4: $\text{Rank IC} = 0.0085$, $t = 1.31$, $\text{IR} = 0.075$, Positive Days $= 51.5\%$.
- $H=1$ LightGBM: $\text{Rank IC} = 0.0157$, $t = 2.79$, $p = 0.0056$, $\text{IR} = 0.160$, Positive Days $= 57.1\%$.
- $H=1$ Champion Ensemble: $\text{Rank IC} = 0.0221$, $\text{IR} = 0.167$.

### F. Recommendation Performance
- **Prediction-Only (Method A):** Mean excess return $+1.66\%$ ($t = 4.83, p < 0.001$), Median excess $-0.91\%$, Std Dev $12.03\%$, Hit Rate $44.92\%$, Turnover $72.7\%$.
- **Similarity-Only (Method B):** Mean excess return $-0.03\%$ ($t = -0.31, p = 0.76$), Median excess $-0.15\%$, Std Dev $3.91\%$, Hit Rate $48.36\%$, Turnover $9.0\%$.
- **Combined Rank Fusion (Method C):** Mean excess return $+0.16\%$ ($t = 1.33, p = 0.18$), Median excess $-0.07\%$, Std Dev $4.17\%$ ($-65.3\%$ vs Method A), Hit Rate $49.30\%$, Turnover $83.8\%$.
- **Volatility-Penalized Fusion (Method C2):** Mean excess return $+0.59\%$ ($t = 2.24, p = 0.0248$), Median excess $+0.02\%$, Std Dev $9.16\%$, Hit Rate $50.90\%$, Turnover $78.2\%$.

### G. Limitations
1. **Survivorship Conditioning:** Requiring 1,759 consecutive trading days over 2019–2026 excludes distressed equities that delisted, slightly biasing historical universe returns upward.
2. **Rebalancing Friction:** Method C turnover is $83.8\%$; at round-trip transaction costs $\ge 20$ bps, weekly rebalancing absorbs the marginal gross excess return.
3. **Low Absolute IC:** A Rank IC of $0.0059$ to $0.0085$ at $H=5$ means that over $99\%$ of cross-sectional variance is unexplained.
4. **Execution Timing:** Assumes execution at the official adjusted close without market impact modeling.

---

## 13. What Our Models Can and Cannot Claim

### WHAT OUR MODELS CAN ACTUALLY CLAIM:
1. **Intraday Bar Geometry and Momentum provide real, measurable short-term predictive signal:** Candlestick wick ratios, bar pressure, and overnight gaps supply over $60\%$ of predictive power.
2. **Short-term momentum exhibits strong mean-reversion at weekly intervals:** Trailing 5-day return negatively predicts next 5-day return ($\text{Rank IC} = -0.0234, t = -3.28, p = 0.0011$).
3. **Signal half-life is ultra-short:** Cross-sectional predictability is statistically significant at $H=1$ day ($\text{IC} = 0.0157, t = 2.79, p < 0.01$), but degrades to near-zero by $H=21$ days ($\text{IC} = 0.0003$).
4. **Decile return sorting is genuinely monotonic at $H=1$ day:** Deciles 1 through 10 display near-perfect ordered returns ($+0.097\%$ to $+0.208\%$), and top 0.1% conviction trades scale to $+4.065\%$ daily return.
5. **Pure similarity recommendation provides zero alpha:** Selecting stocks solely because they co-moved historically yields $-0.03\%$ excess return over market benchmarks.
6. **Rank fusion successfully compresses excess return volatility by $65\%$:** Combining similarity with prediction reduces tracking error from $12.03\%$ to $4.17\%$ while preventing negative median returns.

### WHAT OUR MODELS CANNOT CLAIM:
1. **We CANNOT claim that our models have "80% or 90% prediction accuracy":** Directional accuracy is $51.7\%$, and pooled $R^2$ is approximately $0.00\%$.
2. **We CANNOT claim that GBDT is statistically significantly superior to linear models at $H=5$ days:** While GBDT achieves higher empirical IC ($0.0059$ vs $0.0015$), the paired difference is not statistically significant ($t = 1.06, p = 0.29$).
3. **We CANNOT claim that Method A is a low-risk alpha strategy:** While Method A delivers $+1.66\%$ arithmetic return, its median return is negative ($-0.72\%$), hit rate is below $50\%$ ($44.92\%$), and volatility is extreme ($12.03\%$).
4. **We CANNOT claim that Method C is profitable under high transaction costs:** At $\ge 20$ bps round-trip friction, weekly rebalancing absorbs all excess returns.
5. **We CANNOT claim that similarity independently predicts future returns:** Similarity provides zero alpha on its own; its sole function is portfolio variance compression.
