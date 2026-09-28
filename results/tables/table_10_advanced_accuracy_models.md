### Table 10: Performance Gains from Advanced Architectures, Feature Interactions, and Volatility Penalization

#### Part A: Out-of-Time Forecasting Performance on Test Partition (308 Trading Days)
| Model Architecture & Specification | Features Used | Objective Loss Function | Out-of-Time Mean Rank IC | Information Ratio (IR) | Directional Accuracy | Gain vs Baseline IC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline XGBoost GPU GBDT** | 30 Original | MSE Loss | 0.0059 | 0.061 | 51.71% | - |
| **Advanced XGBoost GPU GBDT (M1)** | 35 (Incl. Interactions) | MSE Loss | 0.0067 | 0.083 | 50.83% | +13.5% |
| **Advanced LightGBM Regressor (M2)** | 35 (Incl. Interactions) | Huber Loss (Tail Robust) | **0.0085** | 0.057 | 51.49% | **+44.1%** |
| **Multi-Model Stacking Blend (M4)** | 35 (XGB + LGBM + Ridge) | Optimal Blended Ranks | **0.0085** | **0.075** | 51.42% | **+44.1%** |

#### Part B: Top-5 Recommendation Performance Gains (Out-of-Time Test Set)
| Recommendation Strategy | Mean 5-Day Excess Return | Standard Deviation of Excess Return | $t$-statistic ($p$-value) | Recommendation Hit Rate (% > Bench) | Performance Advancement |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original Method A (Prediction-Only)** | +1.66% | 12.03% | 4.83 (<0.001) | 44.92% | Baseline unconstrained prediction |
| **Advanced Method A2 (Volatility-Adjusted)** | **+4.60%** | 33.78% | 4.75 (<0.001) | **50.82%** | Hit rate crosses above 50% threshold |
| **Original Method C (Combined Fusion)** | +0.16% | 4.17% | 1.33 (0.184) | 49.30% | Baseline rank fusion |
| **Advanced Method C2 (Volatility-Penalized Fusion)** | **+0.59%** | 9.16% | **2.24 (0.025)** | **50.90%** | **3.7x higher excess return; statistically significant ($p<0.05$)** |

*Note: Advanced models incorporate 5 non-linear interaction features (wick asymmetry, momentum acceleration, volume-turnover pressure) and volatility-adjusted rank scoring. All improvements verified on out-of-time test partition (July 2025 – September 2026).*
