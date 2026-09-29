### Table 11: Champion H=1 Day Model Performance, Decile Monotonicity, and Conviction Scaling

#### Panel A: Out-of-Time Test Performance vs Baseline (Evaluation Period: 2025-07-08 to 2026-09-25; 308 Trading Days)
| Metric | Baseline GBDT (H=5d) | Champion Ensemble (H=1d) | Improvement / Relative Delta |
| :--- | :--- | :--- | :--- |
| **Mean Daily Rank IC** | 0.0059 | **0.0221** | **+274.6%** |
| **IC $t$-statistic** | 1.07 ($p=0.285$) | **2.95 ($p=0.0034$)** | **Statistically Significant ($p<0.01$)** |
| **Information Ratio (IR)** | 0.061 | **0.167** | **+173.8%** |
| **Decile 10 Daily Mean Return** | +0.152% | **+0.208%** | **+36.8%** |
| **D10 - D1 Long-Short Daily Spread** | +0.020% (5d normalized) | **+0.112% (daily)** | **Consistent Daily Edge** |
| **Annualized Long-Short Return** | +4.86% | **+28.12%** | **+5.79x Increase** |
| **Long-Short Annualized Sharpe Ratio** | 0.42 | **1.22** | **Institutional Quality ($\ge 1.0$)** |

#### Panel B: Decile Monotonicity Verification on Out-of-Time Test Partition
| Decile Portfolio | Sample Size | Daily Mean Return | Annualized Return | Outperformance Hit Rate | Win Rate (Days > 0) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Decile 1 (Lowest Predicted)** | 75,884 | +0.097% | +24.4% | 46.61% | 46.52% |
| **Decile 2** | 75,884 | +0.106% | +26.7% | 48.01% | 48.74% |
| **Decile 3** | 75,884 | +0.114% | +28.7% | 48.19% | 49.12% |
| **Decile 4** | 75,884 | +0.118% | +29.7% | 47.98% | 49.08% |
| **Decile 5** | 75,884 | +0.125% | +31.5% | 48.44% | 49.80% |
| **Decile 6** | 75,884 | +0.131% | +33.0% | 48.10% | 49.82% |
| **Decile 7** | 75,884 | +0.138% | +34.8% | 48.70% | 50.66% |
| **Decile 8** | 75,884 | +0.149% | +37.5% | 48.58% | 50.83% |
| **Decile 9** | 75,884 | +0.168% | +42.3% | 49.40% | 51.07% |
| **Decile 10 (Highest Predicted)** | 75,884 | **+0.208%** | **+52.5%** | **50.21%** | **50.76%** |

#### Panel C: Conviction Scaling Across Prediction Tiers
| Conviction Tier | Sample Count | Daily Mean Return | Annualized Return | Outperformance Hit Rate | Win Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decile 10 (Top 10%)** | 75,884 | +0.208% | +52.5% | 50.21% | 50.76% |
| **Vigintile 20 (Top 5%)** | 37,942 | +0.316% | +79.6% | 51.01% | 50.75% |
| **Centile 100 (Top 1%)** | 7,775 | +0.822% | +207.0% | 51.33% | 48.60% |
| **Apex Tier (Top 0.1%)** | 933 | **+4.065%** | **+1,024.3%** | **53.05%** | 48.87% |

*Note: Models evaluated out-of-time using LightGBM Huber + XGBoost GPU Huber ensemble with strict 5-day purged temporal boundaries.*
