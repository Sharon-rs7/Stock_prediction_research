### Table 7: Out-of-Time Top-5 Recommendation Performance vs Benchmarks and Baselines

| Recommendation Strategy / Baseline | Mean 5-Day Return | Median Return | Mean Excess Return vs Benchmark | Std Dev of Excess Return | $t$-stat ($p$-value) | Paired Wilcoxon $p$-value | 95% Bootstrap CI of Excess Return | Hit Rate (% > Bench) | Two-Way Turnover | Net Excess Return (10 bps) | Net Excess Return (20 bps) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Universe Benchmark** | 0.42% | 0.31% | 0.00% | - | - | - | - | - | - | - | - |
| **Random Top-5 (Monte Carlo $B=100$)** | 0.43% | 0.31% | +0.01% | 3.43% | 0.08 (0.936) | 0.491 | [-0.02%, +0.04%] | 47.27% | 99.8% | -0.09% | -0.19% |
| **Method A: Prediction-Only** | **2.09%** | -0.72% | **+1.66%** | 12.03% | 4.83 (<0.001) | **0.0159** | [+1.00%, +2.34%] | 44.92% | 72.7% | **+1.59%** | **+1.52%** |
| **Method B: Similarity-Only** | 0.39% | 0.34% | -0.03% | 3.91% | -0.31 (0.757) | 0.2266 | [-0.25%, +0.19%] | 48.36% | **9.0%** | -0.04% | -0.05% |
| **Method C: Combined Rank Fusion** | **0.58%** | **0.30%** | **+0.16%** | **4.17%** | 1.33 (0.184) | 0.9181 | [-0.08%, +0.39%] | **49.30%** | 83.8% | **+0.08%** | -0.01% |

*Note: Evaluated across 62 out-of-time rebalance dates ($H=5$ days stride) for 20 liquid core target equities. Random Top-5 simulated over 122,000 portfolio draws. Net excess return adjusts for two-way portfolio rebalancing turnover at 10 bps and 20 bps round-trip transaction costs.*
