### Table 8: Robustness Evaluation Across Forecast Horizons ($H \in \{1, 5, 21\}$) and Target Formulations

| Specification ($H$, Target Formulation) | Mean Rank IC | IC IR | $t$-statistic | MAE | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H=1_ZSCORE** | 0.0157 | 0.160 | 2.79 | 0.6105 | 50.86% |
| **H=5_PRIMARY_ZSCORE** | 0.0015 | 0.015 | 0.26 | 0.6148 | 51.06% |
| **H=21_ZSCORE** | 0.0003 | 0.003 | 0.06 | 0.6243 | 51.28% |
| **H=5_RAW_RETURN** | -0.0066 | -0.064 | -1.12 | 0.0463 | 50.05% |
| **H=5_EXCESS_RETURN** | -0.0094 | -0.097 | -1.69 | 0.0441 | 50.75% |

*Note: Evaluated using pre-specified LightGBM model on identical out-of-time test dates.*
