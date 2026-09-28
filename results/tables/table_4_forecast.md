### Table 4: Out-of-Time Forecasting Performance on Test Partition

| Model Architecture | Mean Rank IC | IC IR | $t$-statistic | $p$-value | MAE | RMSE | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BASELINE_ZERO** | 0.0000 | 0.000 | nan | nan | 0.6146 | 0.9998 | 50.00% |
| **BASELINE_HIST_MEAN** | 0.0000 | 0.000 | nan | nan | 0.6146 | 0.9998 | 52.77% |
| **BASELINE_MOMENTUM** | -0.0234 | -0.189 | -3.28 | 1.1448e-03 | 0.6180 | 1.0035 | 48.97% |
| **OLS_LINEAR** | 0.0015 | 0.015 | 0.26 | 7.9552e-01 | 0.6148 | 0.9995 | 51.06% |
| **RIDGE_REGRESSION** | 0.0015 | 0.015 | 0.26 | 7.9533e-01 | 0.6148 | 0.9995 | 51.06% |
| **RANDOM_FOREST_GPU** | 0.0005 | 0.005 | 0.08 | 9.3303e-01 | 0.6153 | 1.0023 | 50.32% |
| **XGBOOST_GBDT_GPU** | 0.0059 | 0.061 | 1.06 | 2.9190e-01 | 0.6150 | 1.0014 | 51.71% |

*Note: Directional accuracy represents percentage of days correctly predicting relative cross-sectional sign.*
