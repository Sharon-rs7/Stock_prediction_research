### Table 5: Daily Rank Information Coefficient (IC) Distributional Statistics

| Model ID | Mean IC | Median IC | Std Dev | Std Error | IC IR | % Positive Days | $t$-statistic ($p$-value) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BASELINE_ZERO** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0% | nan (nan) |
| **BASELINE_HIST_MEAN** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.0% | nan (nan) |
| **BASELINE_MOMENTUM** | -0.0234 | -0.0082 | 0.1238 | 0.0071 | -0.189 | 45.2% | -3.28 (1.145e-03) |
| **OLS_LINEAR** | 0.0015 | 0.0046 | 0.1000 | 0.0057 | 0.015 | 51.8% | 0.26 (7.955e-01) |
| **RIDGE_REGRESSION** | 0.0015 | 0.0046 | 0.1000 | 0.0057 | 0.015 | 51.8% | 0.26 (7.953e-01) |
| **RANDOM_FOREST_GPU** | 0.0005 | -0.0071 | 0.0991 | 0.0057 | 0.005 | 48.5% | 0.08 (9.330e-01) |
| **XGBOOST_GBDT_GPU** | 0.0059 | 0.0023 | 0.0968 | 0.0056 | 0.061 | 50.8% | 1.06 (2.919e-01) |

*Note: Evaluated across cross-sectional daily rank correlations on the out-of-time test partition.*
