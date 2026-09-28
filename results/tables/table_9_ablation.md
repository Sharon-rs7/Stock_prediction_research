### Table 9: Structural Feature Group Ablation Study (Leave-One-Group-Out)

| Feature Configuration | Out-of-Time Mean IC | $\Delta$ vs All Features | IC IR | Directional Acc. |
| :--- | :--- | :--- | :--- | :--- |
| **ALL_30_FEATURES** | 0.0059 | +0.0000 | 0.061 | 51.71% |
| **LEAVE_OUT_G1_MOMENTUM** | 0.0018 | -0.0041 | 0.019 | 51.26% |
| **LEAVE_OUT_G2_VOLATILITY** | 0.0050 | -0.0009 | 0.058 | 51.60% |
| **LEAVE_OUT_G3_TREND** | 0.0039 | -0.0020 | 0.041 | 51.36% |
| **LEAVE_OUT_G4_VOLUME** | 0.0046 | -0.0013 | 0.050 | 51.56% |
| **LEAVE_OUT_G5_BAR_GEOMETRY** | 0.0021 | -0.0038 | 0.021 | 52.24% |

*Note: Tests the empirical marginal contribution of each structural OHLCV feature group.*
