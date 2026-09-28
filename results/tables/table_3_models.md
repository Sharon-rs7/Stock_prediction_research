### Table 3: Machine Learning Model Architectural Specifications and Hyperparameters

| Model ID | Family / Class | Objective Function / Loss | Regularization / Key Hyperparameters | Optimization / Hardware |
| :--- | :--- | :--- | :--- | :--- |
| `BASELINE_ZERO` | Naive Zero | - | $\hat{y} = 0.0$ for all instances | Deterministic O(1) |
| `BASELINE_HIST_MEAN` | Cross-Sectional Mean | - | $\hat{y} = \bar{y}_{train}$ | Deterministic O(1) |
| `BASELINE_MOMENTUM` | Trailing Momentum | - | Raw 5-day return rank | Direct heuristic |
| `OLS_LINEAR` | Ordinary Least Squares | Mean Squared Error | None (Unpenalized) | SVD / Normal Equations |
| `RIDGE_REGRESSION` | Linear Regularized ($L_2$) | MSE + $\alpha \|w\|_2^2$ | $\alpha = 100.0$, StandardScaler (Train-fit) | Closed-form Ridge solver |
| `RANDOM_FOREST_GPU` | Bagged Decision Trees | MSE Variance Reduction | $B=50$ parallel trees, Depth=8, Subsample=0.8 | NVIDIA RTX 5050 GPU (CUDA) |
| `XGBOOST_GBDT_GPU` | Gradient Boosted Trees | MSE / Huber loss | $B=500$ trees, LR=0.03, Depth=6, Early stopping=30 | NVIDIA RTX 5050 GPU (CUDA) |

*Note: All features normalized via StandardScaler fit strictly on the training partition ($t \le 2024-03-28$). Tree models accelerated on NVIDIA GeForce RTX 5050 Laptop GPU.*
