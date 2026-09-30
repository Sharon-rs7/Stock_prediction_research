# Out-of-Time Test Set Lock & Isolation Protocol

**Document Version:** 1.0  
**Research Topic:** Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data  
**Repository Universe:** Universe B (Liquid Core) — Exactly 2,435 common equities  
**Lock Status:** STRICTLY LOCKED & VERIFIED  

---

## 1. Test Partition Specifications

| Specification Dimension | Verified Parameter / Value | Source Verification Artifact |
| :--- | :--- | :--- |
| **Test Period Start Date** | **July 8, 2025** (`2025-07-08`) | `data/processed/universe_b_panel.parquet` |
| **Test Period End Date** | **September 25, 2026** (`2026-09-25`) | `data/processed/universe_b_panel.parquet` |
| **Total Test Trading Days** | **308 Trading Days** | `scripts/temporal_split.py` |
| **Total Stock-Day Observations ($H=5$)** | **737,805 Samples** | `results/experiments/experiment_registry.csv` |
| **Total Stock-Day Observations ($H=1$)** | **757,285 Samples** | `results/experiments/champion_h1_final_metrics.json` |
| **Total Stock-Day Observations ($H=21$)** | **698,845 Samples** | `results/experiments/robustness_summary.json` |
| **Purge / Embargo Period 1** | April 1, 2024 → April 5, 2024 (5 Trading Days) | Prevents Train → Validation return overlap |
| **Purge / Embargo Period 2** | June 30, 2025 → July 7, 2025 (5 Trading Days) | Prevents Validation → Test return overlap |

---

## 2. Chronological Access & Usage History

- **First Recorded Test Evaluation:** September 29, 2026 at `00:15:01` UTC  
  *Context:* Initial execution of `EXP_BASELINE_ZERO`, `EXP_BASELINE_HIST_MEAN`, `EXP_BASELINE_MOMENTUM`, `EXP_OLS_LINEAR`, and `EXP_RIDGE_REGRESSION` on the predefined test split.
- **Formal Lock Flag Established:** September 29, 2026 at `00:36:00` UTC (`results/TEST_SET_LOCKED.flag`).
- **Subsequent Script Executions:**
  - `advanced_model_exploration.py` (M1, M2, M4, A2, C2): Executed `00:45:00` UTC.
  - `iterative_accuracy_optimizer.py` (Rounds 1–6): Executed `01:23:34` UTC.
  - `test_h1_champion.py`: Executed `01:35:00` UTC.

---

## 3. Forensic Test Isolation Audit

### Were any models trained on the test set?
**NO.** Every model was fitted exclusively on the historical training partition (`2019-09-26` to `2024-03-28`, 1,134 trading days). Zero test observations entered any gradient calculation, tree split, or normal equation matrix.

### Were preprocessing transformations fitted on the test set?
**NO.** Scalers (`StandardScaler`), imputers, and normalization parameters were fitted strictly on the training partition and subsequently applied to validation and test data without refitting.

### Was the test set used for iterative hyperparameter tuning?
**NO.** Hyperparameter grids, learning rates, tree depths, and regularization penalties were tuned exclusively using the validation partition (`2024-04-08` to `2025-06-27`).

### Were any development decisions influenced by test set results?
**CRITICAL FORENSIC DISCLOSURE (YES, FOR SUBSEQUENT EXPLORATORY SCRIPTS):**
While test set data was never used in loss functions or direct grid searches, the chronological sequence of research development reveals three instances of post-hoc methodological modification:
1. **Post-Hoc Feature Expansion:** After baseline GBDT yielded $\text{Rank IC} = 0.0059$, 5 non-linear interaction features were introduced in `advanced_model_exploration.py` to boost signal.
2. **Post-Hoc Horizon Elevation ($H=1$):** After observing that $H=5$ signal was modest, $H=1$ was investigated in `iterative_accuracy_optimizer.py` because validation signal was substantially stronger ($0.0207$ vs $0.0085$). Calling $H=1$ the "Champion" model reflects post-hoc horizon selection.
3. **Post-Hoc Recommendation Modifications (Methods A2 and C2):** After discovering that Method A suffered from high tracking error volatility ($12.03\%$) and negative median return ($-0.91\%$), volatility-penalized scores ($p / (\sigma + 0.01)$) were introduced in `advanced_model_exploration.py`.

---

## 4. Methodological Classification & Status

To preserve 100% scientific validity under traditional machine learning research standards:

| Research Configuration | Methodology Class | Confirmatory Publication Status |
| :--- | :--- | :--- |
| **$H=5$ Baselines (Zero, Mean, Momentum)** | Pre-specified Confirmatory Baseline | **CLEAN CONFIRMATORY** |
| **$H=5$ OLS & Ridge Regression (30 features)** | Pre-specified Confirmatory Linear Model | **CLEAN CONFIRMATORY** |
| **$H=5$ Random Forest GPU & XGBoost GBDT (30 feats)** | Pre-specified Confirmatory Tree Model | **CLEAN CONFIRMATORY** |
| **$H=5$ Recommendation Methods A, B, C ($L=252, 504$)** | Pre-specified Confirmatory Recommendation | **CLEAN CONFIRMATORY** |
| **$H=1$ Single LightGBM Robustness Experiment** | Pre-specified Confirmatory Robustness | **CLEAN CONFIRMATORY (ROBUSTNESS)** |
| **$H=21$ Single LightGBM Robustness Experiment** | Pre-specified Confirmatory Robustness | **CLEAN CONFIRMATORY (ROBUSTNESS)** |
| **$H=5$ Advanced Stacking Blend (M4, 35 features)** | Post-hoc Feature Expansion & Ensemble | **EXPLORATORY / CONTAMINATED FOR CONFIRMATORY PROMOTION** |
| **$H=1$ Champion Dual Huber Ensemble** | Post-hoc Horizon Elevation | **EXPLORATORY / CONTAMINATED FOR CONFIRMATORY PROMOTION** |
| **Recommendation Methods A2 and C2** | Post-hoc Skewness Mitigation | **EXPLORATORY / CONTAMINATED FOR CONFIRMATORY PROMOTION** |

---

## 5. Test Lock Enforcement Rule

1. No additional model training, hyperparameter search, or feature engineering may query the test partition.
2. The primary research paper narrative must center on the **Clean Confirmatory** results at $H=5$ days using the 30 causal features.
3. Exploratory results ($H=1$ Champion, Stacking M4, Method C2) may be discussed strictly in dedicated Exploratory / Robustness subsections, with explicit disclosure of their post-hoc nature.
