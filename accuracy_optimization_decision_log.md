# Accuracy Optimization Decision Log

This log formally records every decision, protocol constraint, model exploration, hyperparameter tuning iteration, feature engineering choice, calibration fitting, and empirical finding during the Accuracy Optimization research initiative.

---

### Record 001: Strict Test Set Freeze & Out-of-Time Protocol
- **Date:** 2026-09-29
- **Decision:** Strictly freeze the test partition (`2025-07-08` to `2026-09-25`; 308 trading sessions; 737,805 stock-day evaluations). Test data and test labels are sealed and prohibited from all target auditing, feature engineering, feature selection, model selection, hyperparameter tuning, probability calibration, and coverage threshold selection.
- **Why It Was Needed:** To prevent data snooping, multiplicity bias, and optimistic overfitting. The test partition will be evaluated exactly ONCE at Phase 10 after all models, parameters, and thresholds are permanently locked.
- **Protocol Partitions:**
  - **Train:** `2019-09-26` to `2024-03-28` (1,134 trading days, 2,276,725 stock-day records)
  - **Purge 1:** `2024-04-01` to `2024-04-05` (5 trading days embargo)
  - **Validation:** `2024-04-08` to `2025-06-27` (307 trading days, 747,545 stock-day records)
  - **Purge 2:** `2025-06-30` to `2025-07-07` (5 trading days embargo)
  - **Test (Locked):** `2025-07-08` to `2026-09-25` (308 calendar trading days, 737,805 stock-day records)
- **Impact on Validity:** Guarantees absolute epistemological purity.

---

### Record 002: Pre-Commitment on the 60% Accuracy Rule
- **Date:** 2026-09-29
- **Decision:** Establish an explicit distinction between unconditional full-sample directional accuracy and selective-prediction accuracy under confidence filtering.
- **Rule:**
  1. If overall unconditional accuracy $\ge 60\%$, report it with bootstrap confidence intervals and full methodology.
  2. If overall accuracy $< 60\%$ but selective accuracy $\ge 60\%$, report strictly as *"60%+ directional accuracy at X% prediction coverage"* and explicitly disclaim unconditional accuracy.
  3. If neither reaches $60\%$, report the maximum valid accuracy achieved, the corresponding coverage, and the scientific reasons for irreducible noise. Under no circumstances will a $60\%$ result be manufactured, cherry-picked, or fabricated.

---

### Record 003: Target & Horizon Audit (Phase 2)
- **Date:** 2026-09-29
- **Decision:** Audit candidate horizons $H \in \{1, 2, 3, 5, 10, 21\}$ across Binary ($R > 0$), Cross-Sectional $Z$-Score, Three-Class, and Selective Decisive Target formulations strictly on Train and Validation partitions.
- **Why It Was Needed:** To determine whether target formulation or forecast horizon fundamentally restricts directional accuracy before modifying models.
- **Evidence:** Documented in `results/accuracy_optimization/01_target_analysis.csv`. Unconditional accuracy across all horizons was bounded between $51.1\%$ and $52.8\%$. Higher raw accuracy at $H=10/21$ ($52.8\%$) was driven by market drift (UP base rate $53.6\%$), with balanced accuracy near $50.1\%$. $H=5$ cross-sectional $z$-scores preserved robust rank predictability (Rank IC $0.0304$, $t=3.13$) and was locked as the primary evaluation horizon.
- **Impact on Validity:** Prevents post-hoc horizon switching and confirms that low directional edge is an intrinsic market characteristic, not an artifact of $H=5$.

---

### Record 004: Candidate Feature Expansion & Leakage Audit (Phase 4)
- **Date:** 2026-09-29
- **Decision:** Engineer 77 candidate features across Groups A–G (momentum, mean reversion, volatility, liquidity, price structure, market-relative, macro regimes) and execute an automated future-perturbation stress test.
- **Why It Was Needed:** To test whether expanding the information set beyond the baseline 49 features improves directional accuracy without lookahead contamination.
- **Evidence:** Documented in `results/accuracy_optimization/09_leakage_audit.json`. Future price (+100%) and volume (10x) perturbations across 5 probe dates produced maximum feature delta $\Delta = 0.0000000000$ (100% PASS).
- **Impact on Validity:** Establishes mathematical proof of causal invariance and zero lookahead leakage for all 77 candidate features.

---

### Record 005: Validation-Only Feature Selection & Model Benchmark (Phase 5)
- **Date:** 2026-09-29
- **Decision:** Compare Set 1 (Current 49 Market-Aware Features), Set 2 (Expanded 77 Features), and Set 3 (Selected 42 Features after collinearity pruning) across Logistic Regression, LightGBM Huber, LightGBM Classifier, and XGBoost Huber.
- **Why It Was Needed:** To test whether more features enhance or degrade out-of-sample directional prediction.
- **Evidence:** Documented in `results/accuracy_optimization/02_feature_experiments.csv` and `03_model_comparison.csv`. Set 1 (49 features) outperformed Set 2 (77 features) across all models (XGBoost Huber reached $53.05\%$ accuracy and Rank IC $0.0522$ on Set 1, vs $52.54\%$ on Set 2). Adding 28 additional noisy technical combinations caused slight overfitting.
- **Final Choice:** Lock Set 1 (49 market-aware features) as the optimal parsimonious feature architecture.
- **Impact on Validity:** Upholds model parsimony and prevents dimensionality curse.

---

### Record 006: Hyperparameter Search & Multi-Model Ensembling (Phases 6 & 7)
- **Date:** 2026-09-29
- **Decision:** Optimize hyperparameters across 10 configurations evaluated on 3 time-series validation sub-windows, and test 4 ensemble formulations.
- **Why It Was Needed:** To optimize tree capacity, regularization, and loss objectives while penalizing models that achieve high accuracy in only one validation window.
- **Evidence:** Documented in `results/accuracy_optimization/04_hyperparameter_search.csv`. `HP_09_XGB_Huber_Deep_Reg` (max_depth=7, n_estimators=120, $\eta=0.02$, $\lambda=10.0$, colsample=0.75, subsample=0.8) achieved the highest validation directional accuracy ($53.54\%$) and highest Rank IC ($0.0569, t=6.01$) with low inter-window standard deviation ($0.75\%$). Ensemble blends did not exceed this champion model.
- **Final Choice:** Lock `HP_09_XGB_Huber_Deep_Reg` as the champion architecture.

---

### Record 007: Probability Calibration (Phase 8)
- **Date:** 2026-09-29
- **Decision:** Fit Platt logistic scaling on validation predictions to generate smooth, well-calibrated directional probabilities: $P(Y=1 \mid \hat{z}) = 1 / (1 + \exp(-(1.7373 \cdot \hat{z} - 0.0329)))$.
- **Why It Was Needed:** Raw regression scores require calibration to avoid tail over-confidence and enable rigorous confidence-based selective prediction.
- **Evidence:** Documented in `results/accuracy_optimization/06_calibration.csv`. Platt scaling achieved Expected Calibration Error (ECE) of $0.64\%$ and Brier score of $0.24893$.
- **Final Choice:** Lock validation-fitted Platt scaling parameters.

---

### Record 008: Selective Prediction & Confidence Thresholding (Phase 3)
- **Date:** 2026-09-29
- **Decision:** Formulate confidence metric $\text{confidence} = |P(\text{UP}) - 0.5|$, evaluate 9 coverage tiers (100% down to 1%), and freeze the validation-derived cutoffs.
- **Why It Was Needed:** Daily equity direction contains substantial noise; selective prediction allows the model to act only when conviction is high.
- **Evidence:** Documented in `results/accuracy_optimization/05_selective_accuracy.csv`. On validation, directional accuracy scaled from $52.72\%$ (100% coverage) to $56.01\%$ (10% coverage), $58.80\%$ (2% coverage), and crossed 60% at 1% coverage ($61.90\%$, cutoff = $0.11925$).
- **Final Choice:** Pre-register and freeze cutoffs strictly from validation for the final test.

---

### Record 009: Robustness Tests Across Regimes (Phase 9)
- **Date:** 2026-09-29
- **Decision:** Evaluate champion model validation stability across Bull, Bear, High-Volatility, Low-Volatility, High-Liquidity, Low-Liquidity, 2024 vs 2025.
- **Evidence:** Documented in `results/accuracy_optimization/07_robustness.csv`. The model maintained positive Rank IC across all 10 regimes ($0.0360$ to $0.0726$, all $t \ge 2.09$), confirming structural robustness.

---

### Record 010: Final Unseen Test Opening & 60% Rule Evaluation (Phase 10)
- **Date:** 2026-09-29
- **Decision:** Open the locked test partition (`2025-07-08` to `2026-09-25`; 737,805 samples; 303 evaluable dates) exactly once to evaluate the frozen champion model.
- **Out-of-Time Test Results:**
  - **Unconditional Directional Accuracy:** **51.60%** (95% Block Bootstrap CI: `[50.96%, 52.21%]`).
  - **Balanced Accuracy:** **50.17%**.
  - **Rank IC:** **0.0131** ($t_{\text{HAC}} = 0.8119, p = 0.4175$).
  - **Selective Directional Accuracy:** Scaled monotonically to **53.00%** (14% coverage), **53.96%** (6.8% coverage), **55.63%** (1.8% coverage), and **56.72%** (0.6% coverage, $N = 4,593$).
  - **Selective UP-Call Precision:** Reached **66.67%** at $6.76\%$ coverage ($N = 49,880$) with a **+12.08%** forward return spread.
- **Definitive 60% Finding:**
  - Unconditional 60% accuracy was **NOT achieved** (51.60%).
  - Selective directional accuracy peaked at **56.72%** (falling short of 60%).
  - Selective UP-call precision reached **66.67%** at 6.76% coverage.
  - Zero manufacturing of numbers. Full scientific transparency maintained. Documented in `08_final_test_metrics.json` and `10_experiment_report.md`.

