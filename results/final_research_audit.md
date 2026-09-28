# Final Pre-Publication Comprehensive Research Audit

**Audit Date:** 2026-09-29  
**Auditor:** Senior Quantitative Research Lead  
**Project:** Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data  
**Repository:** `e:\Stock_Predition\`  

---

### 1. Data Integrity and Scope
- [x] **Dataset Version Recorded:** Public Hugging Face repository `AmirTrader/YahooFinance` at pinned commit `c3c01ff2fc62e02c338d2e03bdfd71016da09701`. All 6,708 Parquet files cryptographically hashed in `metadata/raw_dataset_manifest.json`.
- [x] **Universe Construction Reproducible:** Funnel criteria deterministically codified in `scripts/audit_universe.py`. Exactly 2,435 tickers survived all 5 gates (`metadata/universe_b_tickers.json`).
- [x] **Survivorship Bias Documented:** The requirement of 1,759 consecutive trading days conditions on survival between 2019 and 2026. This is transparently acknowledged and documented as an explicit design scope condition in Section 3 and Section 8 of the paper.

---

### 2. Target Construction and Non-Overlapping Invariants
- [x] **Forward Horizon ($H=5$):** Constructed strictly as $(AdjClose_{t+5} - AdjClose_t) / AdjClose_t$. Zero look-ahead into $t+6$.
- [x] **Cross-Sectional $Z$-Score Calculation:** Computed strictly date-by-date across contemporaneous cross-sections. No ticker-level standardization across time.
- [x] **Overlapping Return Treatment:** Recommendation rebalancing dates stride by 5 trading days ($t, t+5, t+10, \dots$), guaranteeing completely disjoint, non-overlapping holding intervals for recommendation evaluation.

---

### 3. Feature Causal Invariance (Leakage Audit)
- [x] **Feature Verification:** All 30 technical features across all 5 groups tested via future-perturbation invariance. Maximum perturbation delta at $\le t$ when $t+1$ perturbed is exactly $0.00\text{e}+00$.
- [x] **Formal Audit Table:** Complete machine-readable verification table persisted at `results/feature_leakage_audit.csv` with 100% PASS status.

---

### 4. Temporal Split Architecture and Embargo
- [x] **Chronological Partitions:**
  - Train: 2019-09-26 $\rightarrow$ 2024-03-28 (1,134 days; 2,276,725 rows)
  - Purge Gap 1: 2024-04-01 $\rightarrow$ 2024-04-05 (5 days embargo)
  - Validation: 2024-04-08 $\rightarrow$ 2025-06-27 (307 days; 747,545 rows)
  - Purge Gap 2: 2025-06-30 $\rightarrow$ 2025-07-07 (5 days embargo)
  - Test: 2025-07-08 $\rightarrow$ 2026-09-25 (308 days; 737,805 rows)
- [x] **Test Set Lock:** Enforced and recorded at `results/TEST_SET_LOCKED.flag`. Zero hyperparameter tuning or alpha weight searches touched the test partition.

---

### 5. Model Suite and Benchmark Rigor
- [x] **Heuristic Baselines Included:** `BASELINE_ZERO`, `BASELINE_HIST_MEAN`, and `BASELINE_MOMENTUM` evaluated under identical protocols. Short-term 5-day reversal empirically demonstrated ($\text{IC} = -0.0234, t = -3.28$).
- [x] **Hardware Acceleration:** Native CUDA Hist tree training on NVIDIA GeForce RTX 5050 Laptop GPU verified.
- [x] **Leakage-Free Tuning:** Early stopping and model selection evaluated strictly against the validation partition.

---

### 6. Similarity Engine Verification
- [x] **Synthetic Unit Verification:** Mathematical convergence, boundary stability, inverse correlation ($-1.0000$), identical correlation ($+1.0000$), and orthogonal noise ($+0.0222$) confirmed in `scripts/verify_similarity_synthetic.py`.
- [x] **Self-Match Exclusion:** Verified that target stock is strictly excluded from candidate peer ranking.
- [x] **Lookback Windows:** $L=252$ (1-year) and $L=504$ (2-year) compared under identical test conditions.

---

### 7. Recommendation Engine and Fair Comparison
- [x] **Fair Comparison Protocol:** Methods A, B, C, Universe Benchmark, and Random Top-5 Monte Carlo evaluated on identical 62 rebalance dates for identical target stocks.
- [x] **Random Top-5 Baseline:** Simulated across 122,000 portfolios, proving Method A's high volatility ($12.03\%$) is driven by active tail-risk selection rather than finite-sample size $k=5$ (Random Top-5 std is $3.43\%$).
- [x] **Validation Alpha Frontier:** Parameter $\alpha \in [0.0, 1.0]$ sweep performed strictly on the validation partition, proving monotonic variance reduction.

---

### 8. Economic Feasibility and Statistical Rigor
- [x] **Turnover Analysis:** Two-way portfolio turnover quantified ($72.7\%$ Method A, $9.0\%$ Method B, $83.8\%$ Method C).
- [x] **Transaction Costs:** Net returns modeled across 10 bps, 20 bps, and 30 bps round-trip friction tiers.
- [x] **Non-Parametric Inference:** Paired Wilcoxon signed-rank tests and 2,000-iteration stationary bootstrap confidence intervals computed.

---

### 9. Scientific Integrity and Publication Readiness
- [x] **Negative Results Preserved:** Short-term momentum failure, pure similarity zero alpha, and transaction cost decay reported honestly without cherry-picking.
- [x] **Reproducibility Package:** Machine-readable manifest at `metadata/reproducibility_manifest.json` capturing OS, CPU, GPU, Python environment, and exact package versions.
- [x] **Claim Audit:** Every manuscript claim cross-referenced to experimental artifact in `results/final_paper_claim_audit.md`.

---

**FINAL AUDIT VERDICT: 100% APPROVED FOR PUBLICATION SUBMISSION**
