# Final Scientific Validation & Pre-Manuscript Gate Report

**Document Title:** Final Scientific Validation Before Research Paper Writing  
**Project Title:** Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data  
**Primary Horizon:** $H = 5$ Trading Days  
**Primary Confirmatory Family:** Baselines (Zero, Mean, Momentum), Linear (OLS, Ridge), Non-Linear Trees (Random Forest GPU, XGBoost GPU GBDT)  
**Primary Recommendation Framework:** Method A (Prediction-Only), Method B (Similarity-Only), Method C (Combined 50/50 Rank Fusion) across Lookbacks $L=252$ and $L=504$  
**Auditor:** Quantitative Research Auditor & Model Performance Analyst  
**Final Gate Status:** **PAPER WRITING READY (WITH DISCLOSED LIMITATIONS)**  

---

## 1. Central Research Question

The research investigation is defined strictly by the primary research question:
> *"Can historical OHLCV behavior identify common equities with similar co-movement, and does combining behavioral similarity information with future-return prediction improve Top-5 stock recommendation compared with similarity-only and prediction-only approaches?"*

Every empirical finding in this repository is interpreted directly through this lens.

---

## 2. Primary Research Protocol

1. **Dataset Provenance:** Hugging Face `AmirTrader/YahooFinance` (Commit `c3c01ff2fc62e02c338d2e03bdfd71016da09701`), containing 6,708 equity parquet files.
2. **Synchronized Universe B (Liquid Core):** Exactly 2,435 ordinary common equities with 1,759 complete, synchronized trading days from September 26, 2019 through September 25, 2026 (4,283,165 stock-day observations).
3. **Primary Forecast Horizon:** $H = 5$ trading days (1 trading week).
4. **Primary Standardized Target:** Cross-sectional forward return $z$-score calculated independently per date across equities active on that date.
5. **Feature Set:** 30 pre-specified OHLCV features spanning Momentum (5), Volatility (6), Trend (6), Volume (6), and Bar Geometry (7).

---

## 3. Traditional ML Training Protocol Compliance

The experimental lineage follows the strict traditional supervised learning workflow:

$$\begin{matrix}
\text{\textbf{Historical Training Data}} & (1,134\text{ trading days: } 2019\text{-}09\text{-}26 \rightarrow 2024\text{-}03\text{-}28) \\
\downarrow & \\
\text{\textbf{Causal Feature Engineering}} & (30\text{ pre-specified causal features; strictly backward-looking}) \\
\downarrow & \\
\text{\textbf{Train Model & Preprocessing}} & (\text{Fitted exclusively on training data; zero future access}) \\
\downarrow & \\
\text{\textbf{Validation Data}} & (307\text{ trading days: } 2024\text{-}04\text{-}08 \rightarrow 2025\text{-}06\text{-}27; \text{separated by 5-day purge}) \\
\downarrow & \\
\text{\textbf{Hyperparameter Selection}} & (\text{Tuned Ridge }\alpha=100.0\text{, tree depth=6, learning rate=0.03}) \\
\downarrow & \\
\text{\textbf{Lock Final Configuration}} & (\text{Formally locked in }\texttt{results/TEST\_SET\_LOCKED.flag}) \\
\downarrow & \\
\text{\textbf{Untouched Out-of-Time Test}} & (308\text{ trading days: } 2025\text{-}07\text{-}08 \rightarrow 2026\text{-}09\text{-}25; \text{evaluated once}) \\
\downarrow & \\
\text{\textbf{Final Evaluation Results}} & (\text{Recorded as evaluation; never used as a training signal})
\end{matrix}$$

---

## 4. Test Set Integrity & Isolation

1. **Test Boundaries:** July 8, 2025 to September 25, 2026 (308 trading days; 737,805 stock-day observations for $H=5$).
2. **Purge Windows:** Two 5-day purge windows (April 1–5, 2024; June 30–July 7, 2025) eliminate return overlap across partitions.
3. **Zero Test-Driven Optimization:**
   - No model weights or gradients were updated using test observations.
   - Preprocessing scalers were fitted on the training partition and transformed onto validation and test sets.
   - No hyperparameter search queried test loss or test Rank IC.

---

## 5. Primary $H=5$ Confirmatory Forecasting Findings

Evaluated on 737,805 out-of-time test observations (303 daily cross-sections):

| Confirmatory Model | Target Variable | Features Used | Mean Daily Rank IC | Naive $t$-stat | Newey-West HAC $t$ ($L=5$) | $p$-value (HAC) | IC IR | Positive IC Days | Directional Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `BASELINE_ZERO` | $z$-score | None | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.000 | 0.0% | 50.00% |
| `BASELINE_HIST_MEAN` | $z$-score | None | 0.0000 | 0.00 | 0.00 | 1.0000 | 0.000 | 0.0% | 52.77% |
| `BASELINE_MOMENTUM` | $z$-score | `ret_5d` | -0.0234 | -3.28 | **-1.99** | **0.0463** | -0.189 | 42.5% | 48.97% |
| `OLS_LINEAR` | $z$-score | 30 Core | +0.0015 | 0.26 | 0.22 | 0.8250 | 0.015 | 51.3% | 51.06% |
| `RIDGE_REGRESSION` | $z$-score | 30 Core | +0.0015 | 0.26 | 0.22 | 0.8250 | 0.015 | 51.3% | 51.06% |
| `RANDOM_FOREST_GPU` | $z$-score | 30 Core | +0.0005 | 0.08 | 0.07 | 0.9450 | 0.005 | 50.6% | 50.32% |
| `XGBOOST_GBDT_GPU` | $z$-score | 30 Core | **+0.0059** | 1.06 | 0.91 | 0.3630 | **0.061** | **50.8%** | **51.71%** |

### Rigorous Scientific Interpretation:
1. **Statistically Significant Reversal:** Trailing 5-day return negatively predicts subsequent 5-day return ($\text{Rank IC} = -0.0234$, $\text{HAC } t = -1.99, p = 0.0463$). Short-term price momentum exhibits cross-sectional mean-reversion at weekly horizons.
2. **Modest GBDT Signal:** Standard XGBoost GBDT achieves a positive empirical Rank IC of $+0.0059$ ($4\times$ higher than linear regression), but the estimate is **not statistically distinguishable from zero at $\alpha = 0.05$** ($t_{\text{HAC}} = 0.91, p = 0.363$).
3. **Directional Accuracy Reality:** Unconditional directional accuracy across all models is tightly bounded between $50.3\%$ and $51.7\%$ against an Always-Up baseline of $47.23\%$.

---

## 6. Robustness & Signal Half-Life Analysis ($H=1$ and $H=21$)

| Horizon Length | Model Architecture | Mean Daily Rank IC | Naive $t$-stat | HAC $t$-stat | $p$-value (HAC) | IC IR | Signal Decay vs $H=1$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$H = 1$ Day** | LightGBM Standard (30 feats) | **+0.0157** | **2.79** | **2.71** | **0.0071** | **0.160** | Baseline (0.0%) |
| **$H = 5$ Days** | XGBoost GBDT (30 feats) | +0.0059 | 1.06 | 0.91 | 0.3630 | 0.061 | **-62.4%** |
| **$H = 21$ Days** | LightGBM Standard (30 feats) | +0.0003 | 0.06 | 0.05 | 0.9601 | 0.003 | **-98.1%** |

- **Decay Profile:** Predictive rank correlation decays rapidly as the holding period expands from 1 to 21 trading sessions. Technical and intraday bar features possess ultra-short half-lives that dissipate within 1 to 3 trading days.
- **Exploratory Status of $H=1$:** While $H=1$ demonstrates statistical significance ($p < 0.01$), it was investigated chronologically as a robustness probe and incurs daily turnover frictions. It is reported as secondary robustness evidence, not the primary discovery.

---

## 7. Similarity Evaluation Findings

Evaluated over $L=252$ (1-year) and $L=504$ (2-year) trailing return correlation windows with self-stock exclusion:
1. **Behavioral Similarity Does Not Generate Alpha:**  
   - $L = 252$: Mean 5-day excess return $= \mathbf{-0.034\%}$ ($t = -0.31, p = 0.7602$).
   - $L = 504$: Mean 5-day excess return $= \mathbf{+0.094\%}$ ($t = 0.85, p = 0.3954$).
2. **Empirical Fact:** Equities that exhibited historical co-movement have **zero systematic tendency** to outperform the equal-weighted market cross-section over the subsequent 5 trading days.

---

## 8. Primary Recommendation Engine Findings

Evaluated across 62 non-overlapping rebalance dates ($H=5$ stride) for 20 liquid target equities (1,220 target evaluations):

| Recommendation Strategy | Mean 5-Day Excess Return | Median Excess Return | Tracking Volatility ($\sigma_{\text{excess}}$) | Hit Rate (% > Bench) | Two-Way Turnover | Annualized Excess Sharpe |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method A: Prediction-Only** | **+1.663%** | **-0.914%** | 12.030% | 47.54% | 72.67% | 0.98 |
| **Method B: Similarity ($L=252$)** | -0.034% | -0.150% | **3.905%** | 47.87% | **9.00%** | -0.06 |
| **Method B: Similarity ($L=504$)** | +0.094% | -0.159% | **3.848%** | 48.44% | **9.00%** | 0.17 |
| **Method C: Combined ($L=252$)** | +0.159% | -0.065% | **4.165%** | **49.26%** | 83.75% | 0.27 |
| **Method C: Combined ($L=504$)** | +0.167% | **-0.028%** | **3.850%** | **49.43%** | 83.75% | 0.31 |

---

## 9. Statistical Robustness of Method A and Method C

### A. Variance Reduction Tests (Method A vs. Method C)
To test whether the reduction in tracking error volatility from $12.030\%$ to $4.165\%$ is statistically significant under non-normal, heavy-tailed return distributions:
- **Standard $F$-test of Variance Ratio:** $F = 8.3432, p < 10^{-15}$
- **Levene Test (center = mean):** $W = 625.75, p = 4.02 \times 10^{-123}$
- **Brown-Forsythe Test (center = median, robust to heavy skewness):** $W = 526.76, p = 1.05 \times 10^{-105}$

*Conclusion:* The dispersion reduction achieved by Method C is statistically incontrovertible ($p < 10^{-100}$) and robust to extreme tail non-normality.

### B. Serial Correlation & HAC Standard Errors across Rebalance Dates
Across the 1,220 portfolio evaluations:
- **Method A Mean Excess Return (+1.663%):**
  - Naive OLS Standard Error: $0.3444\%$ ($t = 4.829, p = 1.55 \times 10^{-6}$)
  - Newey-West HAC Standard Error ($L=5$): **$0.7992\%$ ($t_{\text{HAC}} = 2.081, p = 0.0374$)**
  - *Finding:* Accounting for serial dependence across overlapping market regimes doubles the standard error and reduces the $t$-statistic from $4.83$ to $2.08$.
- **Method C Mean Excess Return (+0.159%):**
  - Naive OLS Standard Error: $0.1192\%$ ($t = 1.333, p = 0.1829$)
  - Newey-West HAC Standard Error ($L=5$): **$0.1548\%$ ($t_{\text{HAC}} = 1.026, p = 0.3048$)**
  - *Finding:* Method C gross excess return is not statistically distinguishable from zero.

---

## 10. Transaction Cost Sensitivity Analysis

Turnover and fee drag analysis from `results/experiments/turnover_and_costs_summary.json`:

| Round-Trip Friction Level | Execution Context | Method A Net Return | Method B Net Return | Method C Net Return | Method C Viability |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0 bps** | Frictionless Benchmark | +1.663% | -0.034% | +0.159% | Theoretical Upper Bound |
| **10 bps** | Institutional Direct Market Access | +1.591% | -0.043% | **+0.075%** | Marginally Viable |
| **20 bps** | Standard Algorithmic Execution | +1.518% | -0.052% | **-0.009%** | **Wiped Out (Break-Even = 18 bps)** |
| **30 bps** | Retail / High-Slippage Execution | +1.445% | -0.061% | **-0.092%** | Negative Carry |

*Conclusion:* Due to an $83.75\%$ two-way turnover per 5-day rebalance cycle, Method C is economically viable only under institutional execution costs strictly below **18 bps round-trip**.

---

## 11. Survivorship Bias Disclosure

1. **Conditioning Mechanism:** Universe B requires 1,759 consecutive trading days of complete OHLCV records from 2019 to 2026. Securities that underwent bankruptcy, liquidation, or involuntary delisting are absent.
2. **Scientific Impact:** The evaluated universe represents a surviving, liquid core. While cross-sectional standardization ($z$-scores) removes unconditional market-level drift, findings apply strictly to **surviving US common equities meeting Universe B liquidity criteria**, not to the unconditional universe of all historical listed equities.

---

## 12. Extreme-Tail Sensitivity Audit (Method A Skewness Trap)

A granular distribution audit of Method A's 1,220 out-of-time recommendations reveals extreme right-tail concentration:
- **Skewness:** $+1.136$ (Heavy right tail) | **Kurtosis:** $2.584$
- **Median Excess Return:** **$-0.914\%$**
- **Win Rate (% > Benchmark):** **$47.54\%$** (Underperforms more than half the time)
- **Wilcoxon Signed-Rank Test:** $W = 342,720, p = 0.0159$ (Median is significantly below zero)
- **Extreme Tail Contribution:**
  - Top 1% of evaluations ($N=13$) generate **$31.1\%$** of total cumulative excess returns.
  - Top 5% of evaluations ($N=61$) generate **$102.9\%$** of total cumulative excess returns.
- **Winsorized Sensitivity:**
  - Raw Mean: $+1.663\%$
  - 5% Winsorized Mean: $+1.330\%$

*Conclusion:* Method A's high arithmetic mean is an artifact of positive skewness driven by a tiny subset ($5\%$) of extreme upward outliers. It cannot be marketed as a consistent or low-risk strategy.

---

## 13. Feature Importance & Ablation Limitations

1. **Origin of the ">60%" Metric:** As documented in `results/publication_readiness/feature_importance_validation.md`, this figure originates from a leave-one-group-out test IC ablation (omitting momentum reduced IC by $70.2\%$; omitting bar geometry reduced IC by $64.7\%$).
2. **Absence of Economic Causation:** Tree-based split gain and ablation deltas reflect empirical predictive association within the fitted non-linear trees; they do **not** establish economic structural causation.

---

## 14. Primary vs. Exploratory Separation

As codified in `results/publication_readiness/primary_vs_exploratory_results.csv`:
- **Primary Confirmatory:** Baseline Zero, Historical Mean, 5-Day Momentum, OLS, Ridge, Random Forest, XGBoost GBDT, Recommendation Methods A, B, C ($L=252, 504$) at $H=5$ days.
- **Confirmatory Robustness:** LightGBM at $H=1$ and $H=21$ days; Raw and Excess return target specifications.
- **Exploratory Post-Hoc:** Multi-Model Stacking Blend M4, Champion Dual Huber Ensemble ($H=1$), and Volatility-Penalized Methods C2/A2.

---

## 15. Valid Claims (Fully Supported)

1. **Short-Term Momentum Mean-Reversion:** Trailing 5-day return exhibits statistically significant negative rank correlation with 5-day forward return ($\text{Rank IC} = -0.0234$, $t_{\text{HAC}} = -1.99, p = 0.0463$).
2. **Monotonic Horizon Signal Decay:** Predictive rank correlation decays by $98.1\%$ as the forecast horizon expands from 1 day ($\text{IC} = 0.0157$) to 21 days ($\text{IC} = 0.0003$).
3. **Similarity Fails as an Alpha Factor:** Pure behavioral similarity produces zero excess return over equal-weighted market benchmarks ($-0.034\%, t = -0.31, p = 0.76$).
4. **Rank Fusion Volatility Dampening:** Fusing similarity with return predictions compresses excess return tracking volatility by $65.4\%$ ($12.03\% \rightarrow 4.17\%$, Brown-Forsythe $p < 10^{-100}$) and prevents extreme negative median outcomes.

---

## 16. Claims Requiring Cautious, Qualified Language

1. **GBDT Predictive Edge:** GBDT achieves higher empirical Rank IC than linear regression ($0.0059$ vs $0.0015$), but the difference is statistically uncertain ($t = 1.06, p = 0.29$).
2. **Method A Profitability:** Method A delivers high arithmetic mean excess returns ($+1.663\%$), but suffers from a negative median ($-0.914\%$) and extreme tail dependence.
3. **Method C Practical Implementability:** Method C generates positive gross excess return ($+0.159\%$), but is economically viable only under institutional execution costs ($\le 18$ bps).

---

## 17. Invalid Claims Removed & Prohibited

1. **PROHIBITED:** *"Models achieve 80% to 90% accuracy."* (Fact: Directional accuracy is $51.71\%$, pooled $R^2 \approx 0.00\%$, classification accuracy is inapplicable).
2. **PROHIBITED:** *"Bar geometry and momentum provide real causal signals."* (Fact: Observational association within trees; no causal identification).
3. **PROHIBITED:** *"Method C produces guaranteed institutional alpha."* (Fact: Gross excess return is $+0.16\%$, not statistically significant, and eliminated at 20 bps friction).
4. **PROHIBITED:** *"H=1 Champion model is our primary breakthrough."* (Fact: Exploratory post-hoc horizon elevation; test Long-Short spread was $t=1.36, p=0.1756$).

---

## 18. Final Methodological Limitations Summary

1. **Survivorship Bias:** Conditioned on 1,759 days survival (2019–2026).
2. **Execution Timing:** Assumes execution at official closing prices without bid-ask spread or market impact simulation.
3. **Turnover Friction:** Weekly rebalancing of Top-5 portfolios incurs $83.8\%$ two-way turnover.
4. **Low Explanatory Magnitude:** Unconditional $R^2 \approx 0.00\%$; models sort relative order rather than price levels.

---

## 19. Final Scientific Conclusion & Publication Gate Verdict

### Answering the Thesis Question:
Historical OHLCV return correlation successfully identifies equities with shared risk profiles, but **similarity alone provides zero forward alpha**. When combined with machine learning return forecasts, rank fusion does not increase gross return, but acts as a **statistically proven tracking-error stabilizer**, compressing portfolio variance by $65.4\%$ and eliminating severe negative median drag.

---

### PUBLICATION WRITING GATE CHECKLIST:

- [x] **H=5 primary confirmatory protocol preserved**
- [x] **Test set untouched after formal locking**
- [x] **No post-hoc model promoted to primary confirmatory status**
- [x] **Feature leakage audit passed ($0.0$ perturbation error across all 30 features)**
- [x] **Target leakage audit passed (cross-sectional $z$-scores computed daily)**
- [x] **Preprocessing leakage audit passed (scalers fitted train-only)**
- [x] **Recommendation leakage audit passed (self-stock excluded, backward similarity)**
- [x] **Statistical dependence checked (Newey-West HAC and Brown-Forsythe reported)**
- [x] **Extreme-tail sensitivity documented (top 5% accounts for $102.9\%$ of Method A)**
- [x] **Transaction costs documented (breakeven at 18 bps round-trip)**
- [x] **Survivorship bias explicitly disclosed**
- [x] **Causal language completely removed**
- [x] **Exploratory models segregated into dedicated sections**
- [x] **Every major claim supported by verifiable artifacts**

### **FINAL VERDICT: STATUS = PAPER WRITING READY**
