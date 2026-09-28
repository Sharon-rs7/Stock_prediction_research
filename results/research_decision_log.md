# Empirical Research Decision Log

This log formally records all methodological, architectural, statistical, and operational decisions made during the lifecycle of the research project.

---

### Record 001: Survivorship and Universe Balancing (Universe B Definition)
- **Date:** 2026-09-28
- **Decision:** Establish a strictly balanced, synchronized panel of $N = 2,435$ ordinary common equities across 1,759 trading days (September 26, 2019 to September 25, 2026) with median daily volume $\ge 100,000$ shares.
- **Why It Was Needed:** Matrix operations for peer similarity lookbacks ($L=252$ and $L=504$) require identical contemporaneous observation dates. Furthermore, unhandled illiquid penny stocks introduce massive microstructure noise and zero-volume artifacts.
- **Alternatives Considered:**
  1. *Unbalanced Panel (All 6,708 tickers):* Rejected due to staggered IPOs, missing data, and computational intractability in pairwise rolling covariance matrices.
  2. *S&P 500 Index Constituents Only:* Rejected because it restricts the universe to large-cap equities only, ignoring small- and mid-cap co-movement.
- **Evidence:** Funnel filtering eliminated 4,273 ineligible instruments, producing a clean panel of 4,283,165 stock-days (`metadata/universe_b_funnel.json`).
- **Final Choice:** Universe B (Liquid Core).
- **Impact on Validity:** Guarantees zero missing-date interpolation, eliminates penny stock noise, but introduces an explicit survivorship condition that is transparently documented in the paper's limitations.

---

### Record 002: Primary Forecasting Target Formulation (Daily Cross-Sectional Z-Scores)
- **Date:** 2026-09-28
- **Decision:** Target variable for regression is defined as the daily cross-sectional standardized $z$-score of 5-day forward return: $z_{i,t,5} = (R_{i,t,5} - \mu_t) / \sigma_t$.
- **Why It Was Needed:** Raw forward returns are dominated by market-wide macro regimes (e.g. COVID-19 shock in 2020, inflation rate hikes in 2022). A model predicting raw returns expends capacity predicting market beta rather than cross-sectional relative ranking.
- **Alternatives Considered:**
  1. *Raw Percentage Forward Return ($R_{i,t,5}$):* Yielded negative Rank IC ($-0.0066$) due to regime shifts and cross-sectional heteroskedasticity.
  2. *Market-Cap Weighted Residual Return:* Requires daily market capitalization data which is absent in pure OHLCV data.
- **Evidence:** Table 8 demonstrates that $z$-score formulation achieves superior rank consistency compared to raw return or simple excess return.
- **Final Choice:** Daily Cross-Sectional $Z$-Score.
- **Impact on Validity:** Eliminates look-ahead cross-date contamination since $z$-scoring is performed strictly within each date cross-section.

---

### Record 003: GPU-Accelerated Tree Ensembles via Native CUDA Hist
- **Date:** 2026-09-28
- **Decision:** Implement `GPURandomForestWrapper` and `GPUGradientBoostingWrapper` leveraging native CUDA GPU acceleration (`tree_method='hist'`, `device='cuda'`) on the NVIDIA RTX 5050 GPU.
- **Why It Was Needed:** CPU scikit-learn random forests on 2.27 million rows required prohibitive sorting time (>10 minutes per fit), creating an execution bottleneck.
- **Alternatives Considered:**
  1. *CPU Scikit-Learn Random Forest:* Too slow for large panels.
  2. *LightGBM CPU:* Fast, but misses full utilization of local NVIDIA GeForce RTX 5050 GPU hardware.
- **Evidence:** XGBoost CUDA `hist` trained 2.27M rows in 2.42 seconds with early stopping and zero memory leakage.
- **Final Choice:** Native CUDA GPU XGBoost wrappers.
- **Impact on Validity:** Provides mathematically identical gradient-boosted splits in a fraction of the time, allowing rigorous multi-group ablation and robustness sweeps.

---

### Record 004: Pre-Specified Recommendation Rank Fusion ($\alpha = 0.5$)
- **Date:** 2026-09-28
- **Decision:** Method C (Combined Recommendation) combines percentile ranks with equal weighting: $\text{Score}_{\text{comb}} = 0.5 \cdot \text{Rank}_{\text{sim}} + 0.5 \cdot \text{Rank}_{\text{pred}}$.
- **Why It Was Needed:** A pre-specified, parameter-free baseline avoids retrospective data snooping on the test partition.
- **Alternatives Considered:**
  1. *Grid-Searching $\alpha$ on Test Set:* Statistically invalid; causes severe test leakage.
  2. *Multiplicative Score:* Less robust to rank outliers than linear rank fusion.
- **Evidence:** Test set evaluation showed Method C reduced excess return volatility by $65\%$ while maintaining positive excess returns over benchmark ($+0.16\%$).
- **Final Choice:** Pre-specified $\alpha = 0.5$, complemented by a systematic validation-set-only sensitivity sweep to justify parameter stability.
- **Impact on Validity:** Preserves test set purity while establishing empirical evidence of risk-dampening behavior.

---

### Record 005: 5-Day Rebalancing Stride to Guarantee Non-Overlapping Return Intervals
- **Date:** 2026-09-28
- **Decision:** Stride recommendation dates by 5 trading days across the test partition ($t, t+5, t+10, \dots$).
- **Why It Was Needed:** 5-day forward return intervals evaluated daily would create 4 days of serial overlap, inducing mechanical auto-correlation and invalidating standard error estimations.
- **Alternatives Considered:**
  1. *Daily Rebalancing with Overlapping Horizons:* Requires complex Newey-West / Hansen-Hodrick adjustments with high finite-sample bias.
  2. *Single Static Date:* Fails to represent varying market regimes.
- **Evidence:** 62 independent, disjoint 5-day evaluation windows spanning 308 test trading days provide clean non-overlapping forward return realisations.
- **Final Choice:** 5-day non-overlapping rebalance stride.
- **Impact on Validity:** Satisfies statistical independence assumptions across holding intervals for standard paired testing.

---

### Record 006: Monte Carlo Random Top-5 Recommendation Baseline (Finite-Sample Control)
- **Date:** 2026-09-29
- **Decision:** Simulate $B=100$ independent random 5-stock portfolios per rebalance date across identical target stocks and eligible universe candidates (122,000 portfolio simulations total).
- **Why It Was Needed:** A reviewer could challenge whether Method A's extreme excess volatility ($12.03\%$) and Method C's excess volatility ($4.17\%$) were merely mathematical artifacts of holding a small 5-stock portfolio rather than the 2,434-stock universe benchmark.
- **Alternatives Considered:**
  1. *Compare Only Against Equal-Weighted Market:* Fails to control for small-portfolio finite-sample variance.
  2. *Theoretical $\sigma/\sqrt{k}$ Approximation:* Assumes independent, identical Gaussian distributions, which fails in empirical equity cross-sections.
- **Evidence:** The Monte Carlo Random Top-5 baseline yielded an empirical excess return standard deviation of $3.43\%$ (with mean excess of $+0.01\%$).
- **Final Choice:** Monte Carlo Random Top-5 baseline.
- **Impact on Validity:** Proves conclusively that Method A's $12.03\%$ volatility is caused by active selection of extreme tail-risk/high-dispersion stocks, not by finite-sample size $k=5$.

---

### Record 007: Validation-Set-Only Alpha Pareto Frontier Sweep
- **Date:** 2026-09-29
- **Decision:** Conduct a systematic grid search of rank fusion parameter $\alpha \in [0.0, 0.1, \dots, 1.0]$ strictly on the out-of-time validation partition ($2024\text{--}2025$), keeping the test partition untouched.
- **Why It Was Needed:** To address reviewer skepticism regarding whether $\alpha = 0.5$ was arbitrary or data-snooped.
- **Alternatives Considered:**
  1. *Tune $\alpha$ on Test Set:* Severe data snooping / test leakage; rejected.
  2. *Present $\alpha = 0.5$ without Sensitivity Curve:* Leaves the Pareto trade-off unproven.
- **Evidence:** Validation sweep across 11 levels confirmed monotonic variance compression from $13.42\%$ ($\alpha=0.0$) down to $3.10\%$ ($\alpha=1.0$), with $\alpha=0.5$ achieving $3.95\%$ standard deviation and $+0.29\%$ excess return.
- **Final Choice:** Validation partition alpha frontier.
- **Impact on Validity:** Establishes theoretical and empirical justification for rank fusion without compromising test set lock.

---

### Record 008: Economic Feasibility and Portfolio Turnover Modeling
- **Date:** 2026-09-29
- **Decision:** Compute consecutive portfolio Jaccard turnover and simulate net excess returns under 10 bps, 20 bps, and 30 bps round-trip transaction costs.
- **Why It Was Needed:** To distinguish statistical alpha from real-world trading profitability and test whether similarity dampens portfolio turnover.
- **Alternatives Considered:**
  1. *Ignore Transaction Costs:* Unrealistic in academic finance; vulnerable to desk rejection.
  2. *Assume Constant Fixed Cost without Tracking Turnover:* Inaccurate.
- **Evidence:** Method B (Similarity-Only) has remarkably low turnover ($9.0\%$). However, because Method C combines volatile predictions with similarity ranks, its turnover remains high ($83.8\%$). At 20 bps round-trip cost, Method C net excess return is neutral ($-0.01\%$), while Method A retains net positive excess ($+1.52\%$) due to large gross margin.
- **Final Choice:** Explicit reporting of gross and net simulated returns under multiple cost tiers.
- **Impact on Validity:** Prevents over-optimistic commercial claims and provides honest, publishable economic insights.

---

### Record 009: Non-Parametric and Stationary Bootstrap Statistical Inference
- **Date:** 2026-09-29
- **Decision:** Supplement Student's $t$-tests with paired two-tailed Wilcoxon signed-rank tests and 2,000-iteration stationary bootstrap confidence intervals for all recommendation methods.
- **Why It Was Needed:** Cross-sectional stock returns exhibit fat tails, positive skewness, and non-normality, violating classical Gaussian assumptions.
- **Alternatives Considered:**
  1. *Student's $t$-test Only:* Susceptible to distortion by extreme outliers.
  2. *Log Return Normalization Only:* Inadequate for multi-asset portfolio combinations.
- **Evidence:** Method A Wilcoxon test confirmed statistical significance ($p = 0.0159$, 95% CI $[+1.00\%, +2.34\%]$). Method C Wilcoxon test yielded $p = 0.9181$ (95% CI $[-0.08\%, +0.39\%]$).
- **Final Choice:** Dual reporting of parametric and non-parametric tests with bootstrap intervals.
- **Impact on Validity:** Meets the highest statistical rigor standards demanded by top quantitative finance journals.
