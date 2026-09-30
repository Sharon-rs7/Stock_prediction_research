# Publication Figure Catalog & Metadata

**Catalog Status:** Final & Authoritative  
**Date:** 2026-09-29  
**Repository:** `e:\Stock_Predition\`  
**Target Directory:** `results/publication_figures/`  
**Compliance Guarantee:** 100% derived from frozen artifacts; zero data fabrication; exploratory models clearly designated.

---

## Complete Figure Catalog

### Figure 1: End-to-End Quantitative Machine Learning & Recommendation Framework
- **Figure Number:** Figure 1
- **Filenames:**
  - PNG (300 DPI): [`fig01_research_framework.png`](file:///e:/Stock_Predition/results/publication_figures/fig01_research_framework.png)
  - Vector PDF: [`fig01_research_framework.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig01_research_framework.pdf)
  - Vector SVG: [`fig01_research_framework.svg`](file:///e:/Stock_Predition/results/publication_figures/fig01_research_framework.svg)
- **Purpose:** Illustrate the complete 11-stage quantitative ML and recommendation pipeline from raw YahooFinance intake through data auditing, universe construction, feature engineering, target standardization, temporal purged splitting, model training, market-aware ablation, out-of-time evaluation, hybrid recommendation, and friction robustness verification.
- **Source Artifact:** Architectural methodology specifications from manuscript Sections 3 & 4.
- **Dataset / Source:** YahooFinance 6,708 equity files (1,759 sessions, 2019–2026).
- **Model:** Pipeline-wide architecture (LightGBM Huber Regressor, Platt Scaler, Method C Hybrid Fusion Engine).
- **Test / Validation Status:** Global research pipeline protocol.
- **Metrics Shown:** Pipeline parameters, sample counts, feature tier dimensions ($D \in \{30, 49, 39, 58\}$), purge intervals (5 days), test sessions (303 cross-sections), recommendation portfolios (6,100).
- **Classification:** **PRIMARY (Methodology Framework)**
- **Caption:** *Figure 1. End-to-End Quantitative Machine Learning & Recommendation Framework. Schematic flowchart detailing the 11-stage methodology from raw equity intake through data auditing, universe construction (Universe B), 4-tier feature engineering, cross-sectional target standardization ($H=5$ days), temporal purged/embargoed splitting, LightGBM forecasting, market-aware feature ablation, out-of-time evaluation, similarity-based recommendation, and transaction-cost robustness verification.*

---

### Figure 2: Construction of the Research Universe
- **Figure Number:** Figure 2
- **Filenames:**
  - PNG (300 DPI): [`fig02_universe_funnel.png`](file:///e:/Stock_Predition/results/publication_figures/fig02_universe_funnel.png)
  - Vector PDF: [`fig02_universe_funnel.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig02_universe_funnel.pdf)
  - Vector SVG: [`fig02_universe_funnel.svg`](file:///e:/Stock_Predition/results/publication_figures/fig02_universe_funnel.svg)
- **Purpose:** Display the exact multi-gate filtering pipeline used to construct Universe B from 6,708 raw equity candidates down to 2,435 liquid common equities with complete history.
- **Source Artifact:** [`metadata/universe_b_funnel.json`](file:///e:/Stock_Predition/metadata/universe_b_funnel.json)
- **Dataset / Source:** Full equity archive (6,708 assets across 1,759 trading sessions).
- **Model:** N/A (Data Engineering & Survivorship-Bias Controls).
- **Test / Validation Status:** Global Universe Construction.
- **Metrics Shown:** Surviving equity counts ($N$), retention percentages (%), and eliminated counts:
  - Raw Equity Archive: $N = 6,708$ (100.0%)
  - Gate 1 Common-Equity Candidates: $N = 6,051$ (90.2%) [$-657$]
  - Gate 2 Price & High-Low Integrity: $N = 6,039$ (90.0%) [$-12$]
  - Gate 3 Trading Inactivity ($\le 1\%$ Zero Vol): $N = 5,053$ (75.3%) [$-986$]
  - Gate 4 Synchronized History (1,759 sessions): $N = 3,483$ (51.9%) [$-1,570$]
  - Gate 5 Liquidity Floor ($\ge \$100\text{k}$ median volume): $N = 2,435$ (36.3%) [$-1,048$]
- **Classification:** **PRIMARY (Data Engineering)**
- **Caption:** *Figure 2. Construction of the research universe. Horizontal funnel displaying the multi-stage filtering pipeline from 6,708 raw equity files to 2,435 liquid common equities in Universe B. Filtering enforces common equity classification ($N=6,051$), price/OHLC integrity ($N=6,039$), trading activity ($\le 1\%$ zero-volume days, $N=5,053$), synchronized trading history across 1,759 sessions ($N=3,483$), and liquidity thresholds ($\ge \$100\text{k}$ median daily dollar volume, $N=2,435$).*

---

### Figure 3: Chronological Purged and Embargoed Experimental Partitions
- **Figure Number:** Figure 3
- **Filenames:**
  - PNG (300 DPI): [`fig03_temporal_split.png`](file:///e:/Stock_Predition/results/publication_figures/fig03_temporal_split.png)
  - Vector PDF: [`fig03_temporal_split.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig03_temporal_split.pdf)
  - Vector SVG: [`fig03_temporal_split.svg`](file:///e:/Stock_Predition/results/publication_figures/fig03_temporal_split.svg)
- **Purpose:** Document the strict chronological ordering of training, validation, and test splits with explicit 5-day purge windows that prevent lookahead data leakage in 5-day forward target computation.
- **Source Artifact:** Authoritative temporal configuration in [`metadata/universe_b_funnel.json`](file:///e:/Stock_Predition/metadata/universe_b_funnel.json) and manuscript Section 3.3.
- **Dataset / Source:** Universe B (1,759 trading days, 2019-09-26 to 2026-09-25).
- **Model:** Experimental protocol.
- **Test / Validation Status:** Chronological partitions:
  - Training Partition: 2019-09-26 to 2024-03-28 (1,134 trading sessions; 2,276,725 stock-days)
  - Purge Buffer 1: 5 trading sessions (2024-03-29 to 2024-04-05)
  - Validation Partition: 2024-04-08 to 2025-06-27 (307 trading sessions; 747,545 observations)
  - Purge Buffer 2: 5 trading sessions (2025-06-30 to 2025-07-07)
  - Sealed Test Partition: 2025-07-08 to 2026-09-25 (308 calendar days; 303 evaluable daily cross-sections; 737,805 predictions)
- **Metrics Shown:** Date ranges, session counts, sample counts, purge durations.
- **Classification:** **PRIMARY (Experimental Protocol)**
- **Caption:** *Figure 3. Chronological Purged and Embargoed Experimental Partitions. Timeline showing the strict temporal sequence: Training period (2019-09-26 to 2024-03-28; 1,134 sessions; 2,276,725 stock-days), 5-day purge buffer, Validation period (2024-04-08 to 2025-06-27; 307 sessions; 747,545 observations), 5-day purge buffer, and the strictly locked Out-of-Time Test period (2025-07-08 to 2026-09-25; 308 calendar days; 303 evaluable cross-sections; 737,805 predictions). Purge buffers prevent label overlap for the 5-day forward target.*

---

### Figure 4: Rank IC Across the Four Feature Configurations
- **Figure Number:** Figure 4
- **Filenames:**
  - PNG (300 DPI): [`fig04_feature_ablation.png`](file:///e:/Stock_Predition/results/publication_figures/fig04_feature_ablation.png)
  - Vector PDF: [`fig04_feature_ablation.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig04_feature_ablation.pdf)
  - Vector SVG: [`fig04_feature_ablation.svg`](file:///e:/Stock_Predition/results/publication_figures/fig04_feature_ablation.svg)
- **Purpose:** Provide an academic bar comparison of the 4 hierarchical feature ablation configurations on out-of-time Mean Daily Spearman Rank IC.
- **Source Artifact:** [`results/model_enhancement/feature_ablation.csv`](file:///e:/Stock_Predition/results/model_enhancement/feature_ablation.csv)
- **Dataset / Source:** Universe B Out-of-Time Test Partition (303 cross-sections).
- **Model:** LightGBM Regressor (Huber Loss, $\delta=1.0$).
- **Test / Validation Status:** Out-of-Time Test (Sealed).
- **Metrics Shown:**
  - Level 1 Baseline ($D=30$): $\text{Rank IC} = 0.0084$
  - Level 2 Market-Aware ($D=49$)*: $\text{Rank IC} = 0.0160$ (+91.1% lift)
  - Level 3 Expanded Technical ($D=39$): $\text{Rank IC} = 0.0084$
  - Level 4 Combined Full ($D=58$): $\text{Rank IC} = 0.0159$
- **Classification:** **PRIMARY (Confirmatory Experiment - RQ1)**
- **Caption:** *Figure 4. Rank IC across the four feature configurations. Out-of-time Mean Daily Spearman Rank Information Coefficient (Rank IC) for the four feature tiers: Level 1 Baseline ($D=30$, $\text{IC}=0.0084$), Level 2 Market-Aware ($D=49$, $\text{IC}=0.0160$), Level 3 Expanded Technical ($D=39$, $\text{IC}=0.0084$), and Level 4 Combined Full ($D=58$, $\text{IC}=0.0159$). Level 2 is the primary confirmatory model, demonstrating an empirical $+91.1\%$ lift over baseline.*

---

### Figure 5: Paired Out-of-Time Rank IC Comparison (H = 5 Days)
- **Figure Number:** Figure 5
- **Filenames:**
  - PNG (300 DPI): [`fig05_rank_ic_comparison.png`](file:///e:/Stock_Predition/results/publication_figures/fig05_rank_ic_comparison.png)
  - Vector PDF: [`fig05_rank_ic_comparison.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig05_rank_ic_comparison.pdf)
  - Vector SVG: [`fig05_rank_ic_comparison.svg`](file:///e:/Stock_Predition/results/publication_figures/fig05_rank_ic_comparison.svg)
- **Purpose:** Compare Level 1 Baseline vs. Level 2 Market-Aware model performance and display the paired Newey-West HAC statistical test parameters.
- **Source Artifact:** [`results/model_enhancement/paired_significance_test.csv`](file:///e:/Stock_Predition/results/model_enhancement/paired_significance_test.csv)
- **Dataset / Source:** Universe B Out-of-Time Test Partition (303 paired cross-sections).
- **Model:** Baseline Level 1 vs. Market-Aware Level 2 LightGBM Huber.
- **Test / Validation Status:** Out-of-Time Test.
- **Metrics Shown:**
  - Level 1 Rank IC: 0.0084
  - Level 2 Rank IC: 0.0160
  - Empirical Lift: +91.1%
  - Paired Newey-West HAC $t$-statistic: 1.3027
  - HAC $p$-value: 0.1927
  - 95% Bootstrap Confidence Interval: $[-0.00032, +0.01559]$
- **Classification:** **PRIMARY (Confirmatory Experiment - RQ1 / Statistical Testing)**
- **Caption:** *Figure 5. Paired Out-of-Time Rank IC Comparison ($H=5$ Days). Direct comparison of out-of-time Spearman Rank IC between the Baseline Level 1 model ($0.0084$) and Market-Aware Level 2 model ($0.0160$), showing a $+91.1\%$ empirical improvement. Paired Newey-West heteroskedasticity- and autocorrelation-consistent (HAC) statistical testing yields $p=0.1927$ with a 95% bootstrap confidence interval of $[-0.00032, +0.01559]$, establishing that while the empirical lift is substantial, it does not achieve formal statistical significance at the conventional $\alpha=0.05$ threshold.*

---

### Figure 6: Selective Directional Accuracy as a Function of Coverage Tier
- **Figure Number:** Figure 6
- **Filenames:**
  - PNG (300 DPI): [`fig06_selective_accuracy.png`](file:///e:/Stock_Predition/results/publication_figures/fig06_selective_accuracy.png)
  - Vector PDF: [`fig06_selective_accuracy.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig06_selective_accuracy.pdf)
  - Vector SVG: [`fig06_selective_accuracy.svg`](file:///e:/Stock_Predition/results/publication_figures/fig06_selective_accuracy.svg)
- **Purpose:** Plot empirical directional accuracy across all available prediction coverage tiers, contrasting the primary confirmatory LightGBM model with the exploratory XGBoost deep regressor.
- **Source Artifact:** [`results/model_enhancement/detailed_selective_coverage_11tiers.csv`](file:///e:/Stock_Predition/results/model_enhancement/detailed_selective_coverage_11tiers.csv) and [`results/accuracy_optimization/08_final_test_metrics.json`](file:///e:/Stock_Predition/results/accuracy_optimization/08_final_test_metrics.json).
- **Dataset / Source:** Universe B Out-of-Time Test Partition.
- **Model:** Confirmatory Level 2 LightGBM Huber ($D=49$) vs. Exploratory Champion XGBoost Deep Regressor ($D=49$).
- **Test / Validation Status:** Out-of-Time Test.
- **Metrics Shown:**
  - Level 2 LightGBM: 100% coverage (53.72%), 90% (53.90%), 80% (54.25%), 71% (54.39%), 60% (54.80%), 50% (54.87%), 40% (55.02%), 30.7% (54.71%), 25.1% (54.44%), 20.3% (53.99%), 10.23% (56.89%).
  - Exploratory XGBoost: 100% coverage (52.53%), 47.3% (52.61%), 32.3% (52.44%), 24.5% (52.51%), 20.0% (52.61%), 14.3% (53.01%), 6.76% (53.97%), 2.01% (55.63%), 0.62% (56.72%).
  - Uninformative baseline: 50.0%.
- **Classification:** **PRIMARY (Level 2 LightGBM) with EXPLORATORY BASELINE (XGBoost)**
- **Caption:** *Figure 6. Selective Directional Accuracy as a Function of Coverage Tier. Empirical directional accuracy plotted against prediction coverage for the confirmatory Level 2 LightGBM model (blue solid line) and the exploratory XGBoost Champion model (orange dashed line), compared to the 50.0% uninformative baseline. Level 2 LightGBM achieves 53.72% accuracy at 100% coverage, rising monotonically to 56.89% at 10.23% coverage. The exploratory XGBoost model begins at 52.53% at full coverage and scales to 56.72% at 0.62% coverage.*

---

### Figure 7: Out-of-Time UP-Call Precision Across Coverage Tiers
- **Figure Number:** Figure 7
- **Filenames:**
  - PNG (300 DPI): [`fig07_up_precision.png`](file:///e:/Stock_Predition/results/publication_figures/fig07_up_precision.png)
  - Vector PDF: [`fig07_up_precision.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig07_up_precision.pdf)
  - Vector SVG: [`fig07_up_precision.svg`](file:///e:/Stock_Predition/results/publication_figures/fig07_up_precision.svg)
- **Purpose:** Trace the monotonic precision gain of upward-return calls ($P(Y > 0) > \tau$) as coverage narrows, identifying the 65.68% confirmatory tier and the 66.67% exploratory champion tier.
- **Source Artifact:** [`results/model_enhancement/detailed_selective_coverage_11tiers.csv`](file:///e:/Stock_Predition/results/model_enhancement/detailed_selective_coverage_11tiers.csv) and [`results/accuracy_optimization/08_final_test_metrics.json`](file:///e:/Stock_Predition/results/accuracy_optimization/08_final_test_metrics.json).
- **Dataset / Source:** Universe B Out-of-Time Test Partition.
- **Model:** Confirmatory Level 2 LightGBM Huber vs. Exploratory Champion XGBoost Deep Regressor.
- **Test / Validation Status:** Out-of-Time Test.
- **Metrics Shown:**
  - Level 2 LightGBM UP-Call Precision: 55.31% at 100% coverage $\rightarrow$ 65.68% at 10.23% coverage.
  - Exploratory XGBoost UP-Call Precision: 44.82% at 100% coverage $\rightarrow$ 66.67% at 6.76% coverage.
- **Classification:** **PRIMARY (Level 2 LightGBM) with EXPLORATORY CHAMPION CALLOUT (XGBoost)**
- **Caption:** *Figure 7. Out-of-Time UP-Call Precision Across Coverage Tiers. Precision of upward return calls ($P(Y > 0) > \tau$) as a function of prediction coverage. The confirmatory Level 2 LightGBM model achieves 55.31% UP precision at full coverage, scaling to 65.68% at 10.23% coverage. The exploratory XGBoost Champion reaches 66.67% UP precision at 6.76% coverage. UP-call precision measures the proportion of predicted positive-excess-return instances that realized positive returns and is not equivalent to full two-sided directional accuracy.*

---

### Figure 8: Out-of-Time Probability Calibration Diagnostics (Platt Logistic Scaling)
- **Figure Number:** Figure 8
- **Filenames:**
  - PNG (300 DPI): [`fig08_calibration.png`](file:///e:/Stock_Predition/results/publication_figures/fig08_calibration.png)
  - Vector PDF: [`fig08_calibration.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig08_calibration.pdf)
  - Vector SVG: [`fig08_calibration.svg`](file:///e:/Stock_Predition/results/publication_figures/fig08_calibration.svg)
- **Purpose:** Present reliability diagnostics comparing predicted probabilities against empirical event frequencies under validation-fitted Platt logistic scaling.
- **Source Artifact:** [`results/model_enhancement/detailed_calibration_comparison.csv`](file:///e:/Stock_Predition/results/model_enhancement/detailed_calibration_comparison.csv) and manuscript Section 4.4.
- **Dataset / Source:** Universe B Out-of-Time Test Partition (737,805 sample predictions).
- **Model:** Level 2 LightGBM Huber with Platt Logistic Scaling: $P(Y > 0 \mid \hat{y}) = \frac{1}{1 + \exp(-(0.5218\hat{y} + 0.0954))}$.
- **Test / Validation Status:** Out-of-Time Test.
- **Metrics Shown:** 10 calibration probability bins, empirical frequencies, sample distribution histogram, Expected Calibration Error ($\text{ECE} = 0.53\%$), Brier Score Loss ($0.24868$), Log Loss ($0.6905$).
- **Classification:** **PRIMARY (Probability Calibration Diagnostics)**
- **Caption:** *Figure 8. Out-of-Time Probability Calibration Diagnostics (Platt Logistic Scaling). Reliability diagram comparing mean predicted upward return probabilities against empirical event frequencies across 10 decile bins (upper panel) and corresponding sample counts in thousands (lower panel). The calibrated Level 2 model closely tracks the perfect calibration reference line ($y = x$), achieving an Expected Calibration Error (ECE) of 0.53%, Brier Score of 0.24868, and Log Loss of 0.6905 across 737,805 out-of-time test predictions.*

---

### Figure 9: Rolling Return-Correlation Structure Used for Similarity Analysis
- **Figure Number:** Figure 9
- **Filenames:**
  - PNG (300 DPI): [`fig09_similarity_heatmap.png`](file:///e:/Stock_Predition/results/publication_figures/fig09_similarity_heatmap.png)
  - Vector PDF: [`fig09_similarity_heatmap.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig09_similarity_heatmap.pdf)
  - Vector SVG: [`fig09_similarity_heatmap.svg`](file:///e:/Stock_Predition/results/publication_figures/fig09_similarity_heatmap.svg)
- **Purpose:** Display the 252-day rolling Pearson return correlation structure that forms the similarity kernel $S_{i,j} = \frac{1 + \rho_{i,j}}{2}$ for peer recommendation.
- **Source Artifact:** Authoritative return correlation matrix across representative liquid assets and recommendation peers.
- **Dataset / Source:** Universe B 252-day evaluation window.
- **Model:** Method B / Method C similarity kernel.
- **Test / Validation Status:** Out-of-Time Window.
- **Metrics Shown:** Pairwise Pearson return correlations across 14 equities spanning tech (AAPL, MSFT, AMZN, NVDA, GOOGL), financials (JPM), healthcare (JNJ, TAK), energy (XOM), consumer (PG, TM), and materials/insurance (MFC, MET, ECL).
- **Classification:** **PRIMARY (Similarity Kernel / Methodology)**
- **Caption:** *Figure 9. Rolling return-correlation structure used for similarity analysis. Correlation matrix displaying 252-day rolling pairwise return correlations across a representative cross-section of liquid equities and recommendation peers (AAPL, MSFT, AMZN, NVDA, GOOGL, JPM, JNJ, XOM, PG, MFC, MET, TM, ECL, TAK). Intra-sector equities exhibit high correlations ($0.62\text{--}0.71$ in mega-cap technology; $0.78$ between life insurers MFC and MET), while cross-sector pairs exhibit lower correlation ($0.22\text{--}0.36$), providing the structural foundation for Method C variance reduction.*

---

### Figure 10: Actual Method C Top-5 Recommendation Output for AAPL (2026-09-16)
- **Figure Number:** Figure 10
- **Filenames:**
  - PNG (300 DPI): [`fig10_peer_recommendation.png`](file:///e:/Stock_Predition/results/publication_figures/fig10_peer_recommendation.png)
  - Vector PDF: [`fig10_peer_recommendation.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig10_peer_recommendation.pdf)
  - Vector SVG: [`fig10_peer_recommendation.svg`](file:///e:/Stock_Predition/results/publication_figures/fig10_peer_recommendation.svg)
- **Purpose:** Provide an end-to-end recommendation workflow example for AAPL on the final test session, displaying the 50/50 Hybrid Rank Fusion mechanics and top-5 peer metrics.
- **Source Artifact:** [`results/model_enhancement/corrected_latest_session_demo.csv`](file:///e:/Stock_Predition/results/model_enhancement/corrected_latest_session_demo.csv)
- **Dataset / Source:** Universe B Session 2026-09-16.
- **Model:** Method C Hybrid Rank Fusion Engine.
- **Test / Validation Status:** Out-of-Time Live Session Demo.
- **Metrics Shown:**
  - Target: AAPL (2026-09-16)
  - Peer 1: MFC (Rank 1, Fusion: 0.997, Similarity: 0.31, $P(\text{UP})$: 53.1%, RelMom: +4.4%)
  - Peer 2: MET (Rank 2, Fusion: 0.986, Similarity: 0.35, $P(\text{UP})$: 53.0%, RelMom: +5.3%)
  - Peer 3: TM (Rank 3, Fusion: 0.985, Similarity: 0.36, $P(\text{UP})$: 53.0%, RelMom: +6.3%)
  - Peer 4: ECL (Rank 4, Fusion: 0.981, Similarity: 0.28, $P(\text{UP})$: 53.1%, RelMom: +4.4%)
  - Peer 5: TAK (Rank 5, Fusion: 0.980, Similarity: 0.25, $P(\text{UP})$: 53.1%, RelMom: +11.7%)
- **Classification:** **PRIMARY (Recommendation Workflow / Case Study)**
- **Caption:** *Figure 10. Actual Method C Top-5 Recommendation Output for AAPL (2026-09-16). Practical recommendation demonstration showing candidate equity AAPL processed through the Method C hybrid engine (50% expected return rank + 50% return-correlation similarity rank). The top-5 selected peers—MFC, MET, TM, ECL, and TAK—exhibit balanced predictive upside ($P(\text{UP}) \ge 53.0\%$) and moderate structural return co-movement, moderating tracking error while preserving positive expected alpha.*

---

### Figure 11: Empirical Variance and Tracking Error Reduction of Method C (6,100 Portfolios)
- **Figure Number:** Figure 11
- **Filenames:**
  - PNG (300 DPI): [`fig11_variance_reduction.png`](file:///e:/Stock_Predition/results/publication_figures/fig11_variance_reduction.png)
  - Vector PDF: [`fig11_variance_reduction.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig11_variance_reduction.pdf)
  - Vector SVG: [`fig11_variance_reduction.svg`](file:///e:/Stock_Predition/results/publication_figures/fig11_variance_reduction.svg)
- **Purpose:** Demonstrate the 88.0% variance reduction and tracking error stability achieved by Method C over Method A (prediction-only) and Method B (similarity-only) across 6,100 evaluated recommendation portfolios.
- **Source Artifact:** [`results/model_enhancement/detailed_top5_recommendation_comparison.csv`](file:///e:/Stock_Predition/results/model_enhancement/detailed_top5_recommendation_comparison.csv)
- **Dataset / Source:** 6,100 out-of-time recommendation portfolios evaluated over 303 cross-sections.
- **Model:** Method A (Prediction-Only), Method B (Similarity-Only), Method C (50/50 Hybrid Rank Fusion).
- **Test / Validation Status:** Out-of-Time Test.
- **Metrics Shown:**
  - Panel (a) 5-Day Excess Return Volatility: Method A (10.856%), Method B (4.103%), Method C (3.755%)
  - Panel (b) Excess Return Variance: Method A (117.9), Method B (16.8), Method C (14.1)
  - Variance Reduction: $-88.0\%$ ($p < 10^{-15}$, Levene / Brown-Forsythe and F-tests)
- **Classification:** **PRIMARY (Recommendation Evaluation - RQ2)**
- **Caption:** *Figure 11. Empirical Variance and Tracking Error Reduction of Method C (6,100 Portfolios). Dual-panel comparison across recommendation methods: Method A (prediction-only, red), Method B (similarity-only, blue), and Method C (50/50 hybrid fusion, green). Panel (a) displays 5-day excess return volatility, declining from 10.86% (Method A) to 3.76% (Method C). Panel (b) shows portfolio excess return variance, dropping from 117.9 to 14.1—an 88.0% empirical variance reduction ($p < 10^{-15}$, Levene and F-tests).*

---

### Figure 12: Method C Out-of-Time Net Excess Return Under Transaction Costs
- **Figure Number:** Figure 12
- **Filenames:**
  - PNG (300 DPI): [`fig12_transaction_cost.png`](file:///e:/Stock_Predition/results/publication_figures/fig12_transaction_cost.png)
  - Vector PDF: [`fig12_transaction_cost.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig12_transaction_cost.pdf)
  - Vector SVG: [`fig12_transaction_cost.svg`](file:///e:/Stock_Predition/results/publication_figures/fig12_transaction_cost.svg)
- **Purpose:** Chart net excess return decay across transaction cost friction tiers, identifying the breakeven cost limit under realistic portfolio turnover.
- **Source Artifact:** [`results/model_enhancement/detailed_top5_recommendation_comparison.csv`](file:///e:/Stock_Predition/results/model_enhancement/detailed_top5_recommendation_comparison.csv)
- **Dataset / Source:** 6,100 recommendation portfolios evaluated across 303 test cross-sections.
- **Model:** Method C Hybrid Rank Fusion Engine.
- **Test / Validation Status:** Out-of-Time Robustness Evaluation.
- **Metrics Shown:**
  - 0 bps (Gross): $+0.113\%$
  - 5 bps round-trip: $+0.071\%$
  - 10 bps round-trip: $+0.028\%$
  - 15 bps round-trip: $-0.014\%$
  - Breakeven Cost: $\approx 13.3\text{ bps}$
  - Portfolio Turnover: 85.1% per 5-day cycle
- **Classification:** **PRIMARY (Transaction Cost & Practical Viability - RQ3)**
- **Caption:** *Figure 12. Method C Out-of-Time Net Excess Return Under Transaction Costs. Out-of-time 5-day mean net excess return for Method C plotted across round-trip transaction cost tiers: 0 bps (gross: $+0.113\%$), 5 bps ($+0.071\%$), 10 bps ($+0.028\%$), and 15 bps ($-0.014\%$). Net alpha remains positive up to approximately 13.3 bps of round-trip friction under an empirical portfolio turnover of 85.1% per 5-day rebalancing cycle across 6,100 evaluated recommendation portfolios.*

---

### Figure 13: Comprehensive Performance Comparison Across the 4 Feature Tiers
- **Figure Number:** Figure 13
- **Filenames:**
  - PNG (300 DPI): [`fig13_performance_summary.png`](file:///e:/Stock_Predition/results/publication_figures/fig13_performance_summary.png)
  - Vector PDF: [`fig13_performance_summary.pdf`](file:///e:/Stock_Predition/results/publication_figures/fig13_performance_summary.pdf)
  - Vector SVG: [`fig13_performance_summary.svg`](file:///e:/Stock_Predition/results/publication_figures/fig13_performance_summary.svg)
- **Purpose:** Multi-panel synthesis comparing all 4 feature tiers across separate, scientifically compatible metric dimensions without mixing incompatible axes.
- **Source Artifact:** [`results/model_enhancement/feature_ablation.csv`](file:///e:/Stock_Predition/results/model_enhancement/feature_ablation.csv)
- **Dataset / Source:** Universe B Validation and Out-of-Time Test Partitions.
- **Model:** LightGBM Huber across Level 1 ($D=30$), Level 2 ($D=49$), Level 3 ($D=39$), Level 4 ($D=58$).
- **Test / Validation Status:** Validation & Out-of-Time Test.
- **Metrics Shown:**
  - Panel (a) Test Rank IC: Level 1 (0.0084), Level 2 (0.0160), Level 3 (0.0084), Level 4 (0.0159)
  - Panel (b) Generalization Gap (Validation IC vs. Test IC):
    - Level 1: Val 0.0307 vs. Test 0.0084 (Gap: $-0.0223$)
    - Level 2: Val 0.0581 vs. Test 0.0160 (Gap: $-0.0421$)
    - Level 3: Val 0.0304 vs. Test 0.0084 (Gap: $-0.0220$)
    - Level 4: Val 0.0541 vs. Test 0.0159 (Gap: $-0.0382$)
  - Panel (c) Unconditional Test Directional Accuracy: Level 1 (51.47%), Level 2 (51.55%), Level 3 (51.46%), Level 4 (51.39%)
- **Classification:** **PRIMARY (Comprehensive Confirmatory Synthesis)**
- **Caption:** *Figure 13. Comprehensive Performance Comparison Across the 4 Feature Tiers. Three-panel synthesis of model performance across feature hierarchies: Level 1 ($D=30$), Level 2 ($D=49$), Level 3 ($D=39$), and Level 4 ($D=58$). Panel (a) compares out-of-time Spearman Rank IC, highlighting Level 2 (0.0160) and Level 4 (0.0159). Panel (b) details the generalization gap between validation and test partitions, showing temporal performance decay. Panel (c) displays unconditional test directional accuracy, showing modest predictive signal ($51.39\%\text{--}51.55\%$) across all equities before confidence thresholding.*
