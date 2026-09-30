# Publication Figure Captions

**Document:** Authoritative Manuscript Figure Captions  
**Date:** 2026-09-29  
**Repository:** `e:\Stock_Predition\`  
**Target Output:** `results/publication_figures/FIGURE_CAPTIONS.md`  

---

### Figure 1. End-to-End Quantitative Machine Learning & Recommendation Framework

**Caption:**  
**Figure 1. End-to-End Quantitative Machine Learning & Recommendation Framework.** Schematic flowchart detailing the 11-stage methodology spanning data intake through practical execution testing.  
- **What is shown:** The complete structural pipeline from raw YahooFinance equity intake (6,708 files across 1,759 sessions, 2019–2026), data auditing and cleaning (splits/dividends adjustment, OHLC integrity, zero-volume filtering), universe construction ($N=2,435$ liquid equities in Universe B), 4-tier feature engineering ($D \in \{30, 49, 39, 58\}$), cross-sectional target standardization ($H=5$ days forward return $z$-scores), strictly chronological train/validation/test splitting with 5-day embargo purge buffers, LightGBM forecasting under Huber loss ($\delta=1.0$) with Platt logistic scaling, market-aware feature ablation, out-of-time evaluation over 303 evaluable test cross-sections (737,805 sample predictions), similarity-based peer recommendation (Method C 50/50 Hybrid Rank Fusion across 6,100 evaluated portfolios), and transaction-cost robustness verification across 5, 10, and 15 bps friction tiers.  
- **Data used:** Complete Universe B dataset (2019-09-26 to 2026-09-25).  
- **Period & Model:** Full pipeline protocol.  
- **Observations:** Readers should observe the strict linear ordering and separation between model calibration, out-of-time evaluation, and downstream portfolio recommendation, ensuring zero lookahead bias at each phase transition.

---

### Figure 2. Construction of the research universe.

**Caption:**  
**Figure 2. Construction of the research universe.** Multi-gate filtering funnel illustrating the step-by-step reduction from raw data to the final liquid equity universe.  
- **What is shown:** The quantitative progression of surviving common equities across five sequential quality and liquidity gates: (1) Raw Equity Archive ($N=6,708$; 100.0%); (2) Gate 1 Common-Equity Candidates eliminating funds, notes, warrants, and rights ($N=6,051$; 90.2%, $-657$ assets); (3) Gate 2 Data Integrity eliminating corrupt OHLC bars and inverted prices ($N=6,039$; 90.0%, $-12$ assets); (4) Gate 3 Inactivity Filtering eliminating illiquid equities with $>1\%$ zero-volume days ($N=5,053$; 75.3%, $-986$ assets); (5) Gate 4 Synchronized Trading History requiring continuous presence across all 1,759 trading sessions ($N=3,483$; 51.9%, $-1,570$ assets); and (6) Gate 5 Liquidity Floor requiring median daily dollar trading volume $\ge \$100\text{k}$ ($N=2,435$; 36.3%, $-1,048$ assets).  
- **Data used:** YahooFinance raw daily archive and metadata recorded in `metadata/universe_b_funnel.json`.  
- **Period & Model:** Full historical window (2019-09-26 to 2026-09-25); data engineering protocol.  
- **Observations:** Readers should observe that survivorship and liquidity controls systematically prune $63.7\%$ of raw assets, resulting in a strictly verified research universe of 2,435 common equities.

---

### Figure 3. Chronological Purged and Embargoed Experimental Partitions

**Caption:**  
**Figure 3. Chronological Purged and Embargoed Experimental Partitions.** Timeline diagram detailing the non-overlapping temporal data partitions and embargo purge buffers.  
- **What is shown:** The strict temporal sequence of the experimental design: Training Partition spanning 2019-09-26 to 2024-03-28 (1,134 trading sessions; 2,276,725 stock-days), followed by an initial 5-trading-day purge window (2024-03-29 to 2024-04-05); Validation Partition spanning 2024-04-08 to 2025-06-27 (307 trading sessions; 747,545 observations), followed by a second 5-trading-day purge window (2025-06-30 to 2025-07-07); and the final locked Out-of-Time Test Partition spanning 2025-07-08 to 2026-09-25 (308 calendar days; 303 evaluable daily cross-sections; 737,805 sample predictions).  
- **Data used:** Universe B equities.  
- **Period & Model:** Full historical experimental partitions.  
- **Observations:** Readers should observe that 5-day purge buffers precisely match the forward return horizon ($H=5$ days), mathematically eliminating overlapping return windows across splits. The test partition was locked and evaluated exactly once without model tuning.

---

### Figure 4. Rank IC across the four feature configurations.

**Caption:**  
**Figure 4. Rank IC across the four feature configurations.** Bar chart comparing out-of-time Mean Daily Spearman Rank Information Coefficient (Rank IC) across the four hierarchical feature sets.  
- **What is shown:** Out-of-time predictive rank correlation for Level 1 Baseline ($D=30$ features; $\text{IC}=0.0084$), Level 2 Market-Aware ($D=49$ features; $\text{IC}=0.0160$), Level 3 Expanded Technical ($D=39$ features; $\text{IC}=0.0084$), and Level 4 Combined Full ($D=58$ features; $\text{IC}=0.0159$).  
- **Data used:** Out-of-Time Test Partition (303 evaluable daily cross-sections, 737,805 predictions) recorded in `results/model_enhancement/feature_ablation.csv`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; LightGBM Huber Regressor ($\delta=1.0$).  
- **Observations:** Readers should observe that incorporating market-regime context and cross-sectional rankings in Level 2 nearly doubles empirical Rank IC (+91.1% over Baseline), whereas adding technical indicators in Level 3 yields zero incremental gain. Combining all features in Level 4 slightly degrades performance ($0.0159$), identifying Level 2 as the primary confirmatory specification.

---

### Figure 5. Paired Out-of-Time Rank IC Comparison (H = 5 Days)

**Caption:**  
**Figure 5. Paired Out-of-Time Rank IC Comparison ($H=5$ Days).** Head-to-head comparison of daily Spearman Rank IC between the Baseline Level 1 and Market-Aware Level 2 models with paired Newey-West HAC statistical diagnostics.  
- **What is shown:** Bar comparison of out-of-time Mean Daily Rank IC between Baseline Level 1 ($0.0084$) and Market-Aware Level 2 ($0.0160$), highlighting the $+91.1\%$ empirical improvement alongside statistical test annotations.  
- **Data used:** Paired daily out-of-time evaluation cross-sections ($N=303$ days) recorded in `results/model_enhancement/paired_significance_test.csv`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Level 1 Baseline vs. Level 2 Market-Aware LightGBM.  
- **Observations:** Readers should observe that while the empirical improvement is substantial (+0.0076 in Rank IC), the paired Newey-West HAC test yields $t = 1.3027$ with $p = 0.1927$ and a 95% bootstrap confidence interval of $[-0.00032, +0.01559]$. Consequently, the research paper rigorously refrains from claiming formal statistical significance at $\alpha = 0.05$.

---

### Figure 6. Selective Directional Accuracy as a Function of Coverage Tier

**Caption:**  
**Figure 6. Selective Directional Accuracy as a Function of Coverage Tier.** Empirical out-of-time directional accuracy plotted against prediction coverage for confirmatory and exploratory model architectures.  
- **What is shown:** Directional accuracy across 11 discrete coverage tiers for the confirmatory Level 2 LightGBM model (blue solid line) and across 9 tiers for the exploratory XGBoost Champion model (orange dashed line), set against the 50.0% uninformative random-guessing baseline.  
- **Data used:** Out-of-time test predictions recorded in `results/model_enhancement/detailed_selective_coverage_11tiers.csv` and `results/accuracy_optimization/08_final_test_metrics.json`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Level 2 LightGBM Huber (Confirmatory) and XGBoost Deep Regressor (Exploratory Champion).  
- **Observations:** Readers should observe that for the primary LightGBM model, directional accuracy increases monotonically from 53.72% at 100% coverage to 56.89% at 10.23% coverage as the rejection threshold excludes lower-confidence predictions. The exploratory XGBoost model begins at 52.53% at full coverage and scales to 56.72% at 0.62% coverage.

---

### Figure 7. Out-of-Time UP-Call Precision Across Coverage Tiers

**Caption:**  
**Figure 7. Out-of-Time UP-Call Precision Across Coverage Tiers.** Precision of upward return predictions ($P(Y > 0) > \tau$) as a function of prediction coverage.  
- **What is shown:** Empirical UP-call precision plotted against prediction coverage for the confirmatory Level 2 LightGBM model (green solid line) and the exploratory XGBoost Champion model (red dashed line).  
- **Data used:** Authoritative selective prediction artifacts in `results/model_enhancement/detailed_selective_coverage_11tiers.csv` and `results/accuracy_optimization/08_final_test_metrics.json`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Level 2 LightGBM Huber vs. XGBoost Deep Regressor.  
- **Observations:** Readers should observe that UP-call precision for Level 2 LightGBM rises from 55.31% at 100% coverage to 65.68% at 10.23% coverage. The exploratory XGBoost Champion reaches 66.67% UP precision at 6.76% coverage. The caption emphasizes that UP-call precision reflects the one-sided positive predictive value of long signals, distinct from two-sided directional accuracy.

---

### Figure 8. Out-of-Time Probability Calibration Diagnostics (Platt Logistic Scaling)

**Caption:**  
**Figure 8. Out-of-Time Probability Calibration Diagnostics (Platt Logistic Scaling).** Reliability diagram and sample distribution verifying post-hoc probability calibration.  
- **What is shown:** Upper panel displays the empirical frequency of positive forward returns plotted against mean predicted probabilities across 10 decile bins for the Platt-calibrated Level 2 model, compared against the perfect calibration reference line ($y = x$). Lower panel shows the distribution of sample counts across probability bins (in thousands).  
- **Data used:** 737,805 out-of-time test predictions evaluated across 303 cross-sections; calibration parameters from `results/model_enhancement/detailed_calibration_comparison.csv`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Level 2 LightGBM Huber with validation-fitted Platt scaler ($P(Y > 0 \mid \hat{y}) = [1 + \exp(-(0.5218\hat{y} + 0.0954))]^{-1}$).  
- **Observations:** Readers should observe excellent calibration fidelity across the active probability range ($0.48\text{--}0.58$), achieving an Expected Calibration Error (ECE) of 0.53%, Brier Score Loss of 0.24868, and Log Loss of 0.6905, confirming that output probabilities represent valid posterior event likelihoods.

---

### Figure 9. Rolling return-correlation structure used for similarity analysis.

**Caption:**  
**Figure 9. Rolling return-correlation structure used for similarity analysis.** Pairwise Pearson return correlation heatmap across a representative cross-section of liquid equities and recommendation peers.  
- **What is shown:** A $14 \times 14$ correlation matrix computed from 252-day rolling daily returns for mega-cap technology (AAPL, MSFT, AMZN, NVDA, GOOGL), diversified financials (JPM), healthcare/pharmaceuticals (JNJ, TAK), energy (XOM), consumer staples (PG, TM), and materials/insurance (MFC, MET, ECL).  
- **Data used:** Universe B 252-day evaluation window.  
- **Period & Model:** Out-of-time similarity evaluation window; Method B / Method C similarity kernel ($S_{i,j} = \frac{1 + \rho_{i,j}}{2}$).  
- **Observations:** Readers should observe strong intra-sector co-movement ($0.62\text{--}0.71$ across large-cap tech; $0.78$ between life insurers MFC and MET) and moderate cross-sector correlations ($0.22\text{--}0.36$), validating the structural basis of the similarity kernel in identifying return-consistent peers without introducing excessive portfolio concentration.

---

### Figure 10. Actual Method C Top-5 Recommendation Output for AAPL (2026-09-16)

**Caption:**  
**Figure 10. Actual Method C Top-5 Recommendation Output for AAPL (2026-09-16).** Architectural and data flow demonstration showing concrete peer recommendations generated by the Method C engine on a test session.  
- **What is shown:** Input target equity AAPL processed through Method C 50/50 Hybrid Rank Fusion ($R_C(j) = 0.5 R_{\text{ret}}(j) + 0.5 R_{\text{sim}}(j)$), producing the top-5 peer basket: MFC (Rank 1), MET (Rank 2), TM (Rank 3), ECL (Rank 4), and TAK (Rank 5). Each peer card displays the composite Fusion Score ($0.980\text{--}0.997$), pairwise return similarity ($0.25\text{--}0.36$), predicted upward probability ($53.0\%\text{--}53.1\%$), and 21-day relative momentum ($+4.4\%\text{--}+11.7\%$).  
- **Data used:** Authoritative live test session recorded in `results/model_enhancement/corrected_latest_session_demo.csv`.  
- **Period & Model:** Test session 2026-09-16; Method C Hybrid Rank Fusion Engine.  
- **Observations:** Readers should observe how the hybrid ranking selects peers that combine attractive model-forecast upside with moderate correlation to the target, avoiding pure momentum chasing while reducing tracking error.

---

### Figure 11. Empirical Variance and Tracking Error Reduction of Method C (6,100 Portfolios)

**Caption:**  
**Figure 11. Empirical Variance and Tracking Error Reduction of Method C (6,100 Portfolios).** Comparative dual-panel evaluation of portfolio stability across recommendation strategies.  
- **What is shown:** Panel (a) compares 5-day excess return volatility across Method A (prediction-only: 10.86%), Method B (similarity-only: 4.10%), and Method C (50/50 hybrid fusion: 3.76%). Panel (b) compares excess return variance across Method A (117.9), Method B (16.8), and Method C (14.1).  
- **Data used:** 6,100 out-of-time recommendation portfolios evaluated across 303 test cross-sections, recorded in `results/model_enhancement/detailed_top5_recommendation_comparison.csv`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Method A, Method B, and Method C recommendation engines.  
- **Observations:** Readers should observe that fusing predictive expected-return ranks with similarity ranks in Method C achieves an 88.0% variance reduction relative to prediction-only recommendations ($p < 10^{-15}$ across Levene, Brown-Forsythe, and Bartlett tests), significantly mitigating idiosyncratic tracking risk.

---

### Figure 12. Method C Out-of-Time Net Excess Return Under Transaction Costs

**Caption:**  
**Figure 12. Method C Out-of-Time Net Excess Return Under Transaction Costs.** Sensitivity curve tracking out-of-time net excess return decay as execution frictions increase.  
- **What is shown:** Mean 5-day net excess return for Method C plotted across round-trip transaction cost tiers: 0 bps (gross: $+0.113\%$), 5 bps ($+0.071\%$), 10 bps ($+0.028\%$), and 15 bps ($-0.014\%$). A dashed red line marks the zero-alpha threshold, and the breakeven cost limit is indicated.  
- **Data used:** 6,100 recommendation portfolios evaluated under empirical turnover recorded in `results/model_enhancement/detailed_top5_recommendation_comparison.csv`.  
- **Period & Model:** 2025-07-08 to 2026-09-25; Method C Hybrid Rank Fusion under proportional round-trip cost models.  
- **Observations:** Readers should observe that net alpha remains strictly positive up to approximately 13.3 bps of round-trip transaction costs under an empirical portfolio turnover of 85.1% per 5-day cycle, demonstrating that the strategy survives typical institutional liquid-equity execution costs ($\le 5\text{--}10\text{ bps}$).

---

### Figure 13. Comprehensive Performance Comparison Across the 4 Feature Tiers

**Caption:**  
**Figure 13. Comprehensive Performance Comparison Across the 4 Feature Tiers.** Multi-panel synthesis comparing the four feature configurations across distinct, non-overlapping performance metrics.  
- **What is shown:** Three separate panels displaying: (a) Out-of-time Spearman Rank IC across Level 1 ($0.0084$), Level 2 ($0.0160$), Level 3 ($0.0084$), and Level 4 ($0.0159$); (b) Generalization Gap comparing validation partition Rank IC against test partition Rank IC, highlighting temporal decay across all tiers; (c) Unconditional out-of-time directional accuracy across all four models ($51.39\%\text{--}51.55\%$) against the 50.0% random baseline.  
- **Data used:** Validation and Out-of-Time Test Partitions from `results/model_enhancement/feature_ablation.csv`.  
- **Period & Model:** Validation (2024-04-08 to 2025-06-27) and Test (2025-07-08 to 2026-09-25); LightGBM Huber Regressors.  
- **Observations:** Readers should observe that market-aware features (Level 2 and Level 4) consistently outperform technical-only features across validation and test periods. Furthermore, panel (c) emphasizes that unconditional directional accuracy remains modest ($\approx 51.5\%$) across all models, proving that selective prediction (Figures 6 & 7) is essential to unlock actionable operational accuracy.
