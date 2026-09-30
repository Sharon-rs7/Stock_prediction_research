# Human Writing Revision Report

**Document:** `FINAL_RESEARCH_PAPER_HUMAN_ACADEMIC_IEEE.docx`  
**Revision Date:** 2026-09-30  
**Target:** Natural Academic Prose (Student/Researcher Co-Authored Tone)  
**Verification:** **100% PRESERVED RESEARCH & METRICS**

---

## 1. Overview of Revision

The research manuscript was thoroughly rewritten to eliminate formulaic, machine-generated language patterns while preserving all verified experimental results, methodology, figures, tables, equations, and references.

The prose now reads as a natural, technically precise academic paper written and edited by a student research team.

---

## 2. Main AI-Like Writing Patterns Removed

| Section | AI-Generated Pattern / Phrasing Removed | Rewritten Natural Academic Phrasing |
| :--- | :--- | :--- |
| **Title / Abstract** | *"This investigation provides a rigorous, bias-controlled machine learning framework..."* | *"We investigate whether cross-sectional stock return prediction can be improved by incorporating market-wide context into single-stock OHLCV features..."* |
| **Title / Abstract** | *"Method C rank fusion achieves an empirical measurement of 88.0% variance reduction..."* | *"Method C lowers 5-day excess-return volatility from 10.856% to 3.755% (an 88.0% variance reduction)..."* |
| **Section I: Intro** | *"automated forecasting of equity price dynamics and automated peer-asset recommendation remain central challenges in empirical quantitative finance..."* | *"Predicting stock returns is notoriously difficult. Asset returns have a low signal-to-noise ratio, non-stationary distributions, and rapid price adjustments..."* |
| **Section I: Intro** | *"The empirical viability of similarity-based recommendation under realistic turnover and execution friction remains undocumented..."* | *"Few studies directly compare single-stock technical indicators with market-wide context within the exact same algorithmic pipeline..."* |
| **Section III: Data** | *"A rigorous forensic audit of the 6,708 ingested securities revealed significant heterogeneity..."* | *"We first examined the 6,708 files for asset type, price validity, trading activity, and date coverage."* |
| **Section IV: Method** | *"To rigorously test whether technical indicators add value relative to market context, we formalize..."* | *"To compare the value of market context against additional technical indicators, we defined four feature configurations..."* |
| **Section IV: Method** | *"The forecasting model utilizes LightGBM optimized under the Huber loss objective..."* | *"We used LightGBM with the Huber loss objective for the forecasting task..."* |
| **Section VI: Results**| *"Level 2 demonstrates a substantial and statistically meaningful empirical superiority..."* | *"The Level 2 model produced a test Rank IC of 0.0160, compared with 0.0084 for Level 1 (+91.1%). The paired HAC test, however, yielded p = 0.1927, so the difference was not statistically significant at the 0.05 level."* |
| **Section VI: Accuracy**| *"The model achieves 65.68% accuracy under selective prediction..."* | *"When predictions were restricted to the highest-conviction 10.23% of cases, UP-call precision reached 65.68% (with directional accuracy of 56.89%)."* |
| **Section VII: Recs** | *"Method C provides an optimal and robust institutional balance between alpha and risk..."* | *"Method C produced substantially lower excess-return volatility than Method A in the evaluated sample."* |
| **Section VIII: Costs**| *"Method C is viable in institutional execution tiers."* | *"In the simulated cost analysis, net excess return remained positive at 5 and 10 bps but became slightly negative at 15 bps. This indicates that transaction costs are an important constraint..."* |
| **Section X: Discussion**| *"Our empirical results provide nuanced insights into equity return predictability..."* | *"The results show two main patterns. Market-aware features improve the observed Rank IC, while hybrid rank fusion reduces the variability of the recommendation portfolios."* |
| **Section XI: Limits** | *"disclosed with radical transparency..."* | *"We note nine specific limitations of this study..."* |
| **Section XII: Concl** | Repeated Abstract text nearly verbatim with buzzwords. | Concise, direct summary answering: What was tested? What was observed? What remains uncertain? What should be studied next? |

---

## 3. Important Technical Claims Preserved

1. **Research Question 1 (RQ1):**
   - Level 1 Baseline (30 features): Rank IC = **0.0084**
   - Level 2 Market-Aware (49 features): Rank IC = **0.0160** (+91.1% empirical gain)
   - Level 3 Expanded Technical (39 features): Rank IC = **0.0084** (0.0% incremental gain)
   - Level 4 Full Combined (58 features): Rank IC = **0.0159**
   - Paired Newey-West HAC inference: **t = 1.3027, p = 0.1927**, bootstrap 95% CI `[-0.00032, +0.01559]`. Strictly documented as **not statistically significant** at alpha = 0.05.
2. **Directional Accuracy & Selective Prediction:**
   - Full-universe unconditional directional accuracy: **53.72%** (737,805 samples).
   - Selective UP precision: **60.82%** at 25.08% coverage; **65.68%** at 10.23% coverage (directional accuracy 56.89%).
   - Platt scaling: ECE = **0.53%**, Brier score = **0.24868**, formula $P(Y > 0 \mid \hat{z}) = [1 + \exp(-(0.5218\hat{z} + 0.0954))]^{-1}$.
3. **Research Question 2 (RQ2):**
   - Method A (Prediction-Only): Gross mean = +1.663%, median = -0.914%, volatility = 10.856%, hit rate = 47.54%.
   - Method B (Similarity-Only): Gross mean = -0.068%, volatility = 4.103%, hit rate = 48.91%.
   - Method C (50/50 Rank Fusion): Gross mean = +0.113%, median = +0.007%, volatility = 3.755% (**88.0% variance reduction**), hit rate = 50.25%.
4. **Turnover & Transaction Costs:**
   - 5-day turnover: **85.1%**.
   - Net excess returns: **+0.071%** at 5 bps, **+0.028%** at 10 bps, **-0.014%** at 15 bps. Breakeven: **~13.3 bps**.
5. **Exploratory Demarcation:**
   - H = 1 Day Champion Model (Rank IC = 0.0221, t = 2.95, p = 0.0034), volatility-penalized Method C2 (+0.59% gross excess), and stacking blends are strictly isolated in Section IX (Exploratory Analysis).
6. **Limitations:**
   - All 9 verified limitations preserved and clearly explained in Section XI.

---

## 4. Figures and Tables Preserved

| Item | Number | In-Text Verification | Status |
| :--- | :---: | :--- | :---: |
| **TABLE I** | Dataset and Universe Construction Funnel | Section III-C | **PRESERVED** |
| **TABLE II** | Four-Tier Feature Configuration | Section IV-D | **PRESERVED** |
| **TABLE III** | Out-of-Time Forecasting Performance Across Tiers | Section VI-A | **PRESERVED** |
| **TABLE IV** | Statistical Significance Analysis | Section VI-C | **PRESERVED** |
| **TABLE V** | Selective Prediction Performance Across 11 Tiers | Section VI-D | **PRESERVED** |
| **TABLE VI** | Out-of-Time Probability Calibration Diagnostics | Section VI-E | **PRESERVED** |
| **TABLE VII** | Top-5 Recommendation Performance Under Frictions | Section VII-B | **PRESERVED** |
| **TABLE VIII** | Method C Recommendations for AAPL (2026-09-16) | Section VII-D | **PRESERVED** |
| **Fig. 1** | End-to-End Research Framework Flowchart | Section IV-A | **PRESERVED & EMBEDDED** |
| **Fig. 2** | Multi-Gate Universe Construction Funnel | Section III-C | **PRESERVED & EMBEDDED** |
| **Fig. 3** | Chronological Purged Split Timeline Diagram | Section III-E | **PRESERVED & EMBEDDED** |
| **Fig. 4** | Rank IC Across Four Feature Configurations | Section VI-A | **PRESERVED & EMBEDDED** |
| **Fig. 5** | Paired Rank IC Comparison (Level 1 vs Level 2) | Section VI-B | **PRESERVED & EMBEDDED** |
| **Fig. 6** | Selective Directional Accuracy Across Coverage | Section VI-D | **PRESERVED & EMBEDDED** |
| **Fig. 7** | UP-Call Precision Across Coverage Tiers | Section VI-D | **PRESERVED & EMBEDDED** |
| **Fig. 8** | Platt Probability Calibration Diagnostics | Section VI-E | **PRESERVED & EMBEDDED** |
| **Fig. 9** | Pairwise Return Correlation Heatmap (14 Assets) | Section IV-G | **PRESERVED & EMBEDDED** |
| **Fig. 10** | Method C Output Demonstration for AAPL | Section VII-D | **PRESERVED & EMBEDDED** |
| **Fig. 11** | Variance Reduction Comparison (Method A vs C) | Section VII-C | **PRESERVED & EMBEDDED** |
| **Fig. 12** | Transaction Cost Sensitivity Curve | Section VIII-A | **PRESERVED & EMBEDDED** |
| **Fig. 13** | Multi-Panel Performance Summary Across Tiers | Section VI-A | **PRESERVED & EMBEDDED** |
| **References** | 14 Real IEEE Citations ([1] to [14]) | References | **PRESERVED** |

---

## 5. Numerical Consistency Check

Every quantitative metric reported in `FINAL_RESEARCH_PAPER_HUMAN_ACADEMIC_IEEE.docx` was cross-checked against the project source files:
- Universe B size (2,435 stocks, 1,759 days, 4,283,165 stock-days): **100% MATCH**
- Training / Validation / Test counts (2,276,725 / 747,545 / 737,805): **100% MATCH**
- 303 evaluable test cross-sections: **100% MATCH**
- Rank ICs (0.0084, 0.0160, 0.0084, 0.0159): **100% MATCH**
- HAC t-stat (1.3027), p-value (0.1927), bootstrap CI `[-0.00032, +0.01559]`: **100% MATCH**
- Unconditional accuracy (53.72%), selective precision (60.82%, 65.68%): **100% MATCH**
- Platt calibration ECE (0.53%), Brier (0.24868): **100% MATCH**
- Method C variance reduction (88.0%), volatility (10.856% -> 3.755%): **100% MATCH**
- Transaction cost net excess (+0.071%, +0.028%, -0.014%): **100% MATCH**

---

## 6. Author Review Flag

No factual or numerical inconsistencies were detected. All values align exactly with project artifacts.
