# Paper Figure Mapping & Manuscript Integration

**Document:** Authoritative Mapping of Publication Figures to Research Manuscript  
**Date:** 2026-09-29  
**Repository:** `e:\Stock_Predition\`  
**Target Output:** `results/publication_figures/PAPER_FIGURE_MAPPING.md`  

---

## 1. Executive Figure-to-Section Mapping Table

Every retained publication figure is directly aligned with a specific section of the final research paper (`paper/RESEARCH_PAPER.md` / `paper/latex/main.tex`). No orphaned, unreferenced, or duplicate figures exist.

| Figure # | Base Filename | Primary Manuscript Section | Subsection / Context | Primary Research Question / Theme | Retained Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **Figure 1** | `fig01_research_framework` | **Section 4: Methodology and Formal Formulation** | §4.0 End-to-End Workflow Architecture | Complete Pipeline Protocol | **RETAINED** |
| **Figure 2** | `fig02_universe_funnel` | **Section 3: Empirical Universe and Panel Construction** | §3.1 Multi-Gate Filtration Funnel | Data Engineering & Survivorship Controls | **RETAINED** |
| **Figure 3** | `fig03_temporal_split` | **Section 3: Empirical Universe and Panel Construction** | §3.3 Purged & Embargoed Partitions | Temporal Integrity & Zero Lookahead | **RETAINED** |
| **Figure 4** | `fig04_feature_ablation` | **Section 5: Empirical Forecasting Results** | §5.1 Performance Across Feature Tiers | **RQ1: Market-Aware vs Baseline** | **RETAINED** |
| **Figure 5** | `fig05_rank_ic_comparison` | **Section 5: Empirical Forecasting Results** | §5.2 Paired Significance Testing | **RQ1: Newey-West HAC Inference** | **RETAINED** |
| **Figure 6** | `fig06_selective_accuracy` | **Section 5: Empirical Forecasting Results** | §5.3 Selective Prediction Across 11 Tiers | Directional Accuracy vs Coverage | **RETAINED** |
| **Figure 7** | `fig07_up_precision` | **Section 5: Empirical Forecasting Results** | §5.3 Selective Prediction Across 11 Tiers | UP-Call Precision vs Coverage | **RETAINED** |
| **Figure 8** | `fig08_calibration` | **Section 4: Methodology and Formal Formulation** | §4.2 Platt Scaling & Probability Calibration | Posterior Probability Reliability | **RETAINED** |
| **Figure 9** | `fig09_similarity_heatmap` | **Section 4: Methodology and Formal Formulation** | §4.3 Behavioral Peer Matching & Kernel | Similarity Kernel Foundation | **RETAINED** |
| **Figure 10** | `fig10_peer_recommendation` | **Section 6: Top-5 Recommendation Mechanics** | §6.1 Live Recommendation Demonstration | Concrete Case Study (AAPL) | **RETAINED** |
| **Figure 11** | `fig11_variance_reduction` | **Section 6: Top-5 Recommendation Mechanics** | §6.2 Portfolio Variance & Volatility | **RQ2: Variance Reduction (88.0%)** | **RETAINED** |
| **Figure 12** | `fig12_transaction_cost` | **Section 6: Top-5 Recommendation Mechanics** | §6.3 Transaction Cost & Practical Viability | **RQ3: Friction Sensitivity & Breakeven**| **RETAINED** |
| **Figure 13** | `fig13_performance_summary` | **Section 5: Empirical Forecasting Results** | §5.1–§5.2 Multi-Panel Synthesis | Comprehensive 4-Tier Comparison | **RETAINED** |

---

## 2. Detailed Sectional Integration Guide

### Section 3: Empirical Universe and Panel Construction
- **Primary Visuals:**
  - **Figure 2 (`fig02_universe_funnel`):** Accompanying Table 1 ("Dataset and Universe Construction Funnel"). It visually grounds the sequence from 6,708 ingested assets down to 2,435 liquid common equities in Universe B, illustrating the exact attrition at each quality and liquidity gate.
  - **Figure 3 (`fig03_temporal_split`):** Positioned immediately following the universe description. It anchors the chronological timeline (Training: 1,134 days; Validation: 307 days; Test: 308 calendar days / 303 evaluable sessions) and highlights the two 5-day embargo purge buffers, proving to reviewers that the forward return horizon ($H=5$) is strictly purged of lookahead leakage.

### Section 4: Methodology and Formal Formulation
- **Primary Visuals:**
  - **Figure 1 (`fig01_research_framework`):** Serves as the introductory visual for Section 4, presenting the end-to-end 11-stage quantitative pipeline from raw YahooFinance intake through data auditing, universe construction, feature engineering, cross-sectional target standardization, purged temporal splitting, LightGBM forecasting, market-aware feature ablation, out-of-time evaluation, similarity-based recommendation, and transaction-cost robustness verification.
  - **Figure 8 (`fig08_calibration`):** Placed in §4.2 ("Forecasting Objective and Huber Regressor") or §7 ("Tree Probability Degeneracy & Post-Hoc Calibration"). Demonstrates how Platt logistic scaling resolves tree classification degeneracy, mapping regression $z$-scores into empirically calibrated probabilities with an ECE of 0.53% and Brier score of 0.24868.
  - **Figure 9 (`fig09_similarity_heatmap`):** Placed in §4.3 ("Recommendation Paradigms and Rank Fusion"). Displays the empirical 252-day rolling correlation matrix for 14 representative liquid assets and peers, illustrating the structural clustering that powers the similarity kernel $S_{i,j} = \frac{1 + \rho_{i,j}}{2}$.

### Section 5: Empirical Forecasting Results
- **Primary Visuals:**
  - **Figure 4 (`fig04_feature_ablation`):** Placed in §5.1 ("Performance Across Feature Tiers") alongside Table 2. Graphically presents the Mean Daily Spearman Rank IC across Level 1 (0.0084), Level 2 (0.0160), Level 3 (0.0084), and Level 4 (0.0159), highlighting Level 2 as the primary confirmatory architecture.
  - **Figure 5 (`fig05_rank_ic_comparison`):** Placed in §5.2 ("Paired Significance and Generalization Degradation"). Emphasizes the paired out-of-time comparison between Baseline Level 1 and Market-Aware Level 2 (+91.1% empirical improvement), while explicitly presenting the paired Newey-West HAC inference ($t=1.3027, p=0.1927$, 95% bootstrap CI $[-0.00032, +0.01559]$) to prevent overstated significance claims.
  - **Figure 6 (`fig06_selective_accuracy`):** Placed in §5.3 ("Selective Prediction Across 11 Tiers"). Shows empirical directional accuracy plotted against prediction coverage, contrasting the primary Level 2 LightGBM model (scaling from 53.72% at 100% coverage to 56.89% at 10.23% coverage) with the exploratory XGBoost champion model.
  - **Figure 7 (`fig07_up_precision`):** Placed in §5.3. Focuses on UP-call precision, displaying the monotonic gain from 55.31% at full coverage to 65.68% at 10.23% coverage (Level 2 LightGBM) and highlighting the 66.67% at 6.76% coverage tier (Exploratory XGBoost Champion).
  - **Figure 13 (`fig13_performance_summary`):** Placed as a summary figure for Section 5. Integrates out-of-time Rank IC, validation-to-test generalization gap, and unconditional directional accuracy across the 4 feature tiers into a single multi-panel visualization without combining incompatible axes.

### Section 6: Top-5 Recommendation Mechanics and Performance
- **Primary Visuals:**
  - **Figure 10 (`fig10_peer_recommendation`):** Placed in §6.1 ("Recommendation Paradigm Mechanics"). Illustrates a concrete out-of-time recommendation example for target equity AAPL on session 2026-09-16, displaying the top-5 peers (MFC, MET, TM, ECL, TAK) selected by Method C 50/50 Hybrid Rank Fusion alongside their fusion scores, similarity scores, predicted upward probabilities, and relative momentum.
  - **Figure 11 (`fig11_variance_reduction`):** Placed in §6.2 ("Portfolio Stability and Risk Dampening") alongside Table 3. Dual-panel comparison demonstrating that Method C achieves an 88.0% variance reduction (excess return variance drops from 117.9 to 14.1; volatility drops from 10.86% to 3.76%; $p < 10^{-15}$ across Levene/Brown-Forsythe tests) relative to prediction-only recommendations.
  - **Figure 12 (`fig12_transaction_cost`):** Placed in §6.3 ("Transaction Cost Sensitivity and Practical Viability"). Charts net excess return decay across friction tiers (0 bps: $+0.113\%$; 5 bps: $+0.071\%$; 10 bps: $+0.028\%$; 15 bps: $-0.014\%$) under an 85.1% 5-day portfolio turnover, clearly highlighting the breakeven cost limit of approximately 13.3 bps.

---

## 3. Exclusion and Archival Log

The following legacy and diagnostic figures were evaluated during the figure audit and are formally excluded from the final paper to prevent redundancy, maintain publication focus, and eliminate exploratory bias:

1. **`results/figures/fig_1_filtering_funnel.png`**: Excluded. Replaced by publication-standard Figure 2 (`fig02_universe_funnel`).
2. **`results/figures/fig_4_feature_correlation.png`**: Excluded. Replaced by publication-standard Figure 9 (`fig09_similarity_heatmap`).
3. **`results/figures/fig_6_daily_ic_series.png`**: Excluded. Replaced by Figure 5 (`fig05_rank_ic_comparison`).
4. **`results/figures/fig_8_cumulative_trajectories.png`**: Excluded. Superseded by Figure 11 (`fig11_variance_reduction`) and Figure 12 (`fig12_transaction_cost`).
5. **`results/figures/fig_9_recommendation_comparison.png`**: Excluded. Superseded by the authoritative 6,100 portfolio evaluation in Figure 11.
6. **`results/figures/fig_10_alpha_frontier.png`**: Removed. Exploratory non-confirmatory plot not cited in manuscript.
7. **`results/figures/fig_11_model_comparison.png`**: Excluded. Replaced by Figure 4 (`fig04_feature_ablation`) and Figure 13 (`fig13_performance_summary`).
8. **`results/figures/fig_12_horizon_decay.png`**: Removed. Exploratory post-hoc horizon plot; relegated to Section 8 text.
9. **`results/statistics/figures/fig01_return_distribution.png` through `fig15_recommendation_drawdown.png`**: Retained as internal empirical statistical audit records, but not embedded as primary paper figures.
