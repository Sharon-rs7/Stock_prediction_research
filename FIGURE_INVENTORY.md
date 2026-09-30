# Figure Inventory

**Repository:** `e:\Stock_Predition\`  
**Classification Date:** 2026-09-30  
**Associated Manuscript:** `FINAL_RESEARCH_PAPER_IEEE_STYLE.docx`

---

## 1. Primary Manuscript Figures (All Included)

| Figure # | Filename | Description | Included in Paper? | Reason for Inclusion |
| :---: | :--- | :--- | :---: | :--- |
| **Fig. 1** | `results/publication_figures/fig01_research_framework.png` | End-to-End Quantitative Machine Learning & Recommendation Framework flowchart | **YES** | Core architecture overview illustrating the 11-stage pipeline protocol. |
| **Fig. 2** | `results/publication_figures/fig02_universe_funnel.png` | Multi-gate filtering funnel from 6,708 raw assets to Universe B (2,435 stocks) | **YES** | Visually grounds sample selection, quality filtration, and liquidity floor attrition. |
| **Fig. 3** | `results/publication_figures/fig03_temporal_split.png` | Timeline diagram of chronological purged splits with 5-day embargo buffers | **YES** | Proves zero-lookahead temporal integrity across train, validation, and test partitions. |
| **Fig. 4** | `results/publication_figures/fig04_feature_ablation.png` | Out-of-time Mean Daily Rank IC across the four hierarchical feature tiers | **YES** | Core RQ1 evidence demonstrating +91.1% empirical lift for Level 2 over Level 1. |
| **Fig. 5** | `results/publication_figures/fig05_rank_ic_comparison.png` | Head-to-head daily Rank IC comparison between Baseline and Market-Aware models | **YES** | Core RQ1 inferential visualization displaying paired Newey-West HAC t = 1.3027, p = 0.1927. |
| **Fig. 6** | `results/publication_figures/fig06_selective_accuracy.png` | Selective directional accuracy as a function of prediction coverage across 11 tiers | **YES** | Demonstrates how directional accuracy increases as low-conviction predictions are excluded. |
| **Fig. 7** | `results/publication_figures/fig07_up_precision.png` | Precision of upward return calls scaling from 55.31% to 65.68% at 10.23% coverage | **YES** | Distinguishes one-sided positive predictive value from two-sided full-universe accuracy. |
| **Fig. 8** | `results/publication_figures/fig08_calibration.png` | Reliability diagram and sample distribution for Platt logistic scaling | **YES** | Verifies post-hoc calibration quality (ECE = 0.53%, Brier score = 0.24868). |
| **Fig. 9** | `results/publication_figures/fig09_similarity_heatmap.png` | 14x14 pairwise Pearson return correlation heatmap across representative assets | **YES** | Grounds the mathematical kernel S_{i,j} = (1 + rho_{i,j})/2 used in recommendation. |
| **Fig. 10** | `results/publication_figures/fig10_peer_recommendation.png` | Concrete Method C recommendation output for target equity AAPL on session 2026-09-16 | **YES** | Concrete out-of-time demonstration illustrating fusion scores, similarities, and probabilities. |
| **Fig. 11** | `results/publication_figures/fig11_variance_reduction.png` | Dual-panel volatility (10.86% to 3.76%) and variance (117.9 to 14.1) reduction of Method C | **YES** | Core RQ2 evidence proving 88.0% variance reduction over prediction-only selection. |
| **Fig. 12** | `results/publication_figures/fig12_transaction_cost.png` | Sensitivity curve tracking net excess return decay across 0, 5, 10, and 15 bps friction tiers | **YES** | Practical economic evaluation illustrating the breakeven cost limit of ~13.3 bps under 85.1% turnover. |
| **Fig. 13** | `results/publication_figures/fig13_performance_summary.png` | Multi-panel summary of Rank IC, generalization gap, and directional accuracy across 4 tiers | **YES** | Comprehensive synthesis of the 4-tier feature ablation without combining incompatible axes. |

---

## 2. Excluded / Diagnostic Figures

| Filename | Type / Path | Role | Included in Paper? | Reason for Exclusion |
| :--- | :--- | :--- | :---: | :--- |
| `fig_1_filtering_funnel.png` | `results/figures/` | Legacy funnel chart | **NO** | Superseded by publication-standard Fig. 2 (`fig02_universe_funnel.png`). |
| `fig_4_feature_correlation.png` | `results/figures/` | 30-feature correlation | **NO** | Superseded by similarity correlation heatmap Fig. 9 (`fig09_similarity_heatmap.png`). |
| `fig_6_daily_ic_series.png` | `results/figures/` | Daily IC time series | **NO** | Superseded by clean paired comparison Fig. 5 (`fig05_rank_ic_comparison.png`). |
| `fig_8_cumulative_trajectories.png` | `results/figures/` | Cumulative returns | **NO** | Superseded by variance reduction (Fig. 11) and transaction cost curves (Fig. 12). |
| `fig_9_recommendation_comparison.png`| `results/figures/` | Old recommendation checks | **NO** | Superseded by authoritative 6,100 portfolio evaluation in Fig. 11. |
| `fig_10_alpha_frontier.png` | `results/figures/` | Exploratory alpha sweep | **NO** | Exploratory non-confirmatory plot; not cited in final manuscript. |
| `fig_11_model_comparison.png` | `results/figures/` | 30-feature model zoo | **NO** | Superseded by 4-tier ablation (Fig. 4) and performance summary (Fig. 13). |
| `fig_12_horizon_decay.png` | `results/figures/` | Exploratory horizon plot | **NO** | Exploratory post-hoc horizon plot; relegated to Section IX text. |
| `fig01_return_distribution.png` to `fig15_recommendation_drawdown.png` | `results/statistics/figures/` | Internal audit diagnostics | **NO** | Retained strictly in `results/statistics/` as internal empirical audit records. |
