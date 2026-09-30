# Publication Figures Inventory & Classification

**Document:** Publication Figure Audit & Inventory  
**Classification Date:** 2026-09-29  
**Target Repository:** `e:\Stock_Predition\`  
**Target Output Directory:** `results/publication_figures/`  

---

## 1. Existing Figure Audit

The repository was systematically audited across `paper/`, `results/`, `results/model_enhancement/`, `results/publication_readiness/`, `results/accuracy_optimization/`, and `results/statistics/`.

### 1.1 Legacy Experiment Figures (`results/figures/`)

| Filename | Dimensions / Type | Generating Script / Phase | Current Role | Action Classification | Justification |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `fig_1_filtering_funnel.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 2) | Universe filtering funnel | **REPLACE** | Outdated formatting; replaced by publication-standard [fig02_universe_funnel.png](file:///e:/Stock_Predition/results/publication_figures/fig02_universe_funnel.png). |
| `fig_4_feature_correlation.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 3) | 30-feature correlation matrix | **REPLACE** | Replaced by publication-standard similarity correlation heatmap [fig09_similarity_heatmap.png](file:///e:/Stock_Predition/results/publication_figures/fig09_similarity_heatmap.png). |
| `fig_6_daily_ic_series.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 6) | Daily IC time series | **REPLACE** | Replaced by clean confirmatory comparison [fig05_rank_ic_comparison.png](file:///e:/Stock_Predition/results/publication_figures/fig05_rank_ic_comparison.png). |
| `fig_8_cumulative_trajectories.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 8) | Cumulative return curves | **REPLACE** | Superseded by variance reduction and transaction cost analyses. |
| `fig_9_recommendation_comparison.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 8) | Old 1,220 recommendation checks | **REPLACE** | Replaced by authoritative 6,100 portfolio figures: [fig11_variance_reduction.png](file:///e:/Stock_Predition/results/publication_figures/fig11_variance_reduction.png) and [fig12_transaction_cost.png](file:///e:/Stock_Predition/results/publication_figures/fig12_transaction_cost.png). |
| `fig_10_alpha_frontier.png` | PNG | `scripts/advanced_model_exploration.py` | Exploratory alpha sweep | **REMOVE** | Exploratory non-confirmatory plot; not cited in final manuscript. |
| `fig_11_model_comparison.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 6) | 30-feature model zoo | **REPLACE** | Replaced by authoritative 4-tier ablation [fig04_feature_ablation.png](file:///e:/Stock_Predition/results/publication_figures/fig04_feature_ablation.png) and [fig13_performance_summary.png](file:///e:/Stock_Predition/results/publication_figures/fig13_performance_summary.png). |
| `fig_12_horizon_decay.png` | PNG | `scripts/pipeline_orchestrator.py` (Phase 9) | Exploratory horizon decay | **REMOVE** | Exploratory post-hoc horizon plot; relegated to Section 8 text. |

---

### 1.2 Statistical Audit Diagnostic Figures (`results/statistics/figures/`)

| Filename | Type | Generating Script | Status | Action Classification | Justification |
| :--- | :---: | :--- | :--- | :---: | :--- |
| `fig01_return_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly in `results/statistics/` as internal empirical audit record. |
| `fig02_fwd_return_distribution.png`| PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig03_feature_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig04_correlation_heatmap.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **REPLACE** | Replaced in paper by [fig09_similarity_heatmap.png](file:///e:/Stock_Predition/results/publication_figures/fig09_similarity_heatmap.png). |
| `fig05_prediction_probability_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig06_calibration_curve.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **REPLACE** | Replaced by publication-grade Platt calibration reliability diagram [fig08_calibration.png](file:///e:/Stock_Predition/results/publication_figures/fig08_calibration.png). |
| `fig07_accuracy_vs_coverage.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **REPLACE** | Replaced by authoritative [fig06_selective_accuracy.png](file:///e:/Stock_Predition/results/publication_figures/fig06_selective_accuracy.png) and [fig07_up_precision.png](file:///e:/Stock_Predition/results/publication_figures/fig07_up_precision.png). |
| `fig08_rank_ic_over_time.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig09_rank_ic_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig10_residual_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig11_feature_importance.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig12_market_regime_performance.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig13_recommendation_return_distribution.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig14_cumulative_recommendation_return.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |
| `fig15_recommendation_drawdown.png` | PNG | `scripts/run_complete_statistical_audit.py` | Diagnostic | **KEEP** | Retained strictly as internal empirical audit record. |

---

### 1.3 Manuscript Compiled Artifacts (`paper/`)

| Filename | Type | Role | Action Classification | Justification |
| :--- | :---: | :--- | :---: | :--- |
| `research_paper.pdf` | PDF | Compiled 8-page paper | **KEEP** | Submission-ready compiled manuscript PDF. |

---

## 2. Summary of Actions

- **Total Existing Figures Audited:** 24
- **Figures Retained as Internal Diagnostics:** 12 (`results/statistics/figures/`)
- **Figures Replaced by Publication Set:** 8 (`results/figures/` and `results/statistics/figures/`)
- **Figures Removed / Marked Obsolete:** 2 (`fig_10_alpha_frontier.png`, `fig_12_horizon_decay.png`)
- **Compiled Paper Retained:** 1 (`paper/research_paper.pdf`)
- **Required New Publication Figures:** 13 (Figures 1 through 13 in both 300 DPI PNG and vector SVG/PDF formats)
