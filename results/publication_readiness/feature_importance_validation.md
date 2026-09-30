# Feature Importance & Ablation Forensic Validation

**Document Version:** 1.0  
**Research Topic:** Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data  
**Primary Horizon:** $H = 5$ Trading Days  
**Primary Model Architecture:** XGBoost GPU GBDT (30 Causal OHLCV Features)  
**Verification Target:** Empirical Origin and Methodological Limits of Feature Importance & Group Ablation Findings  

---

## 1. Executive Summary & Epistemological Clarification

Prior draft statements claimed that *"bar geometry and momentum provide real, causal short-term predictive signal accounting for >60% of feature importance."* 

This statement has been audited and found to conflate two distinct concepts:
1. **Temporal Causality $\neq$ Economic Causality:** While all 30 OHLCV features strictly satisfy temporal causality (invariance to future data perturbations at $t+1, t+5$), empirical feature attribution in machine learning models demonstrates **predictive statistical association**, not structural economic causation.
2. **Origin of the ">60%" Metric:** The figure does not originate from internal tree split counts or Gini/Gain impurity metrics. Rather, it derives from an out-of-time **leave-one-group-out ablation experiment** conducted on the locked test partition (`results/experiments/ablation_summary.json`).

---

## 2. Forensic Tracing of the Ablation Experiment

The ablation experiment evaluated five pre-specified thematic feature groups by training identical XGBoost models (500 trees, $\eta = 0.03$, max depth = 6) with each respective feature block omitted:

| Evaluated Feature Configuration | Omitted Feature Group | Number of Features | Test Mean Daily Rank IC | Absolute $\Delta$ vs Full Model | Relative Signal Loss (%) | Test IC $t$-statistic | Test IC $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Primary Model** | *None (All 30 Features)* | **30** | **+0.00587** | **0.00000** | **0.0%** | **1.06** | **0.2919** |
| `LEAVE_OUT_G1_MOMENTUM` | G1: Momentum (`ret_1d` to `ret_63d`) | 25 | +0.00175 | -0.00412 | **-70.2%** | 0.33 | 0.7409 |
| `LEAVE_OUT_G5_BAR_GEOMETRY` | G5: Bar Geometry (`shadows`, `pressure`, etc.)| 23 | +0.00207 | -0.00380 | **-64.7%** | 0.37 | 0.7118 |
| `LEAVE_OUT_G3_TREND` | G3: Moving Average / Trend Indicators | 24 | +0.00389 | -0.00198 | **-33.7%** | 0.71 | 0.4754 |
| `LEAVE_OUT_G4_VOLUME` | G4: Volume & Liquidity Dynamics | 24 | +0.00456 | -0.00131 | **-22.3%** | 0.87 | 0.3866 |
| `LEAVE_OUT_G2_VOLATILITY` | G2: Historical Volatility Estimators | 24 | +0.00499 | -0.00088 | **-15.0%** | 1.02 | 0.3096 |

### Rigorous Methodological Findings:
1. **The True Empirical Meaning:** Omitting the 5 momentum features reduces the out-of-time test Rank IC by **$70.2\%$** (from $0.00587$ down to $0.00175$). Omitting the 7 intraday bar geometry features reduces the test Rank IC by **$64.7\%$** (from $0.00587$ down to $0.00207$).
2. **What This Proves:** This proves that within the gradient-boosted decision tree architecture, momentum and bar geometry features supply the primary source of incremental rank correlation.
3. **What This Does NOT Prove:**
   - It does **not** prove that candlestick shadows or momentum "cause" future stock prices to move.
   - Because the baseline full model itself has a $t$-statistic of $1.06$ ($p = 0.2919$), the differences between the full model and ablated models are **statistically uncertain**.

---

## 3. Methodological Limitations of Tree-Based Feature Attributions

When discussing feature importance in the manuscript, the following limitations must be explicitly maintained:
1. **Collinearity Dilution:** Features within the momentum family (`ret_1d`, `ret_5d`, `ret_10d`) exhibit moderate cross-sectional correlation. In tree-based models, correlated predictors dilute each other's split frequency and gain attribution.
2. **Context-Specific Importance:** Feature importance measures how the tree partitioning algorithm partitioned the historical training observations (`2019-2024`); it does not represent an invariant law of market mechanics.
3. **Absence of Exogenous Identification:** No natural experiment, instrumental variable, or regression discontinuity was used. Therefore, all claims of "causal signal" are ungrounded and strictly forbidden.

---

## 4. Mandatory Manuscript Language

- **PROHIBITED:** *"Bar geometry and momentum provide real causal signal explaining stock returns."*
- **REQUIRED:** *"In leave-one-group-out ablation experiments on the out-of-time test partition, omitting trailing momentum or bar geometry features resulted in a $>60\%$ reduction in empirical Rank IC within the fitted gradient-boosted tree architecture. However, this represents observational predictive association rather than economic causation, and differences remain statistically uncertain given the modest baseline IC ($t = 1.06$)."*
