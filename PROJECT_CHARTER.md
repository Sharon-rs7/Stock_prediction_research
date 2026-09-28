# Project Charter: Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation

## 1. Executive Summary & Research Question
This research investigates the empirical viability of historical price/volume dynamics for both cross-sectional return forecasting and peer similarity modeling within US equity markets. Specifically, we address the primary research question:
> *"Can historical OHLCV behavior identify stocks with similar behavior, and does combining similarity information with future-return prediction improve Top-5 stock recommendation compared with similarity-only and prediction-only approaches?"*

Crucially, in adherence to strict scientific rigor, we do **not** assume:
- That machine learning models necessarily outperform naive or linear baselines.
- That behavioral similarity necessarily improves recommendation quality.
- That combined recommendation rules dominate individual prediction or similarity models.
- That one similarity lookback ($L=252$ vs. $L=504$) is universally superior.

All conclusions are established empirically via out-of-time evaluation and paired statistical hypothesis testing.

---

## 2. Locked Research Scope & Universe Definition
- **Data Source**: Hugging Face `AmirTrader/YahooFinance` (Pinned Commit: `c3c01ff2fc62e02c338d2e03bdfd71016da09701`).
- **Synchronized Observation Period**: 2019-09-26 $\rightarrow$ 2026-09-25 (1,759 synchronized trading days).
- **Exclusion of Alternative Data**: Strictly restricted to historical price, volume, and adjusted close series. No news sentiment, social media, FinBERT, fundamentals, or external analyst ratings.
- **Target Universe (Universe B — Liquid Core)**: Exactly 2,435 ordinary common equities selected via a five-gate filtering protocol:
  1. *Instrument Type*: Ordinary common stock candidates (excluding preferreds, warrants, rights, units).
  2. *Data Integrity*: Zero negative prices, zero invalid OHLC relations, strictly positive minimum close.
  3. *Trading Activity*: Zero-volume trading days $\le 1.0\%$.
  4. *Calendar Synchronization*: Exactly 1,759 balanced trading records over the 7-year window.
  5. *Liquidity Floor*: Median daily volume $\ge 100,000$ shares.

---

## 3. Empirical Design & Leakage Controls
- **Forecast Horizon**: Primary horizon $H=5$ trading days; robustness horizons $H \in \{1, 21\}$ trading days.
- **Primary Forecast Target**: Cross-sectional $z$-score of 5-day forward return:
  $$z(i, t, 5) = \frac{y(i, t, 5) - \mu_t(y(\cdot, t, 5))}{\sigma_t(y(\cdot, t, 5))}$$
- **Chronological Partitions & Purge Gaps**:
  - **Train**: 2019-09-26 $\rightarrow$ 2024-03-28 (followed by a 5-day purge gap).
  - **Validation**: 2024-04-08 $\rightarrow$ 2025-06-27 (followed by a 5-day purge gap).
  - **Out-of-Time Test**: 2025-07-08 $\rightarrow$ 2026-09-25.
- **Leakage Prevention**: All 30 OHLCV features computed using strictly causal backward-looking windows ($\le t$). Scalers and normalizers fitted exclusively on training partitions. Test data untouched until final out-of-time validation.

---

## 4. Methodological Matrix
1. **Feature Space**: 30 technical indicators spanning 5 structural categories:
   - Group 1: Momentum / Log Returns (5 features)
   - Group 2: Volatility / Tail Risk (6 features)
   - Group 3: Trend / Moving Averages (6 features)
   - Group 4: Volume / Liquidity (6 features)
   - Group 5: Bar Geometry (7 features)
2. **Predictive Models**: Zero Baseline, Historical Mean, Momentum Baseline, OLS Regression, Ridge Regression ($L_2$), Random Forest, and LightGBM GBDT.
3. **Similarity Engine**: Trailing Pearson Return Correlation, Factor Cosine Similarity, and Normalized Trajectory Proximity across $L \in \{252, 504\}$ trading days.
4. **Recommendation Schemes**:
   - **Method A (Prediction-Only)**: Cross-sectional top-5 based purely on model expected return score.
   - **Method B (Similarity-Only)**: Top-5 most behaviorally similar peer equities to the target stock.
   - **Method C (Combined Fusion)**: Pre-specified rank-based fusion of similarity and expected return score.
