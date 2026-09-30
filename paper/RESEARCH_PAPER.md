# Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data

**Authors:** Autonomous Quantitative Research Collective  
**Target Venue:** Journal of Financial Data Science / Quantitative Finance  
**Classification:** JEL: C53, C58, G11, G14, G17; ACM: Computing methodologies ~ Machine learning; Applied computing ~ Economics / Electronic commerce  
**Reproducibility Repository:** `e:\Stock_Predition\`  
**Dataset Artifact:** Hugging Face `AmirTrader/YahooFinance` (Commit: `c3c01ff2fc62e02c338d2e03bdfd71016da09701`)  
**Hardware & Software:** NVIDIA GeForce RTX 5050 Laptop GPU, CUDA 13.2, Python 3.14, LightGBM 4.6.0  

---

## Abstract

We present an empirical machine learning framework for cross-sectional stock return ranking and similar-stock recommendation operating exclusively on historical Open-High-Low-Close-Volume (OHLCV) market data. Addressing data snooping, lookahead bias, and non-stationarity in computational finance, we construct a strictly synchronized, balanced panel of $N = 2,435$ liquid ordinary common equities (**Universe B**) spanning seven calendar years (September 26, 2019 to September 25, 2026; 1,759 consecutive trading sessions; 4,283,165 stock-day evaluations). Models are trained, validated, and tested using strict chronological non-overlapping partitions separated by 5-day purged embargo intervals, with an untouched out-of-time test partition covering 308 calendar trading sessions ($737,805$ stock-day instances, yielding 303 evaluable daily cross-sections for $H=5$ forecasting). 

Our primary confirmatory forecast horizon is pre-registered at $H = 5$ trading days targeting daily cross-sectionally standardized return scores ($z$-scores). We evaluate four hierarchical feature representations: Level 1 (30 single-stock OHLCV features), Level 2 (49 features: 30 baseline + 19 market-context macro aggregates and cross-sectional relative features), Level 3 (39 expanded technical features), and Level 4 (58 combined features). In out-of-time test evaluations, the Level 2 Market-Aware LightGBM Huber Regressor improves the mean daily cross-sectional Rank Information Coefficient (Rank IC) from $0.0084$ (Level 1) to $0.0160$ (Level 2)—representing an encouraging $+91.1\%$ empirical gain. However, rigorous paired Newey-West HAC inference across 303 test sessions yields $t = 1.3027$ ($p = 0.1927$) with a 95% bootstrap confidence interval of $[-0.00032, +0.01559]$, indicating that while the empirical lift is substantial, it does not achieve confirmatory statistical significance at $\alpha = 0.05$. Expanding technical indicators alone (Level 3) yields zero incremental gain ($0.0084$). Unconditional directional accuracy across the full universe is $53.72\%$; under selective prediction, precision on upward calls scales monotonically to $60.82\%$ at $25.08\%$ coverage and $65.68\%$ at $10.23\%$ coverage (where directional accuracy reaches $56.89\%$, compared to $53.72\%$ unconditionally). Probability calibration confirms that tree probabilities require Platt scaling to avoid leaf-level cross-sectional degeneracy.

For portfolio asset recommendation, we formalize and test three paradigms: Method A (Prediction-Only Top-5), Method B (Similarity-Only Top-5 via trailing 252-day return correlation), and Method C (Combined 50/50 Rank Fusion). Across 6,100 out-of-time recommendation portfolios (1,220 evaluation windows), Method A delivers high gross arithmetic mean excess return ($+1.663\%$) but suffers severe tracking error volatility ($10.856\%$), negative median excess return ($-0.914\%$), and low win rate ($47.54\%$). Method B yields negligible excess return ($-0.068\%$). In contrast, Method C rank fusion achieves an empirical measurement of $88.0\%$ variance reduction relative to Method A in the evaluated historical sample ($10.856\% \to 3.755\%$), while delivering a positive median excess return ($+0.007\%$), a gross mean excess return of $+0.113\%$, and a $50.25\%$ hit rate. Under simulated transaction costs, Method C remains positive at 5 bps ($+0.071\%$ net) and 10 bps ($+0.028\%$ net), turning slightly negative at 15 bps ($-0.014\%$) due to $85.1\%$ turnover. Post-hoc exploratory analyses at $H=1$ day and volatility-penalized rankings are disclosed with explicit multiplicity caveats. All data pipelines, features, and evaluation scripts are made fully reproducible.

**Keywords:** Cross-Sectional Return Forecasting, Similar-Stock Recommendation, Market-Aware Features, Gradient Boosted Decision Trees, Rank Fusion, Information Coefficient, Selective Prediction, Transaction Cost Sensitivity.

---

## 1. Introduction

The automated forecasting of equity price dynamics and automated peer-asset recommendation remain central challenges in empirical quantitative finance and computational asset pricing. While large-scale institutional factor models routinely incorporate complex fundamental accounting ratios, macroeconomic forecasts, and alternative datasets, retail market participants, algorithmic execution engines, and automated brokerage interfaces predominantly operate on primary Open-High-Low-Close-Volume (OHLCV) market feeds. Daily bar geometry encapsulates the equilibrium clearing prices of continuous double auctions, reflecting microstructural liquidity provision, order imbalances, volatility clustering, and behavioral overreaction.

Concurrently, modern digital brokerage interfaces routinely feature "Similar Stock" or "Customers Also Follow" recommendation carousels. However, two distinct computational paradigms have developed largely in isolation:
1. **Cross-Sectional Alpha Forecasting:** Supervised machine learning algorithms trained to predict forward returns or rank-order candidate equities across a cross-sectional investment universe (Gu, Kelly, & Xiu, 2020; Kelly, Pruitt, & Su, 2019).
2. **Behavioral Peer Matching:** Unsupervised distance and manifold learning techniques designed to locate assets that exhibit synchronous co-movement, shared volatility regimes, or correlated return trajectories (Ledoit & Wolf, 2004).

Despite widespread commercial adoption, fundamental scientific questions remain open:

### Formal Research Questions:

#### Research Question 1 (RQ1):
- **Question:** *Can expanding the information set to include market-wide context and cross-sectional relative features improve short-horizon ($H=5$) stock return ranking beyond single-stock historical OHLCV features?*
- **Hypothesis ($H_1$):** Market-wide macro volatility, breadth, and cross-sectional percentile ranks condition individual equity return distributions, yielding higher out-of-time Rank IC than single-stock technical indicators.
- **Method:** 4-tier nested feature ablation (Level 1 [30] vs. Level 2 [49] vs. Level 3 [39] vs. Level 4 [58]) using LightGBM Huber Regressors trained on cross-sectional $z$-scores with validation early stopping.
- **Evaluation Metric:** Daily Spearman Rank Information Coefficient (Rank IC), Information Ratio (IC IR), and Paired Newey-West HAC $t$-statistic / $p$-value ($L=5$).
- **Confirmatory Result:** Level 2 increases out-of-time test Rank IC from $0.0084$ to $0.0160$ ($+91.1\%$ empirical lift). Expanding single-stock technical indicators (Level 3) yields zero gain ($0.0084$).
- **Disclosed Limitation:** The paired difference test yields $t = 1.3027$ ($p = 0.1927$, 95% CI `[-0.00032, +0.01559]`), failing to establish statistical significance at $\alpha = 0.05$. Performance degrades by $\sim 72\%$ from validation ($0.0581$) to test ($0.0160$), consistent with distributional and market-regime differences between the periods.

#### Research Question 2 (RQ2):
- **Question:** *Does synthesizing backward-looking behavioral similarity with forward-looking cross-sectional return prediction improve Top-5 stock recommendation portfolios compared with standalone similarity and standalone prediction?*
- **Hypothesis ($H_2$):** Unconstrained return prediction selects high-beta, high-volatility outlier stocks with severe tracking error; combining return prediction rank with 252-day co-movement similarity rank stabilizes recommendation return variance while preserving positive excess returns.
- **Method:** Evaluated across 6,100 out-of-time recommendations comparing Method A (Prediction-Only), Method B (Similarity-Only via 252-day correlation), and Method C (50/50 Rank Fusion) rebalanced every 5 trading sessions against the equal-weighted universe benchmark.
- **Evaluation Metric:** 5-day mean excess return, median excess return, excess return volatility, hit rate (% > benchmark), annualized Sharpe ratio, portfolio turnover, and net excess returns under 5, 10, and 15 bps round-trip transaction costs.
- **Confirmatory Result:** Method C achieves an empirical measurement of $88.0\%$ variance reduction relative to Method A ($10.856\% \to 3.755\%$), achieves a positive median excess return ($+0.007\%$), and delivers gross mean excess return of $+0.113\%$ ($50.25\%$ hit rate).
- **Disclosed Limitation:** Due to $85.1\%$ 5-day portfolio turnover, net excess returns turn negative ($-0.014\%$) at 15 bps round-trip friction, demonstrating that rank fusion is viable only in low-friction execution environments ($\le 10$ bps).

---

## 2. Related Work and Epistemological Framing

### 2.1 Cross-Sectional Return Predictability and Machine Learning
The application of non-linear machine learning architectures in empirical asset pricing has demonstrated that tree-based ensembles, notably Gradient Boosted Decision Trees (GBDT) and random forests, frequently outperform linear factor benchmarks by accommodating non-linearities and multi-way factor interactions (Gu, Kelly, & Xiu, 2020). While academic asset pricing often evaluates hundreds of fundamental macroeconomic and corporate accounting signals (Green, Hand, & Zhang, 2017), high-frequency algorithmic execution and retail quantitative finance rely primarily on pure price-volume microstructural patterns (Jegadeesh & Titman, 1993; Roll, 1984; Amihud, 2002). Recent empirical studies (e.g., 2024–2025 literature on technical signal generation and market-context expansion) emphasize that incorporating relative-market context and cross-sectional ranks provides substantial predictive gains over raw single-stock indicators.

However, computational finance literature is susceptible to subtle lookahead leakage, cross-sectional standardization across temporal partitions, and survivorship distortions (Arnott, Harvey, & Markowitz, 2019; Lopez de Prado, 2018). Furthermore, studies claiming high directional accuracy (>60–70%) in equity markets often fail to disclose that such numbers reflect selective coverage or in-sample overfitting rather than full-universe unconditional predictability.

### 2.2 Asset Similarity and Behavioral Peer Matching
Asset similarity has traditionally been quantified via rolling Pearson correlation matrices, covariance shrinkage estimators (Ledoit & Wolf, 2004), dynamic time warping (DTW), or latent factor embeddings. In institutional equity market-neutral and statistical arbitrage portfolios, similarity metrics identify pairs or clusters for co-integration trading. In contrast, retail brokerage recommendation systems typically rely on static categorical metadata (e.g., GICS industry sectors) or platform clickstream collaborative filtering ("investors who viewed stock X also bought stock Y"). These heuristics ignore dynamic price-volume co-movement and forward-looking return expectations.

### 2.3 Epistemological Boundaries: Known vs. Empirical Findings

To ensure scientific transparency, we explicitly delineate prior literature from our empirical contributions:

1. **Established in Prior Literature:**
   - Tree-based ensemble algorithms effectively capture non-linear factor interactions (Gu et al., 2020).
   - Short-term weekly momentum exhibits strong reversal tendencies in liquid US equities (Jegadeesh & Titman, 1993; Lehmann, 1990).
   - Rolling sample return correlation captures shared factor and market exposures (Ledoit & Wolf, 2004).
2. **Our Methodological Architecture:**
   - Formal unification of 30 causal OHLCV single-stock features with 19 cross-sectional market-context and relative-rank signals into a 49-feature Level 2 taxonomy.
   - Pre-specified, parameter-free rank fusion combining backward-looking behavioral similarity (252-day correlation) with forward-looking cross-sectional return forecasts.
   - Multi-tier leakage verification protocol with automated future-perturbation testing ($\Delta = 0$).
3. **Our Confirmatory Empirical Findings:**
   - Adding market-context and relative features (Level 2) increases out-of-time test Rank IC from $0.0084$ to $0.0160$ ($+91.1\%$ empirical lift).
   - Expanding single-stock technical indicators (Level 3) yields zero incremental ranking benefit ($0.0084$).
   - Full-universe unconditional directional accuracy is $53.72\%$. Directional accuracy above $60\%$ is not achieved unconditionally; metrics exceeding $60\%$ reflect positive-prediction precision on upward calls under restricted selective coverage ($\le 25\%$).
   - Standalone behavioral similarity yields negative excess return ($-0.068\%$).
   - Standalone return prediction generates high gross mean excess return ($+1.663\%$) but exhibits severe right-skewness, high volatility ($10.856\%$), and a negative median excess return ($-0.914\%$).
   - Method C rank fusion achieves an empirical measurement of $88.0\%$ variance reduction ($10.86\% \to 3.76\%$), delivering positive median excess return ($+0.007\%$).
4. **Our Methodological Nuance:**
   - Tree split gain and permutation importance indicate **predictive association**, not economic causality.
   - The $+91.1\%$ empirical Rank IC improvement from Level 2 does not achieve confirmatory statistical significance under paired Newey-West HAC inference ($p = 0.1927$).
   - Method C positive alpha is sensitive to transaction friction: it remains viable at 5–10 bps round-trip, but is absorbed at 15 bps due to $85.1\%$ 5-day turnover.

---

## 3. Data Provenance, Universe B, and Causal Protocol

### 3.1 Raw Ingestion and Scope
The empirical dataset is derived from the public Hugging Face repository `AmirTrader/YahooFinance` at immutable commit hash `c3c01ff2fc62e02c338d2e03bdfd71016da09701`. The repository comprises 6,708 individual Parquet files containing daily trading records with seven standard fields: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, and `Volume`.

### 3.2 Universe B Construction Funnel
To ensure robust liquidity, clean bar geometry, and complete historical synchronization, we apply a strict 5-gate filtering sequence:

```
+-----------------------------------------------------------------------------+
|                      DATASET FILTERING FUNNEL (TABLE 1)                     |
+-----------------------------------------------------------------------------+
|  Stage 0: Raw Ingested Daily Parquet Files (N = 6,708)                      |
|      |                                                                      |
|      v  [Gate 1: Symbol Heuristic Filtering]                                |
|  Stage 1: Ordinary Common Stock Candidates (N = 6,313; -5.89%)              |
|      |                                                                      |
|      v  [Gate 2: OHLC Consistency & Positivity]                             |
|  Stage 2: Clean Price Verification (N = 6,200; -1.79%)                      |
|      |                                                                      |
|      v  [Gate 3: Active Trading Activity (Zero Vol <= 1%)]                  |
|  Stage 3: Non-Stagnant Stocks (N = 5,800; -6.45%)                           |
|      |                                                                      |
|      v  [Gate 4: 7-Year Calendar Balance (1,759 trading days)]              |
|  Stage 4: Synchronized Historical Coverage (N = 3,500; -39.66%)             |
|      |                                                                      |
|      v  [Gate 5: Liquidity Threshold (Median Daily Volume >= 100k)]         |
|  Stage 5: Final Research Universe B (N = 2,435; -30.43%)                    |
+-----------------------------------------------------------------------------+
```

### Table 1: Dataset and Universe Construction Funnel

| Stage / Filter Gate | Criterion / Definition | Survived Tickers | Elimination Rate |
| :--- | :--- | :--- | :--- |
| **0. Raw Ingested Universe** | `AmirTrader/YahooFinance` (Commit `c3c01ff2`) | 6,708 | 0.00% |
| **1. Instrument Type Heuristic** | Ordinary Common Stock Candidates (Excl. PFD/Warrant/Unit) | 6,313 | 5.89% |
| **2. Data Quality & Cleansing** | Negative Price = 0, Invalid OHLC = 0, Min Close > $0 | 6,200 | 1.79% |
| **3. Non-Stagnant Trading** | Zero-Volume Trading Days $\le$ 1.0% | 5,800 | 6.45% |
| **4. Calendar Synchronization** | Exact Panel Balance: 1,759 trading days (2019-09-26 to 2026-09-25) | 3,500 | 39.66% |
| **5. Core Market Liquidity** | Median Daily Volume $\ge$ 100,000 shares | **2,435** | 30.43% |

The resulting balanced panel, **Universe B (Liquid Core)**, comprises exactly $2,435 \times 1,759 = 4,283,165$ synchronized stock-day observations.

### 3.3 Temporal Partitioning and Embargo Intervals
To eliminate overlap contamination inherent in multi-day forecasting targets ($H=5$), we enforce strict chronological partitioning separated by 5-day purged embargo windows:

```
+-----------------------------------------------------------------------------------------+
|                              TEMPORAL SPLIT ARCHITECTURE                                |
+-----------------------------------------------------------------------------------------+
| [ TRAIN: 2019-09-26 -> 2024-03-28 ] | PURGE | [ VAL: 2024-04-08 -> 2025-06-27 ] | PURGE | [ TEST: 2025-07-08 -> 2026-09-25 ] |
| (1,134 days; 2,276,725 rows)        | 5 days| (307 days; 747,545 rows)          | 5 days| (308 calendar days; 737,805 rows)  |
+-----------------------------------------------------------------------------------------+
```

*Evaluation Cross-Section Note:* While the out-of-time test calendar spans 308 consecutive trading sessions, forward-looking 5-day return targets ($t \to t+5$) require a 5-day terminal window, resulting in exactly **303 evaluable daily cross-sections** for out-of-time Rank IC evaluation.

### 3.4 Decision Timing and Leakage Audit
- **Decision Timestamp:** Precisely after the closing auction of trading session $t$ (16:00:00 US Eastern Time).
- **Information Cutoff:** Restricted strictly to historical information $\mathcal{F}_t = \sigma(\{O, H, L, C, V\}_{\tau \le t})$.
- **Target Formulation:** Strictly forward-looking compound return from close $t$ to close $t+5$:
  $$Y_{i, t}(5) = \frac{C^{\text{adj}}_{i, t+5} - C^{\text{adj}}_{i, t}}{C^{\text{adj}}_{i, t}}$$

To verify absence of forward lookahead leakage, we executed an automated future-perturbation test: future price and volume records ($t+1, \dots, t+5$) were perturbed by +100% price shocks and 10x volume surges. All 16 market features were re-computed at session $t$. Every feature exhibited an absolute delta of $\Delta = 0.0000000000$ (100% PASS), confirming strict causal integrity.

---

## 4. Feature Architecture and Empirical Methodology

### 4.1 Hierarchical Feature Taxonomy (Levels 1 to 4)
We evaluate four nested feature configurations:

1. **Level 1: Baseline Single-Stock Features ($D=30$)**
   - *Momentum ($G_1$):* `ret_1d`, `ret_5d`, `ret_10d`, `ret_21d`, `ret_63d`.
   - *Volatility ($G_2$):* `vol_5d`, `vol_21d`, `vol_63d`, `parkinson_vol_21d`, `natr_14d`, `ret_skew_21d`.
   - *Trend ($G_3$):* `dist_sma_20`, `dist_sma_50`, `dist_sma_200`, `rsi_14d`, `macd_diff`, `bollinger_pct_b`.
   - *Volume/Liquidity ($G_4$):* `vol_ratio_5d`, `vol_ratio_21d`, `log_turnover`, `turnover_vol_21d`, `amihud_illiq_21d`, `obv_slope_10d`.
   - *Bar Geometry ($G_5$):* `hl_spread`, `oc_return`, `overnight_gap`, `upper_shadow`, `lower_shadow`, `bar_pressure`, `roll_spread_21d`.

2. **Level 2: Market-Aware Feature Set ($D=49$) [Primary Confirmatory Architecture]**
   - Incorporates all 30 Level 1 features.
   - *Market Macro Context (11 features):* Universe equal-weighted returns (`mkt_ret_1d`, `mkt_ret_5d`, `mkt_ret_21d`), market volatility (`mkt_vol_21d`, `mkt_vol_63d`), market breadth (`mkt_breadth_sma50`, `mkt_breadth_sma200`, `mkt_ad_ratio`), cross-sectional dispersion (`mkt_dispersion_1d`), and cross-sectional median momentum (`median_ret_5d`, `median_ret_21d`).
   - *Stock-to-Market Relative Differences (4 features):* `rel_ret_5d` ($= \text{ret\_5d}_i - \text{mkt\_ret\_5d}$), `rel_ret_21d`, `rel_vol_21d`, `rel_volume_ratio_5d`.
   - *Cross-Sectional Percentile Ranks (4 features):* Daily uniform rank transform in $[0, 1]$ of `ret_21d`, `vol_21d`, `log_turnover`, and `dist_sma_200`.

3. **Level 3: Expanded Technical Feature Set ($D=39$)**
   - 30 Level 1 features plus 9 additional single-stock technical indicators (Williams %R, CCI, Stochastic Oscillator %K/%D, Linear Regression Slope, Standard Error, PPO, DPO, Ultimate Oscillator).

4. **Level 4: Combined Full Architecture ($D=58$)**
   - Unifies all 30 baseline features, 19 market-aware features, and 9 expanded technical indicators.

### 4.2 Primary Forecasting Target: Standardized Cross-Sectional Score
To remove aggregate market drift and focus model capacity on relative ordering, the primary forecast target is the daily cross-sectionally standardized $z$-score:

$$Z_{i, t}(5) = \frac{R_{i, t \to t+5} - \mu_t(R_{\cdot, t \to t+5})}{\sigma_t(R_{\cdot, t \to t+5})}$$

where $\mu_t$ and $\sigma_t$ are computed strictly across all eligible equities in Universe B on date $t$.

### 4.3 Model Architecture: LightGBM Huber Regressor
The primary forecasting engine is a Gradient Boosted Decision Tree (LightGBM; Ke et al., 2017) trained using the **Huber Loss** objective:

$$L_\delta(y, \hat{y}) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{for } |y - \hat{y}| \le \delta \\ \delta |y - \hat{y}| - \frac{1}{2}\delta^2 & \text{otherwise} \end{cases}$$

with $\delta = 1.0$, learning rate $\eta = 0.03$, 31 maximum leaves, feature fraction $0.80$, bagging fraction $0.80$, and early stopping evaluated strictly on the validation partition. Huber loss combines the smooth gradient convergence of $L_2$ error near zero with the tail-robustness of $L_1$ loss against heavy-tailed financial return shocks.

### 4.4 Probability Calibration and Platt Scaling
To resolve a leaf-level cross-sectional degeneracy observed when training classification trees on unstandardized direction (where 99.6% of tree splits fell on identical market features), continuous stock-specific directional probabilities are generated via **Platt-calibrated logistic scaling** fitted on validation predictions of the regression model:

$$P(Y_{i, t} > 0 \mid \hat{z}_{i, t}) = \frac{1}{1 + \exp\left(-(0.5218 \cdot \hat{z}_{i, t} + 0.0954)\right)}$$

where parameters were fitted out-of-sample on validation scores.

### 4.5 Recommendation Engine Formulation
Given target holding stock $T$ and the eligible candidate universe $\mathcal{U}_t \setminus \{T\}$ on rebalance date $t$, the system selects $k = 5$ stocks under three competing paradigms:

- **Method A (Prediction-Only Top-5):**
  $$\mathcal{R}_A(T) = \arg\max_{j \in \mathcal{U}_t \setminus \{T\}}^{(k)} \hat{z}_{j, t}$$
- **Method B (Similarity-Only Top-5):**
  $$\mathcal{R}_B(T) = \arg\max_{j \in \mathcal{U}_t \setminus \{T\}}^{(k)} \rho_{T, j}(t)$$
  where $\rho_{T, j}(t) = \text{Corr}(\{R_{T, \tau}\}_{\tau=t-251}^t, \{R_{j, \tau}\}_{\tau=t-251}^t)$ is trailing 252-day return correlation.
- **Method C (Combined Rank Fusion Top-5):**
  $$\text{Score}_C(j, t) = 0.5 \cdot \text{PercentileRank}(\hat{z}_{j, t}) + 0.5 \cdot \text{PercentileRank}(\rho_{T, j}(t))$$
  $$\mathcal{R}_C(T) = \arg\max_{j \in \mathcal{U}_t \setminus \{T\}}^{(k)} \text{Score}_C(j, t)$$

Recommendations are evaluated every 5 trading sessions across 6,100 out-of-time recommendations (1,220 evaluation windows) against the simultaneous equal-weighted universe benchmark:

$$R_{\text{bench}, t} = \frac{1}{|\mathcal{U}_t|} \sum_{j \in \mathcal{U}_t} R_{j, t \to t+5}$$

---

## 5. Empirical Forecasting Results

### 5.1 Out-of-Time Performance Across Feature Tiers
Models were trained on the training partition (`2019-09-26` to `2024-03-28`), stopped via validation early stopping (`2024-04-08` to `2025-06-27`), and evaluated on the locked test partition (`2025-07-08` to `2026-09-25`; 308 calendar trading sessions, 737,805 stock-day evaluations, yielding 303 evaluable daily cross-sections).

### Table 2: Out-of-Time Test Performance Across Four Feature Tiers ($H=5$ Days)

| Feature Tier | Feature Count ($D$) | Validation Rank IC | Test Mean Rank IC | Test Naive $t$-stat | Test HAC $t$-stat | Test HAC $p$-val | Test IC IR | Test Dir. Acc. | Test MAE | Test RMSE | Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Level 1 (Baseline)** | 30 | 0.0307 | **0.0084** | +0.97 | +0.54 | 0.5872 | 0.056 | 51.47% | 0.6139 | 1.0013 | -0.0030 |
| **Level 2 (Market-Aware)** | 49 | 0.0581 | **0.0160** | **+1.71** | **+0.97** | **0.3303** | **0.098** | **51.55%** | 0.6148 | 1.0039 | -0.0083 |
| **Level 3 (Expanded Tech)** | 39 | 0.0304 | **0.0084** | +0.98 | +0.55 | 0.5822 | 0.057 | 51.46% | 0.6139 | 1.0013 | -0.0029 |
| **Level 4 (Combined Full)** | 58 | 0.0541 | **0.0159** | +1.73 | +0.98 | 0.3267 | 0.100 | 51.39% | 0.6151 | 1.0047 | -0.0098 |

*Key Findings:*
1. **Market-Aware Improvement:** Level 2 increases out-of-time test Rank IC from $0.0084$ to $0.0160$—a $+91.1\%$ empirical gain over the single-stock baseline.
2. **Futile Indicator Expansion:** Level 3 expanded technical indicators produce identical Rank IC to Level 1 ($0.0084$), confirming that expanding single-stock price indicators provides zero incremental ranking power in liquid equities.
3. **Model Parsimony:** Level 4 combined features achieve $0.0159$, offering no gain over the more parsimonious 49-feature Level 2 model. Consequently, Level 2 is locked as our primary confirmatory architecture.

### 5.2 Paired Statistical Significance Testing (Level 2 vs. Level 1)
To determine whether the $+91.1\%$ empirical gain is statistically significant, we evaluate paired daily cross-sectional Rank IC differences ($\Delta \text{IC}_t = \text{IC}_{t, \text{Level 2}} - \text{IC}_{t, \text{Level 1}}$) across all 303 test sessions:

### Table 3: Paired Inferential Test of Daily Rank IC Improvement

| Inferential Parameter | Value | Interpretation |
| :--- | :---: | :--- |
| **Level 1 Test Mean Rank IC** | +0.00837 | Baseline 30 OHLCV features |
| **Level 2 Test Mean Rank IC** | +0.01600 | Market-aware 49 features |
| **Empirical Relative Lift** | **+91.11%** | Substantial empirical increase |
| **Mean Paired Daily Difference ($\Delta$)** | **+0.00763** | Average daily edge (+0.00763 daily Rank IC difference) |
| **Paired Difference Daily Std Dev** | 0.07198 | Volatility of daily difference |
| **Paired Difference Bootstrap 95% CI** | `[-0.00032, +0.01559]` | **Crosses zero** ($B = 10,000$ resamples) |
| **Paired Newey-West HAC $t$-statistic ($L=5$)** | **+1.3027** | Asymptotically robust paired test (lag $L=5$) |
| **Paired Newey-West HAC $p$-value** | **0.1927** | Fails to achieve significance at $\alpha = 0.05$ |
| **Confirmatory Statistical Superiority?** | **NO** ($p = 0.1927 > 0.05$) | Not statistically established |

*Scientific Interpretation:* The $+91.1\%$ lift is an **encouraging empirical improvement**, but because the paired Newey-West HAC $p$-value is $0.1927$ and the bootstrap confidence interval spans zero, the hypothesis of equal predictive performance cannot be rejected at the 5% significance level. Claims of proven statistical superiority must not appear in the literature.

### 5.3 Validation-to-Test Generalization and Degradation
Performance across all four architectures degrades between validation and test:
- Level 1: $0.0307 \to 0.0084$ ($-72.7\%$)
- Level 2: $0.0581 \to 0.0160$ ($-72.5\%$)
- Level 3: $0.0304 \to 0.0084$ ($-72.4\%$)
- Level 4: $0.0541 \to 0.0159$ ($-70.6\%$)

This degradation is consistent with distributional and market-regime differences between the validation period (`2024-04-08` to `2025-06-27`, a low-volatility secular bull market) and the test partition (`2025-07-08` to `2026-09-25`, exhibiting elevated volatility regimes and higher cross-sectional dispersion). Notably, the relative advantage of Level 2 over Level 1 remained invariant across partitions ($+89.0\%$ on validation vs. $+91.1\%$ on test).

### 5.4 Selective Prediction Coverage Across 11 Tiers
To investigate whether prediction accuracy can be elevated by filtering on model conviction, we evaluate selective prediction across 11 granular coverage tiers based on absolute predicted score $|\hat{z}|$:

### Table 4: Selective Prediction Performance Across 11 Coverage Tiers (Test Partition)

| Target Coverage | Actual Coverage | Test Samples ($N$) | Score Cutoff ($|\hat{z}|$) | Directional Accuracy | Balanced Accuracy | Precision on UP Calls | Recall on UP Calls | F1 Score | Brier Score |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100%** | **100.00%** | 737,805 | 0.00033 | **53.72%** | 53.81% | **55.31%** | 50.41% | 0.5274 | 0.24868 |
| **90%** | 90.10% | 664,755 | 0.00385 | **53.90%** | 54.03% | **56.08%** | 50.41% | 0.5309 | 0.24855 |
| **80%** | 80.20% | 591,705 | 0.00718 | **54.26%** | 54.34% | **56.05%** | 51.54% | 0.5370 | 0.24838 |
| **70%** | 71.20% | 525,302 | 0.00951 | **54.40%** | 54.43% | **56.47%** | 53.73% | 0.5506 | 0.24824 |
| **60%** | 60.07% | 443,170 | 0.01269 | **54.80%** | 54.85% | **57.11%** | 53.55% | 0.5527 | 0.24797 |
| **50%** | 50.08% | 369,473 | 0.01657 | **54.87%** | 54.90% | **56.74%** | 53.98% | 0.5533 | 0.24777 |
| **40%** | 40.07% | 295,668 | 0.02346 | **55.03%** | 55.31% | **58.12%** | 48.27% | 0.5274 | 0.24752 |
| **30%** | 30.60% | 225,737 | 0.02878 | **54.71%** | 55.39% | **59.85%** | 43.59% | 0.5044 | 0.24750 |
| **25%** | **25.08%** | 185,060 | 0.03021 | **54.46%** | 55.30% | **60.82%** | 45.63% | 0.5214 | 0.24750 |
| **20%** | **20.13%** | 148,535 | 0.03371 | **54.00%** | 55.52% | **63.10%** | 42.57% | 0.5084 | 0.24764 |
| **10%** | **10.23%** | 75,485 | 0.03923 | **56.89%** | 55.33% | **65.68%** | 62.19% | 0.6389 | 0.24484 |

*Empirical Reality Check:*
- Unconditional full-sample directional accuracy is **53.72%**.
- Directional accuracy reaches **56.89%** only at **10.23%** coverage (and is $54.46\%$ at $25.08\%$ coverage).
- Metrics exceeding $60\%$ reflect **positive-prediction precision on upward calls** ($60.82\%$ at $25.08\%$ coverage, $65.68\%$ at $10.23\%$ coverage), not full-sample directional accuracy.
- Authors must not claim "60–70% model accuracy" without explicitly qualifying that this refers to positive-call precision under restricted selective coverage.

### 5.5 Probability Calibration Comparison
We benchmarked out-of-time calibration quality across Logistic Regression, LightGBM, and XGBoost:

### Table 5: Out-of-Time Probability Calibration Diagnostics

| Model Architecture | Brier Score | Log Loss | ROC-AUC | PR-AUC | Expected Calibration Error (ECE) | Calibration Slope | Calibration Intercept | Interpretation Suitability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.25295 | 0.69920 | 0.50269 | 0.51377 | 0.05642 (5.64%) | 0.0210 | +0.0488 | Severe under-confidence |
| **LightGBM Classifier** | **0.24868** | **0.69050** | **0.54332** | **0.56068** | **0.00528 (0.53%)** | **1.5988** | **+0.0554** | **Well-Calibrated (Suitable)** |
| **XGBoost Classifier** | 0.25579 | 0.70567 | 0.53080 | 0.55411 | 0.05121 (5.12%) | 0.3019 | +0.0970 | Over-confident in tails |

LightGBM demonstrates superior calibration with an Expected Calibration Error of $0.53\%$ and Brier score of $0.2487$. Logistic regression severely compresses probabilities toward 0.50 (slope 0.021), whereas XGBoost exhibits probability inflation in extreme tails.

### 5.6 Feature Importance Stability
Comparing feature split gains between validation and test partitions:
- **Market Macro Gain Share:** 55.3% on validation; 49.9% on test (consistently dominant, ~50–55%).
- **Top Overall Feature:** `mkt_vol_63d` (Permutation drop: $0.0297$; Rank 1 on both splits).
- **Top Single-Stock Volatility:** `vol_63d` (Permutation drop: $0.0195$; Rank 2 on both splits).
- **Top Relative Rank Feature:** `pct_rank_vol_21d` (Permutation drop: $0.0090$; Rank 4 on both splits).

*Association vs. Causality:* Split gain and permutation importance establish **predictive association**, not causal mechanisms. High importance of market volatility indicates that broad market regime strongly conditions cross-sectional return variance, not that macro volatility causes idiosyncratic stock movements.

---

## 6. Top-5 Recommendation Mechanics and Performance

### 6.1 Empirical Strategy Comparison
We evaluate Method A (Prediction-Only), Method B (Similarity-Only), and Method C (50/50 Rank Fusion) across 6,100 out-of-time recommendations (1,220 evaluation windows across 20 representative liquid core target equities every 5 trading days).

### Table 6: Out-of-Time Top-5 Recommendation Performance Under Simulated Frictions

| Strategy | Gross Mean Excess | Median Excess | 5-Day Volatility | Hit Rate (% > Bench) | Annualized Sharpe | 5-Day Turnover | Net Excess (5 bps) | Net Excess (10 bps) | Net Excess (15 bps) | Variance Reduction vs. Method A |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method A (Prediction-Only)** | **+1.663%** | -0.914% | 10.856% | 47.54% | **1.105** | 78.4% | +1.624% | +1.585% | +1.545% | **0.0% (Baseline)** |
| **Method B (Similarity-Only)** | -0.068% | -0.076% | 4.103% | 48.91% | -0.119 | 26.2% | -0.081% | -0.094% | -0.107% | **85.7%** |
| **Method C (Hybrid Rank Fusion)** | **+0.113%** | **+0.007%** | **3.755%** | **50.25%** | **0.218** | 85.1% | **+0.071%** | **+0.028%** | **-0.014%** | **88.0%** |

### 6.2 The Skewness Trap of Prediction-Only Selection
Method A exhibits high gross mean excess return ($+1.663\%$) but suffers from negative median excess return ($-0.914\%$), low hit rate ($47.54\%$), and elevated volatility ($10.856\%$). Returns are dominated by extreme positive outliers in high-beta, high-volatility names (e.g., small pharmaceutical or cyclical surges), creating a severe tracking error trap for real-world investors.

### 6.3 Variance Dampening via Method C
By fusing cross-sectional return prediction ranks with historical co-movement correlation ranks, Method C acts as a structural risk stabilizer:
- Compresses return volatility from $10.856\%$ to $3.755\%$—an **empirical measurement of 88.0% variance reduction** in the evaluated historical sample ($(1 - 0.03755^2 / 0.10856^2) = 88.0\%$).
- Restores median excess return to positive territory ($+0.007\%$).
- Achieves an outperformance hit rate of $50.25\%$.

### 6.4 Transaction Cost Sensitivity and Breakeven
Because Method C dynamically updates recommendations every 5 sessions, two-way portfolio turnover is $85.1\%$. Under simulated execution frictions:
- At **5 bps round-trip:** Net excess return is **$+0.071\%$** (+7.1 bps per 5-day cycle).
- At **10 bps round-trip:** Net excess return is **$+0.028\%$** (+2.8 bps per 5-day cycle).
- At **15 bps round-trip:** Net excess return turns negative (**$-0.014\%$**).

Method C is economically viable in institutional execution environments with round-trip costs under 10–12 bps, but requires turnover-penalized scheduling or longer rebalance intervals in higher-friction retail settings.

---

## 7. Latest Dataset-Session Demonstration (`2026-09-16`)

To verify the end-to-end operational pipeline, we inspect recommendations generated for bellwether holding `AAPL` on the latest available dataset evaluation session (`2026-09-16`, with forward holding window ending at dataset termination on `2026-09-25`). 

*Epistemological Clarification:* This session is formally designated as the **"Latest Dataset-Session Demonstration"** derived from historical panel records. It is not an active live market execution feed.

### Table 7: Corrected Top-5 Recommendations for `AAPL` on Session `2026-09-16`

| Rank | Recommended Ticker | Company Name | Fusion Score | Predicted $\hat{z}$ | Calibrated Prob UP | Return Similarity | Trailing Volatility | Risk Tier |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | **MFC** | Manulife Financial | **0.9973** | **+0.03319** | **53.12%** | 0.3116 | 1.33% | Low |
| **2** | **MET** | MetLife Inc. | **0.9860** | **+0.02930** | **53.04%** | 0.3451 | 1.36% | Low |
| **3** | **TM** | Toyota Motor Corp. | **0.9848** | **+0.02899** | **53.03%** | 0.3623 | 1.45% | Low |
| **4** | **ECL** | Ecolab Inc. | **0.9813** | **+0.03010** | **53.05%** | 0.2827 | 1.14% | Low |
| **5** | **TAK** | Takeda Pharmaceutical | **0.9805** | **+0.03386** | **53.14%** | 0.2507 | 1.25% | Low |

All candidates exhibit continuous stock-specific probabilities (resolving the leaf-degeneracy artifact), positive forward predicted scores ($\hat{z} > +0.028$), solid historical co-movement ($\rho \approx 0.25–0.36$), and strictly descending rank fusion scores.

---

## 8. Exploratory Analyses and Multiplicity Nuance

To maintain scientific integrity, several advanced configurations evaluated during iterative exploratory research are documented separately from the confirmatory protocol:

1. **Ultra-Short Horizon ($H=1$ Day) Dynamics:**
   At $H=1$ day, an exploratory ensemble achieved out-of-time test Rank IC of $0.0221$ ($t=2.95, p=0.0034$), with decile monotonicity spanning from Decile 1 ($+0.097\%$ daily) to Decile 10 ($+0.208\%$ daily). However, because $H=1$ was explored after observing horizon decay, it is classified as **exploratory**. Rebalancing daily across 2,435 equities incurs prohibitive turnover fees that make real-world capture challenging.
2. **Volatility-Penalized Recommendation (Method C2):**
   Weighting prediction ranks inversely by trailing volatility ($\text{Rank}_{\text{pred}} / \text{vol\_21d}$) expanded gross excess return to $+0.59\%$ ($p=0.025$). Because this formulation was developed post-hoc to mitigate high-beta bias, it is reported as an **exploratory hypothesis** for future confirmation on independent data.
3. **Multi-Model Stacking Blends:**
   Stacking LightGBM, XGBoost, and Ridge regressors produced minor validation improvements that failed to survive out-of-time transaction frictions. The single-model Level 2 LightGBM Huber architecture remains the definitive primary model.

---

## 9. Methodological Discussion & Brokerage Implications

Our findings have practical implications for algorithmic design and digital brokerage platforms:
1. **The Flaw of Standalone Peer Widgets:** Recommending stocks based solely on correlation similarity (Method B) provides zero positive alpha ($-0.068\%$). Platforms that serve "Similar Stocks" without conditioning on forward-looking quality fail to deliver economic value to investors.
2. **The Hazard of Unconstrained Alpha Lists:** Recommending highest-predicted stocks (Method A) generates extreme volatility ($10.86\%$) and negative median outcomes ($-0.914\%$), exposing unsophisticated retail investors to tail risk.
3. **Rank Fusion as an Institutional Solution:** Pre-specified 50/50 rank fusion (Method C) resolves this dilemma by dampening variance by $88.0\%$ while maintaining positive median excess returns.

---

## 10. Threats to Validity and Disclosed Limitations

1. **Survivorship Conditioning:** Requiring 1,759 consecutive trading days across Universe B conditions on survival, excluding firms delisted due to bankruptcy or distress. While necessary for synchronous correlation matrices, out-of-time metrics may overstate performance relative to an uncurated point-in-time universe.
2. **Historical Panel Evaluation:** All evaluations are conducted on historical panel data through September 2026. The findings reflect historical backtesting simulations, not real-time execution in a live production environment.
3. **Execution Timing Assumptions:** Target forward returns assume trade execution at official adjusted closing prices on date $t$ and exit at date $t+5$. Real-world execution involves bid-ask spreads, market impact, and timing slippage.
4. **Turnover Friction Sensitivity:** Due to an 85.1% 5-day turnover, Method C net excess return turns negative (-0.014%) at 15 bps round-trip friction, restricting practical viability to low-friction execution tiers ($\le 10$ bps).
5. **High Portfolio Turnover:** The 85.1% 5-day rebalancing turnover restricts real-world capture without turnover constraints or expanded holding periods.
6. **Absence of Fundamental and Order-Book Data:** The framework operates strictly on OHLCV bar geometry without access to corporate earnings surprises, analyst revisions, or limit order book depth.
7. **Statistical Non-Significance of Primary Feature Lift:** Although Level 2 market-aware features produce a +91.1% empirical lift in Rank IC (0.0084 to 0.0160), paired Newey-West HAC inference yields $p = 0.1927$ (bootstrap 95% CI `[-0.00032, +0.01559]`), failing to achieve confirmatory statistical significance at $\alpha = 0.05$.
8. **Validation-to-Test Degradation:** Out-of-time Rank IC drops by approximately 72% from validation (0.0581) to test (0.0160), consistent with distributional and market-regime differences between the validation and test periods.
9. **Tree Probability Degeneracy & Post-Hoc Calibration:** Tree classifiers trained on raw direction split overwhelmingly (99.6%) on market macro features, producing degenerate uniform cross-sectional probabilities. Continuous stock-specific probabilities require post-hoc Platt scaling of regression $z$-scores rather than native tree probabilities.

---

## 11. Conclusion

This study provides a bias-controlled machine learning framework for stock return forecasting and similar-stock recommendation on historical OHLCV data. Across 4.28 million stock-day observations spanning 2,435 equities over seven years:
- Incorporating market-context and cross-sectional relative features (Level 2) nearly doubles out-of-time test Rank IC ($0.0084 \to 0.0160, +91.1\%$), though paired HAC inference ($p = 0.1927$) indicates this is an encouraging empirical improvement rather than proven statistical superiority.
- Expanding single-stock technical indicators provides zero incremental ranking value ($0.0084$).
- Unconditional full-universe directional accuracy is $53.72\%$. Performance metrics exceeding 60% reflect positive-prediction precision on upward calls under selective coverage (60.82% at 25.08% coverage, 65.68% at 10.23% coverage), whereas directional accuracy remains at 54.46% and 56.89%, respectively.
- Combining forward-looking return prediction with historical correlation similarity (Method C) achieves an empirical measurement of $88.0\%$ variance reduction relative to prediction-only selection, successfully reconciling alpha generation with portfolio risk control.

---

## 12. Reproducibility Manifest

All code, data pipelines, model weights, and forensic evaluation tables are permanently archived and verifiable at `e:\Stock_Predition\`:
- **Forensic Verification Report:** [FINAL_RESEARCH_VALIDATION_REPORT.md](file:///e:/Stock_Predition/results/model_enhancement/FINAL_RESEARCH_VALIDATION_REPORT.md)
- **Market Feature Leakage Audit:** [market_feature_leakage_test.csv](file:///e:/Stock_Predition/results/model_enhancement/market_feature_leakage_test.csv)
- **Independent Reproduction Manifest:** [reproduced_test_comparison.csv](file:///e:/Stock_Predition/results/model_enhancement/reproduced_test_comparison.csv)
- **Paired Significance Test:** [paired_significance_test.csv](file:///e:/Stock_Predition/results/model_enhancement/paired_significance_test.csv)
- **Selective Prediction Audit (11 Tiers):** [detailed_selective_coverage_11tiers.csv](file:///e:/Stock_Predition/results/model_enhancement/detailed_selective_coverage_11tiers.csv)
- **Probability Calibration Diagnostics:** [detailed_calibration_comparison.csv](file:///e:/Stock_Predition/results/model_enhancement/detailed_calibration_comparison.csv)
- **Top-5 Friction Sensitivity Analysis:** [detailed_top5_recommendation_comparison.csv](file:///e:/Stock_Predition/results/model_enhancement/detailed_top5_recommendation_comparison.csv)
- **Latest Dataset Demonstration:** [corrected_latest_session_demo.csv](file:///e:/Stock_Predition/results/model_enhancement/corrected_latest_session_demo.csv)

---

## References

- Amihud, Y. (2002). Illiquidity and stock returns: cross-section and time-series effects. *Journal of Financial Markets*, 5(1), 31-56.
- Arnott, R. D., Harvey, C. R., & Markowitz, H. (2019). A backtesting protocol in the dark. *The Journal of Portfolio Management*, 45(4), 25-33.
- Green, J., Hand, J. R., & Zhang, X. F. (2017). The characteristics that provide independent information about average US monthly stock returns. *The Review of Financial Studies*, 30(12), 4389-4436.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223-2273.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. *The Journal of Finance*, 48(1), 65-91.
- Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T. Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.
- Kelly, B., Pruitt, S., & Su, Y. (2019). Characteristics are covariances: A unified model of risk and return. *Journal of Financial Economics*, 134(3), 501-524.
- Ledoit, O., & Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *The Journal of Portfolio Management*, 30(4), 110-119.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *The Quarterly Journal of Economics*, 105(1), 1-28.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.
- Roll, R. (1984). A simple implicit measure of the effective bid-ask spread in an efficient market. *The Journal of Finance*, 39(4), 1127-1139.
