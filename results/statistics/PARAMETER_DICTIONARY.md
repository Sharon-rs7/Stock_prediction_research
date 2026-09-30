# Scientific Parameter Dictionary: Quantitative Metrics, Features, & Statistical Diagnostics

This document provides the formal mathematical definitions, statistical interpretations, measurement units, data origins, lookahead protection requirements, and confirmatory versus exploratory classification for all parameters analyzed in the study:

**"Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data"**

---

## 1. Raw Market Data & Microstructure Parameters

### 1.1 Unadjusted & Adjusted Pricing (`open`, `high`, `low`, `close`, `adj_close`)
- **Mathematical Formula**:
  $$\text{Pricing Series } P_{i,t} \in \{O_{i,t}, H_{i,t}, L_{i,t}, C_{i,t}, C^{\text{adj}}_{i,t}\}$$
  Adjusted closing price corrects for cash dividends and stock splits:
  $$C^{\text{adj}}_{i,t} = C_{i,t} \prod_{\tau > t} \left(1 - \frac{D_{i,\tau}}{C_{i,\tau}}\right) \times \frac{1}{S_{i,\tau}}$$
- **Statistical Interpretation**: Foundational daily auction pricing bounds determined on primary US exchanges (NYSE, NASDAQ). Unadjusted prices reflect physical transaction prices; adjusted prices reflect total shareholder equity returns.
- **Units**: US Dollars ($)
- **Source**: Yahoo Finance / AmirTrader parquet repository (`Universe B`).
- **Lookahead Protection Required**: Yes. All indicators and features at time $t$ must only use information available at or prior to $t$'s market close.
- **Confirmatory / Exploratory Status**: Base Empirical Inputs (Structural).

---

### 1.2 Trading Volume & Dollar Volume (`volume`, `dollar_volume`)
- **Mathematical Formula**:
  $$\text{Dollar Volume}_{i,t} = C_{i,t} \times V_{i,t}$$
- **Statistical Interpretation**: Daily liquidity scale. $V_{i,t}$ is share volume; Dollar Volume is currency turnover, reflecting capital commitment and market depth.
- **Units**: $V_{i,t}$ in Shares; Dollar Volume in US Dollars ($).
- **Source**: Primary exchange consolidated tape.
- **Lookahead Protection Required**: Yes. Measured contemporaneously at close of day $t$.
- **Confirmatory / Exploratory Status**: Base Empirical Inputs.

---

### 1.3 High-Low Spread Percentage (`hl_spread_pct`)
- **Mathematical Formula**:
  $$\text{hl\_spread\_pct}_{i,t} = \frac{H_{i,t} - L_{i,t}}{C_{i,t} + \epsilon}$$
- **Statistical Interpretation**: Intraday price range normalized by closing price. Measures intraday price dispersion, volatility, and uncertainty.
- **Units**: Dimensionless decimal ratio (1.00 = 100%).
- **Source**: Derived from daily OHLC bar.
- **Lookahead Protection Required**: Yes. Evaluated at day $t$ close.
- **Confirmatory / Exploratory Status**: Base Feature.

---

### 1.4 Open-to-Close Spread Percentage (`oc_spread_pct`)
- **Mathematical Formula**:
  $$\text{oc\_spread\_pct}_{i,t} = \frac{C_{i,t} - O_{i,t}}{O_{i,t} + \epsilon}$$
- **Statistical Interpretation**: Pure daytime auction return from morning market open (09:30 ET) to closing cross (16:00 ET). Isolates intraday order flow from overnight news.
- **Units**: Dimensionless decimal return.
- **Source**: Derived from daily OHLC bar.
- **Lookahead Protection Required**: Yes. Available at close of day $t$.
- **Confirmatory / Exploratory Status**: Base Feature.

---

### 1.5 Overnight Gap (`overnight_gap`)
- **Mathematical Formula**:
  $$\text{overnight\_gap}_{i,t} = \frac{O_{i,t} - C_{i,t-1}}{C_{i,t-1} + \epsilon}$$
- **Statistical Interpretation**: Price adjustment occurring between previous day's close and current day's open. Captures non-trading hours information flow (earnings releases, macro announcements).
- **Units**: Dimensionless decimal return.
- **Source**: Derived from $O_{i,t}$ and $C_{i,t-1}$.
- **Lookahead Protection Required**: Yes. Evaluated at open of day $t$.
- **Confirmatory / Exploratory Status**: Base Feature.

---

## 2. Multi-Horizon Return Parameters

### 2.1 Backward Logarithmic Return (`log_return_1d`)
- **Mathematical Formula**:
  $$r^{\text{log}}_{i,t} = \ln\left(\frac{C_{i,t}}{C_{i,t-1}}\right)$$
- **Statistical Interpretation**: Time-additive rate of return over a 1-day holding period.
- **Units**: Logarithmic return (dimensionless).
- **Source**: Daily adjusted close series.
- **Lookahead Protection Required**: Yes. Available at close of day $t$.
- **Confirmatory / Exploratory Status**: Core Input Variable.

---

### 2.2 Multi-Horizon Discrete Backward Returns (`ret_1d`, `ret_2d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_21d`, `ret_63d`)
- **Mathematical Formula**:
  $$R_{i,t}(k) = \frac{C_{i,t} - C_{i,t-k}}{C_{i,t-k}}$$
  for $k \in \{1, 2, 3, 5, 10, 21, 63\}$.
- **Statistical Interpretation**: Compounded capital appreciation over short-term (1–5 days), medium-term (10–21 days / 1 month), and intermediate-term (63 days / 1 quarter) trailing windows.
- **Units**: Decimal percentage return.
- **Source**: Daily adjusted close series.
- **Lookahead Protection Required**: Yes. Strict causal trailing window.
- **Confirmatory / Exploratory Status**: Confirmatory Feature Inputs.

---

## 3. The 30 Core Causal Feature Signals

### 3.1 Trend & Moving Average Distances (`dist_sma20`, `dist_sma50`, `dist_sma200`)
- **Mathematical Formula**:
  $$\text{dist\_sma}_k(i, t) = \frac{C_{i,t} - \text{SMA}_k(C_{i, t})}{\text{SMA}_k(C_{i, t})}, \quad \text{where } \text{SMA}_k(C_{i,t}) = \frac{1}{k}\sum_{\tau=0}^{k-1} C_{i,t-\tau}$$
- **Statistical Interpretation**: Trend deviation and mean-reversion pressure relative to short (20D), intermediate (50D), and secular (200D) moving averages.
- **Units**: Relative ratio (dimensionless).
- **Lookahead Protection Required**: Yes. Lagged backward moving window.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.2 52-Week High & Low Extremes (`dist_52w_high`, `dist_52w_low`)
- **Mathematical Formula**:
  $$\text{dist\_52w\_high}_{i,t} = \frac{C_{i,t} - \max_{0 \le \tau < 252} H_{i,t-\tau}}{\max_{0 \le \tau < 252} H_{i,t-\tau}}$$
  $$\text{dist\_52w\_low}_{i,t} = \frac{C_{i,t} - \min_{0 \le \tau < 252} L_{i,t-\tau}}{\min_{0 \le \tau < 252} L_{i,t-\tau}}$$
- **Statistical Interpretation**: Proximity to 1-year psychological and institutional anchor price boundaries.
- **Units**: Decimal proportion.
- **Lookahead Protection Required**: Yes. 252-day backward trailing window.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.3 Historical Volatilities (`volatility_21d`, `volatility_63d`)
- **Mathematical Formula**:
  $$\sigma_{i,t}(k) = \sqrt{\frac{252}{k-1} \sum_{\tau=0}^{k-1} \left(r_{i,t-\tau} - \bar{r}_{i,t,k}\right)^2}$$
- **Statistical Interpretation**: Sample annualized standard deviation of daily percentage returns over 1-month ($k=21$) and 3-month ($k=63$) windows.
- **Units**: Annualized volatility (dimensionless decimal).
- **Lookahead Protection Required**: Yes. Causal backward window.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.4 Microstructure Range Estimators (`parkinson_vol`, `garman_klass_vol`)
- **Mathematical Formula**:
  $$\sigma^2_{\text{Parkinson}} = \frac{1}{4 \ln 2} \frac{1}{k}\sum_{\tau=0}^{k-1} \left(\ln\frac{H_{i,t-\tau}}{L_{i,t-\tau}}\right)^2$$
  $$\sigma^2_{\text{Garman-Klass}} = \frac{1}{k}\sum_{\tau=0}^{k-1}\left[0.5\left(\ln\frac{H_{i,t-\tau}}{L_{i,t-\tau}}\right)^2 - (2\ln 2 - 1)\left(\ln\frac{C_{i,t-\tau}}{O_{i,t-\tau}}\right)^2\right]$$
- **Statistical Interpretation**: Continuous Brownian-motion intraday volatility estimators offering up to 5x-8x higher statistical efficiency than close-to-close variance.
- **Units**: Daily variance rate (annualized).
- **Lookahead Protection Required**: Yes. Historical OHLC trailing data.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.5 Average True Range (`atr_14`)
- **Mathematical Formula**:
  $$\text{TR}_{i,t} = \max\left(H_{i,t} - L_{i,t}, |H_{i,t} - C_{i,t-1}|, |L_{i,t} - C_{i,t-1}|\right)$$
  $$\text{ATR}_{14}(i,t) = \frac{1}{14}\sum_{\tau=0}^{13} \text{TR}_{i,t-\tau}$$
- **Statistical Interpretation**: Volatility gauge that incorporates overnight gap jumps into the physical trading range. Normalized by close price.
- **Units**: Price spread normalized by price.
- **Lookahead Protection Required**: Yes. Trailing 14-day window.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.6 Volume & Liquidity Ratios (`volume_ratio_5d`, `volume_ratio_21d`, `log_turnover`)
- **Mathematical Formula**:
  $$\text{volume\_ratio}_k(i,t) = \frac{V_{i,t}}{\frac{1}{k}\sum_{\tau=1}^k V_{i,t-\tau}}$$
  $$\text{log\_turnover}_{i,t} = \ln\left(1 + C_{i,t} \cdot V_{i,t}\right)$$
- **Statistical Interpretation**: Relative volume surge / exhaustion relative to short- and medium-term baselines; logarithmic scale of trading activity.
- **Units**: Ratio (dimensionless) and log-dollars.
- **Lookahead Protection Required**: Yes. Denominator strictly precedes or includes $t$.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.7 Amihud Illiquidity Metric (`amihud_illiq`)
- **Mathematical Formula**:
  $$\text{Amihud}_{i,t} = \frac{1}{21}\sum_{\tau=0}^{20} \frac{|R_{i,t-\tau}|}{\text{Dollar Volume}_{i,t-\tau} + 1e-5}$$
- **Statistical Interpretation**: Average price impact per dollar of trading volume over 21 trading days; measures market depth and liquidity friction.
- **Units**: $\% / \$ \text{ Million}$.
- **Lookahead Protection Required**: Yes. Trailing 21 days.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.8 Roll Effective Spread (`roll_spread`)
- **Mathematical Formula**:
  $$\text{Roll}_{i,t} = 2 \sqrt{-\min\left(0, \text{Cov}\left(\Delta P_{i,t}, \Delta P_{i,t-1}\right)\right)}$$
- **Statistical Interpretation**: Serial covariance estimator of the bid-ask bounce; proxy for transaction friction from daily price changes.
- **Units**: Price spread units.
- **Lookahead Protection Required**: Yes. Trailing 21-day covariance.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.9 Intraday Bar Pressure (`bar_pressure`)
- **Mathematical Formula**:
  $$\text{bar\_pressure}_{i,t} = \frac{C_{i,t} - L_{i,t}}{H_{i,t} - L_{i,t} + \epsilon}$$
- **Statistical Interpretation**: Location of closing price within the day's high-low auction span. $1.0$ indicates closing at high (strong buying pressure); $0.0$ indicates closing at low (strong selling pressure).
- **Units**: Bound $[0, 1]$ index.
- **Lookahead Protection Required**: Yes. Intraday bar at day $t$.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.10 Relative Strength Index (`rsi_14`)
- **Mathematical Formula**:
  $$\text{RSI}_{14} = 100 - \frac{100}{1 + \frac{\text{EMA}_{14}(\text{Up Moves})}{\text{EMA}_{14}(\text{Down Moves})}}$$
- **Statistical Interpretation**: Bounded momentum oscillator measuring the relative magnitude of recent gains versus losses.
- **Units**: Bounded $[0, 100]$.
- **Lookahead Protection Required**: Yes. Wilder's exponential smoothing over 14 days.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.11 Moving Average Convergence Divergence (`macd_diff`)
- **Mathematical Formula**:
  $$\text{MACD Line} = \text{EMA}_{12}(C) - \text{EMA}_{26}(C)$$
  $$\text{Signal Line} = \text{EMA}_9(\text{MACD Line})$$
  $$\text{macd\_diff} = \frac{\text{MACD Line} - \text{Signal Line}}{C_{i,t}}$$
- **Statistical Interpretation**: Trend acceleration and momentum divergence normalized by asset price.
- **Units**: Dimensionless percentage.
- **Lookahead Protection Required**: Yes. Trailing recursive EMA.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.12 Bollinger Bands %B (`bb_pct_b`)
- **Mathematical Formula**:
  $$\text{bb\_pct\_b}_{i,t} = \frac{C_{i,t} - \left(\text{SMA}_{20} - 2\sigma_{20}\right)}{4\sigma_{20}}$$
- **Statistical Interpretation**: Quantifies price position within a 2-standard-deviation envelope around the 20-day mean. Values $>1.0$ denote upper-band breakout; $<0.0$ denote lower-band penetration.
- **Units**: Dimensionless ratio.
- **Lookahead Protection Required**: Yes. Trailing 20 days.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.13 VWAP Ratios & Proxies (`vwap_ratio_5d`, `dist_vwap_proxy`)
- **Mathematical Formula**:
  $$\text{Typical Price}_{i,t} = \frac{H_{i,t} + L_{i,t} + C_{i,t}}{3}$$
  $$\text{VWAP}_k(i,t) = \frac{\sum_{\tau=0}^{k-1} \text{Typical Price}_{i,t-\tau} \cdot V_{i,t-\tau}}{\sum_{\tau=0}^{k-1} V_{i,t-\tau}}$$
  $$\text{vwap\_ratio\_5d}_{i,t} = \frac{C_{i,t}}{\text{VWAP}_5(i,t)}$$
- **Statistical Interpretation**: Deviation of current price from volume-weighted average price benchmark.
- **Units**: Ratio.
- **Lookahead Protection Required**: Yes. Trailing historical bars.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

### 3.14 Chaikin Money Flow & Money Flow Index (`cmf_20`, `mfi_14`)
- **Mathematical Formula**:
  $$\text{MF Multiplier}_{i,t} = \frac{(C_{i,t} - L_{i,t}) - (H_{i,t} - C_{i,t})}{H_{i,t} - L_{i,t} + \epsilon}$$
  $$\text{CMF}_{20} = \frac{\sum_{\tau=0}^{19} \text{MF Multiplier}_{i,t-\tau} \cdot V_{i,t-\tau}}{\sum_{\tau=0}^{19} V_{i,t-\tau}}$$
- **Statistical Interpretation**: Volume-weighted measures of institutional accumulation versus distribution.
- **Units**: $[-1, 1]$ for CMF; $[0, 100]$ for MFI.
- **Lookahead Protection Required**: Yes. Trailing historical windows.
- **Confirmatory / Exploratory Status**: Confirmatory Base Feature.

---

## 4. Enhanced Cross-Sectional & Technical Features (Levels 2 & 3)

### 4.1 Cross-Sectional Relative Metrics (`rel_ret_5d`, `rel_ret_21d`, `rel_vol_21d`, `rel_volume_5d`)
- **Mathematical Formula**:
  $$\text{rel\_ret}_k(i,t) = R_{i,t}(k) - \frac{1}{N_t}\sum_{j=1}^{N_t} R_{j,t}(k)$$
  $$\text{rel\_vol}_{21d}(i,t) = \frac{\sigma_{i,t}(21)}{\text{Median}_{j \in \mathcal{U}_t}\left(\sigma_{j,t}(21)\right)}$$
- **Statistical Interpretation**: Isolates idiosyncratic stock behavior from systemic market-wide beta moves across the 2,435-equity universe.
- **Units**: Excess return decimal and relative variance ratio.
- **Lookahead Protection Required**: Strict contemporaneous cross-sectional standardization across day $t$.
- **Confirmatory / Exploratory Status**: Confirmatory Enhanced Feature (Level 2).

---

### 4.2 Cross-Sectional Percentile Ranks (`pct_rank_ret_21d`, `pct_rank_vol_21d`, `pct_rank_turnover`, `pct_rank_dist_sma200`)
- **Mathematical Formula**:
  $$\text{pct\_rank}(X_{i,t}) = \frac{\text{Rank}\left(X_{i,t} \mid \{X_{j,t}\}_{j=1}^{N_t}\right) - 1}{N_t - 1}$$
- **Statistical Interpretation**: Non-parametric uniform mapping $U[0, 1]$ of asset metrics across the entire equity cross-section on day $t$. Robust to extreme outliers and fat-tailed distribution shifts.
- **Units**: Quantile range $[0, 1]$.
- **Lookahead Protection Required**: Cross-sectional grouping restricted strictly to session $t$.
- **Confirmatory / Exploratory Status**: Confirmatory Enhanced Feature (Level 2).

---

### 4.3 Second-Order Technical Accelerations (`tech_mom_accel`, `tech_trend_accel`, `tech_vol_accel`, `tech_pv_interaction`, `tech_shadow_asym`, `tech_bar_efficiency`)
- **Mathematical Formula**:
  $$\text{tech\_mom\_accel}_{i,t} = R_{i,t}(5) - \frac{R_{i,t}(21)}{4.2}$$
  $$\text{tech\_pv\_interaction}_{i,t} = R_{i,t}(1) \cdot \ln(1 + \text{volume\_ratio\_5d}_{i,t})$$
  $$\text{tech\_shadow\_asym}_{i,t} = \frac{(H_{i,t} - \max(O_{i,t}, C_{i,t})) - (\min(O_{i,t}, C_{i,t}) - L_{i,t})}{H_{i,t} - L_{i,t} + \epsilon}$$
- **Statistical Interpretation**: Non-linear derivatives representing physical acceleration of price momentum, buyer/seller wick asymmetry, and volume-price synergy.
- **Units**: Dimensionless derivatives.
- **Lookahead Protection Required**: Yes. Historical OHLCV values.
- **Confirmatory / Exploratory Status**: Exploratory Technical Feature (Level 3).

---

## 5. Forecast Target Variables

### 5.1 Raw Forward Return (`fwd_ret_H`)
- **Mathematical Formula**:
  $$Y^{\text{raw}}_{i,t}(H) = \frac{C_{i, t+H} - C_{i,t}}{C_{i,t}}$$
  for $H \in \{1, 5, 21\}$.
- **Statistical Interpretation**: Unadjusted percentage capital appreciation over the forward holding period $t \to t+H$.
- **Units**: Percentage return decimal.
- **Lookahead Protection Required**: Strict target separation. $Y_{i,t}(H)$ is used ONLY as supervision ground truth during training, never as a feature.
- **Confirmatory / Exploratory Status**: $H=5$ is Primary Confirmatory; $H=1$ and $H=21$ are Robustness/Exploratory.

---

### 5.2 Market-Excess Forward Return (`excess_fwd_ret_H`)
- **Mathematical Formula**:
  $$Y^{\text{excess}}_{i,t}(H) = Y^{\text{raw}}_{i,t}(H) - \bar{Y}^{\text{raw}}_t(H), \quad \text{where } \bar{Y}^{\text{raw}}_t(H) = \frac{1}{N_t}\sum_{j=1}^{N_t} Y^{\text{raw}}_{j,t}(H)$$
- **Statistical Interpretation**: Forward alpha return relative to the equal-weighted cross-sectional universe benchmark. Eliminates systemic market drift.
- **Units**: Decimal excess return.
- **Lookahead Protection Required**: Target variable only.
- **Confirmatory / Exploratory Status**: Confirmatory Target (H=5).

---

### 5.3 Cross-Sectional Standardized Forward Target (`zscore_fwd_ret_H`)
- **Mathematical Formula**:
  $$Z_{i,t}(H) = \frac{Y^{\text{raw}}_{i,t}(H) - \text{Mean}_{j \in \mathcal{U}_t}(Y^{\text{raw}}_{j,t}(H))}{\text{Std}_{j \in \mathcal{U}_t}(Y^{\text{raw}}_{j,t}(H)) + \epsilon}$$
- **Statistical Interpretation**: Standard normal target ($Z \sim (0, 1)$ across cross-section) removing cross-sectional heteroscedasticity and market regime volatility spikes.
- **Units**: Standard deviations ($z$-score).
- **Lookahead Protection Required**: Target variable only.
- **Confirmatory / Exploratory Status**: Primary Confirmatory Target for Regression Models.

---

### 5.4 Binary Directional Target (`dir_target`)
- **Mathematical Formula**:
  $$\text{dir\_target}_{i,t} = \mathbb{I}\left(Y^{\text{raw}}_{i,t}(5) > 0\right) = \begin{cases} 1 & \text{if } C_{i,t+5} > C_{i,t} \\ 0 & \text{if } C_{i,t+5} \le C_{i,t} \end{cases}$$
- **Statistical Interpretation**: Ground truth indicator of positive price direction over the 5-day horizon. Base rate in locked test set: $51.23\%$.
- **Units**: Binary $\{0, 1\}$.
- **Lookahead Protection Required**: Target variable only.
- **Confirmatory / Exploratory Status**: Primary Confirmatory Classification Target.

---

## 6. Statistical Moments & Descriptive Parameters (32 Required Parameters)

| Parameter Name | Formal Mathematical Formula | Exact Statistical Interpretation | Measurement Units |
| :--- | :--- | :--- | :--- |
| **$N$** | $N = \sum_{k=1}^K 1$ | Total sample observations count | Integer count |
| **Missing Count** | $N_{\text{nan}} = \sum \mathbb{I}(X_k \text{ is NaN})$ | Count of unrecorded or undefined entries | Integer count |
| **Missing %** | $\frac{N_{\text{nan}}}{N} \times 100\%$ | Proportion of missing observations | Percentage (%) |
| **Zero Count** | $N_0 = \sum \mathbb{I}(X_k == 0)$ | Count of exact zero values | Integer count |
| **Zero %** | $\frac{N_0}{N_{\text{clean}}} \times 100\%$ | Proportion of exact zero observations | Percentage (%) |
| **Unique Count** | $\|\{x \in X\}\|$ | Number of distinct discrete values | Integer count |
| **Sum** | $\sum_{k=1}^{N_{\text{clean}}} X_k$ | Total aggregate sum | Native variable units |
| **Mean** | $\mu = \frac{1}{N}\sum_{k=1}^N X_k$ | First central moment (arithmetic average) | Native variable units |
| **Median ($P_{50}$)** | $\inf\{x : F(x) \ge 0.50\}$ | 50th percentile (50% probability midpoint) | Native variable units |
| **Modal Interval** | $\arg\max_b \int_{e_b}^{e_{b+1}} dF(x)$ | Most frequent empirical density bin | Interval range $[a, b]$ |
| **Minimum** | $\min_{k} X_k$ | Absolute lower empirical boundary | Native variable units |
| **Maximum** | $\max_{k} X_k$ | Absolute upper empirical boundary | Native variable units |
| **Range** | $\max(X) - \min(X)$ | Empirical dispersion span | Native variable units |
| **Variance** | $s^2 = \frac{1}{N-1}\sum_{k=1}^N (X_k - \mu)^2$ | Second central sample moment | Squared native units |
| **Std Dev** | $s = \sqrt{s^2}$ | Standard dispersion metric | Native variable units |
| **Coeff. of Variation** | $\text{CV} = \frac{s}{\|\mu\| + \epsilon}$ | Scale-free relative dispersion | Dimensionless ratio |
| **Std Error of Mean** | $\text{SEM} = \frac{s}{\sqrt{N}}$ | Sampling distribution standard deviation of mean | Native variable units |
| **Quartile 1 ($Q_1$)** | $\inf\{x : F(x) \ge 0.25\}$ | 25th percentile | Native variable units |
| **Quartile 2 ($Q_2$)** | $\inf\{x : F(x) \ge 0.50\}$ | 50th percentile (Median) | Native variable units |
| **Quartile 3 ($Q_3$)** | $\inf\{x : F(x) \ge 0.75\}$ | 75th percentile | Native variable units |
| **IQR** | $\text{IQR} = Q_3 - Q_1$ | Interquartile range (middle 50% spread) | Native variable units |
| **Percentile 1 ($P_1$)** | $\inf\{x : F(x) \ge 0.01\}$ | Extreme 1st percentile left tail | Native variable units |
| **Percentile 5 ($P_5$)** | $\inf\{x : F(x) \ge 0.05\}$ | 5th percentile risk boundary | Native variable units |
| **Percentile 10 ($P_{10}$)** | $\inf\{x : F(x) \ge 0.10\}$ | 10th percentile boundary | Native variable units |
| **Percentile 25 ($P_{25}$)** | $Q_1$ | Lower quartile | Native variable units |
| **Percentile 50 ($P_{50}$)** | Median | Median value | Native variable units |
| **Percentile 75 ($P_{75}$)** | $Q_3$ | Upper quartile | Native variable units |
| **Percentile 90 ($P_{90}$)** | $\inf\{x : F(x) \ge 0.90\}$ | 90th percentile boundary | Native variable units |
| **Percentile 95 ($P_{95}$)** | $\inf\{x : F(x) \ge 0.95\}$ | 95th percentile right tail | Native variable units |
| **Percentile 99 ($P_{99}$)** | $\inf\{x : F(x) \ge 0.99\}$ | Extreme 99th percentile right tail | Native variable units |
| **MAD** | $\text{Median}(\|X_k - \text{Median}(X)\|)$ | Median Absolute Deviation (robust scale) | Native variable units |
| **Skewness** | $\frac{1}{N}\sum \left(\frac{X_k - \mu}{s}\right)^3$ | Third standardized moment (asymmetry) | Dimensionless |
| **Kurtosis** | $\frac{1}{N}\sum \left(\frac{X_k - \mu}{s}\right)^4 - 3$ | Excess kurtosis (fat tails relative to Normal) | Dimensionless |

---

## 7. Model Performance, Diagnostics & Inferential Parameters

### 7.1 Cross-Sectional Spearman Rank Information Coefficient (`Rank IC`)
- **Mathematical Formula**:
  $$\text{Rank IC}_t = \rho_s\left(\text{Rank}(\hat{Y}_{\cdot, t}), \text{Rank}(Y_{\cdot, t})\right) = 1 - \frac{6 \sum_{i=1}^{N_t} d_{i,t}^2}{N_t(N_t^2 - 1)}$$
- **Statistical Interpretation**: Cross-sectional monotonic correlation between predicted rankings and realized forward rankings on session $t$.
- **Units**: Bounded $[-1, 1]$.
- **Confirmatory / Exploratory Status**: Primary Confirmatory Evaluation Metric.

---

### 7.2 Information Ratio (`IR`)
- **Mathematical Formula**:
  $$\text{IR} = \frac{\overline{\text{Rank IC}}}{\sigma(\text{Rank IC})}$$
- **Statistical Interpretation**: Risk-adjusted consistency of predictive ranking capability over the test timeline.
- **Units**: Dimensionless ratio.
- **Confirmatory / Exploratory Status**: Primary Confirmatory Evaluation Metric.

---

### 7.3 Newey-West Heteroskedasticity & Autocorrelation Consistent $t$-Statistic (`HAC t-stat`)
- **Mathematical Formula**:
  $$\hat{\sigma}^2_{\text{HAC}} = \hat{\Gamma}_0 + 2\sum_{l=1}^L \left(1 - \frac{l}{L+1}\right)\hat{\Gamma}_l, \quad t_{\text{HAC}} = \frac{\bar{X}}{\sqrt{\hat{\sigma}^2_{\text{HAC}} / T}}$$
  where $L = 5$ lags for 5-day overlapping returns.
- **Statistical Interpretation**: Asymptotically valid inferential test statistic robust to autocorrelation induced by overlapping rolling return windows and heteroscedasticity.
- **Units**: Standardized $t$-ratio.
- **Confirmatory / Exploratory Status**: Primary Inferential Standard (replaces invalid naive i.i.d. $t$-tests).

---

### 7.4 Expected Calibration Error (`ECE`) & Brier Score
- **Mathematical Formula**:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
  $$\text{Brier} = \frac{1}{N}\sum_{i=1}^N (p_i - y_i)^2$$
- **Statistical Interpretation**: ECE measures average probability deviation from empirical reliability across confidence bins; Brier score is the mean squared probabilistic calibration error.
- **Units**: Probability deviation $[0, 1]$.
- **Confirmatory / Exploratory Status**: Probabilistic Diagnostic Metric.

---

### 7.5 Population Stability Index (`PSI`) & Wasserstein Distance
- **Mathematical Formula**:
  $$\text{PSI} = \sum_{b=1}^B \left(P_{\text{actual}, b} - P_{\text{expected}, b}\right) \times \ln\left(\frac{P_{\text{actual}, b}}{P_{\text{expected}, b}}\right)$$
  $$\mathcal{W}_1(u, v) = \int_{-\infty}^\infty |U(x) - V(x)| dx$$
- **Statistical Interpretation**: PSI measures degree of feature distribution drift between training reference population and out-of-time test partitions. $\text{PSI} < 0.10$ indicates stable distribution; $>0.25$ indicates significant structural shift. Wasserstein distance is the optimal transport cost (Earth Mover's Distance) between distributions.
- **Units**: PSI in entropy nats; Wasserstein in feature units.
- **Confirmatory / Exploratory Status**: Diagnostic / Audit Metric.

---

### 7.6 Benjamini-Hochberg False Discovery Rate Adjusted $p$-Value (`BH FDR`)
- **Mathematical Formula**:
  $$\text{Reject } H_{(i)} \text{ if } P_{(i)} \le \frac{i}{m} \alpha$$
  where $P_{(1)} \le P_{(2)} \le \dots \le P_{(m)}$ are ordered $p$-values across $m$ simultaneous hypotheses, and $\alpha = 0.05$.
- **Statistical Interpretation**: Rigorous statistical control bounding the expected proportion of false positive discoveries among all rejected null hypotheses across the experimental family.
- **Units**: Probability boundary $[0, 1]$.
- **Confirmatory / Exploratory Status**: Confirmatory Multiplicity Correction.
