### Table 2: Formal Specification of 30 OHLCV Features Across 5 Structural Groups

| Feature Symbol | Group | Mathematical Formulation / Definition | Window ($w$) |
| :--- | :--- | :--- | :--- |
| `ret_1d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-1})$ | 1 day |
| `ret_5d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-5})$ | 5 days |
| `ret_10d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-10})$ | 10 days |
| `ret_21d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-21})$ | 21 days |
| `ret_63d` | G1 Momentum | $\\ln(AdjClose_t / AdjClose_{t-63})$ | 63 days |
| `vol_5d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 5 days |
| `vol_21d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 21 days |
| `vol_63d` | G2 Volatility | Rolling standard deviation of `ret_1d` | 63 days |
| `parkinson_vol_21d` | G2 Volatility | $\\sqrt{\\frac{1}{4 \\ln 2 \\cdot 21} \\sum_{i=0}^{20} (\\ln(H_{t-i}/L_{t-i}))^2}$ | 21 days |
| `natr_14d` | G2 Volatility | Normalized ATR: $\\text{ATR}(14)_t / Close_t$ | 14 days |
| `ret_skew_21d` | G2 Volatility | Rolling sample skewness of `ret_1d` | 21 days |
| `dist_sma_20` | G3 Trend | $(Close_t - \\text{SMA}_{20}) / \\text{SMA}_{20}$ | 20 days |
| `dist_sma_50` | G3 Trend | $(Close_t - \\text{SMA}_{50}) / \\text{SMA}_{50}$ | 50 days |
| `dist_sma_200` | G3 Trend | $(Close_t - \\text{SMA}_{200}) / \\text{SMA}_{200}$ | 200 days |
| `rsi_14d` | G3 Trend | Relative Strength Index: $100 - (100 / (1 + RS))$ | 14 days |
| `macd_diff` | G3 Trend | Normalized MACD: $(\\text{MACD Line} - \\text{Signal Line}) / Close_t$ | 12, 26, 9 days |
| `bollinger_pct_b` | G3 Trend | $(Close_t - \\text{LowerBand}) / (\\text{UpperBand} - \\text{LowerBand})$ | 20 days ($\\pm 2\\sigma$) |
| `vol_ratio_5d` | G4 Volume | $Volume_t / \\text{SMA}(Volume, 5)_t$ | 5 days |
| `vol_ratio_21d` | G4 Volume | $Volume_t / \\text{SMA}(Volume, 21)_t$ | 21 days |
| `log_turnover` | G4 Volume | $\\ln(Close_t \\cdot Volume_t + 1)$ (Dollar volume) | 1 day |
| `turnover_vol_21d` | G4 Volume | Rolling standard deviation of `log_turnover` | 21 days |
| `amihud_illiq_21d` | G4 Volume | Amihud illiquidity ratio: $\\frac{1}{21}\\sum \\frac{\|ret\\_1d\|}{DollarVolume}$ | 21 days |
| `obv_slope_10d` | G4 Volume | Normalized slope of On-Balance Volume over trailing window | 10 days |
| `hl_spread` | G5 Bar Geometry | High-Low relative range: $(High_t - Low_t) / Close_t$ | 1 day |
| `oc_return` | G5 Bar Geometry | Intraday bar return: $(Close_t - Open_t) / Open_t$ | 1 day |
| `overnight_gap` | G5 Bar Geometry | Overnight price jump: $(Open_t - Close_{t-1}) / Close_{t-1}$ | 1 day |
| `upper_shadow` | G5 Bar Geometry | Candle upper wick: $(High_t - \\max(Open_t, Close_t)) / Close_t$ | 1 day |
| `lower_shadow` | G5 Bar Geometry | Candle lower wick: $(\\min(Open_t, Close_t) - Low_t) / Close_t$ | 1 day |
| `bar_pressure` | G5 Bar Geometry | Intra-bar buying pressure: $(Close_t - Low_t) / (High_t - Low_t)$ | 1 day |
| `roll_spread_21d` | G5 Bar Geometry | Roll (1984) effective bid-ask spread estimator | 21 days |

*Note: All 30 features use strictly historical information $\\le t$ to guarantee zero look-ahead bias.*
