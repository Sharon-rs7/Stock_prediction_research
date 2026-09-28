"""
Feature Engineering Module for Stock Price Forecasting.
Implements the 30 strictly defined OHLCV features across 5 groups.
All calculations use strictly historical information available at or before time t (zero look-ahead).
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    # Group 1: Momentum / Log Returns
    "ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d",
    # Group 2: Volatility / Tail Risk
    "vol_5d", "vol_21d", "vol_63d", "parkinson_vol_21d", "natr_14d", "ret_skew_21d",
    # Group 3: Trend / Moving Averages
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "rsi_14d", "macd_diff", "bollinger_pct_b",
    # Group 4: Volume / Liquidity
    "vol_ratio_5d", "vol_ratio_21d", "log_turnover", "turnover_vol_21d", "amihud_illiq_21d", "obv_slope_10d",
    # Group 5: Bar Geometry
    "hl_spread", "oc_return", "overnight_gap", "upper_shadow", "lower_shadow", "bar_pressure", "roll_spread_21d"
]

FEATURE_GROUPS = {
    "G1_MOMENTUM": ["ret_1d", "ret_5d", "ret_10d", "ret_21d", "ret_63d"],
    "G2_VOLATILITY": ["vol_5d", "vol_21d", "vol_63d", "parkinson_vol_21d", "natr_14d", "ret_skew_21d"],
    "G3_TREND": ["dist_sma_20", "dist_sma_50", "dist_sma_200", "rsi_14d", "macd_diff", "bollinger_pct_b"],
    "G4_VOLUME": ["vol_ratio_5d", "vol_ratio_21d", "log_turnover", "turnover_vol_21d", "amihud_illiq_21d", "obv_slope_10d"],
    "G5_BAR_GEOMETRY": ["hl_spread", "oc_return", "overnight_gap", "upper_shadow", "lower_shadow", "bar_pressure", "roll_spread_21d"]
}

def compute_obv_slope(obv_series, vol_series, window=10):
    """Computes rolling linear regression slope of OBV over window normalized by average volume."""
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    
    # We can compute rolling slope using convolution / rolling apply
    # slope = sum((x_i - x_mean) * (y_i - y_mean)) / x_var
    # weights for y_i are (x_i - x_mean) / x_var
    weights = (x - x_mean) / x_var
    
    rolling_slope = obv_series.rolling(window).apply(lambda y: np.dot(y, weights), raw=True)
    norm_vol = vol_series.rolling(window).mean() + 1e-8
    return rolling_slope / norm_vol

def compute_roll_spread(close_series, window=21):
    """Computes Roll (1984) effective bid-ask spread estimate over window."""
    dp = close_series.diff()
    dp_lag = dp.shift(1)
    
    # Rolling covariance between dp and dp_lag
    roll_cov = (dp * dp_lag).rolling(window).mean() - (dp.rolling(window).mean() * dp_lag.rolling(window).mean())
    # Roll spread = 2 * sqrt(max(0, -cov)) / close
    spread = 2.0 * np.sqrt(np.maximum(0.0, -roll_cov)) / (close_series + 1e-8)
    return spread

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all 30 OHLCV features for a single stock's time series.
    Input df must contain: ['date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
    Sorted chronologically by date.
    """
    df = df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    open_p = df['open'].values
    high_p = df['high'].values
    low_p = df['low'].values
    close_p = df['close'].values
    adj_close = df['adj_close']
    vol = df['volume']
    
    out = pd.DataFrame(index=df.index)
    out['date'] = df['date']
    if 'ticker' in df.columns:
        out['ticker'] = df['ticker']
    out['adj_close'] = adj_close
    out['close'] = close_p
    out['volume'] = vol
    
    # ----------------------------------------------------
    # GROUP 1: MOMENTUM / LOG RETURNS (using adj_close)
    # ----------------------------------------------------
    out['ret_1d'] = np.log(adj_close / adj_close.shift(1))
    out['ret_5d'] = np.log(adj_close / adj_close.shift(5))
    out['ret_10d'] = np.log(adj_close / adj_close.shift(10))
    out['ret_21d'] = np.log(adj_close / adj_close.shift(21))
    out['ret_63d'] = np.log(adj_close / adj_close.shift(63))
    
    # ----------------------------------------------------
    # GROUP 2: VOLATILITY / TAIL RISK
    # ----------------------------------------------------
    out['vol_5d'] = out['ret_1d'].rolling(5).std()
    out['vol_21d'] = out['ret_1d'].rolling(21).std()
    out['vol_63d'] = out['ret_1d'].rolling(63).std()
    
    # Parkinson volatility 21d: sqrt( (1 / (4 * ln(2) * 21)) * sum(ln(H/L)^2) )
    log_hl_sq = (np.log(np.maximum(high_p, 1e-8) / np.maximum(low_p, 1e-8))) ** 2
    out['parkinson_vol_21d'] = np.sqrt(
        pd.Series(log_hl_sq, index=df.index).rolling(21).sum() / (4.0 * np.log(2.0) * 21.0)
    )
    
    # NATR 14d: ATR(14) / close
    prev_close = pd.Series(close_p, index=df.index).shift(1)
    tr1 = high_p - low_p
    tr2 = np.abs(high_p - prev_close.values)
    tr3 = np.abs(low_p - prev_close.values)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr, index=df.index).rolling(14).mean()
    out['natr_14d'] = atr_14 / (close_p + 1e-8)
    
    # Rolling skewness of ret_1d over 21 days
    out['ret_skew_21d'] = out['ret_1d'].rolling(21).skew()
    
    # ----------------------------------------------------
    # GROUP 3: TREND / MOVING AVERAGES
    # ----------------------------------------------------
    close_s = pd.Series(close_p, index=df.index)
    sma_20 = close_s.rolling(20).mean()
    sma_50 = close_s.rolling(50).mean()
    sma_200 = close_s.rolling(200).mean()
    
    out['dist_sma_20'] = (close_s - sma_20) / (sma_20 + 1e-8)
    out['dist_sma_50'] = (close_s - sma_50) / (sma_50 + 1e-8)
    out['dist_sma_200'] = (close_s - sma_200) / (sma_200 + 1e-8)
    
    # RSI 14d
    delta = close_s.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain, index=df.index).rolling(14).mean()
    avg_loss = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    out['rsi_14d'] = 100.0 - (100.0 / (1.0 + rs))
    
    # MACD diff: EMA12 - EMA26 minus Signal (EMA9 of MACD line), normalized by close
    ema_12 = close_s.ewm(span=12, adjust=False).mean()
    ema_26 = close_s.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out['macd_diff'] = (macd_line - macd_signal) / (close_s + 1e-8)
    
    # Bollinger %B: (close - lower) / (upper - lower) with 20-day SMA +/- 2 std
    std_20 = close_s.rolling(20).std()
    upper_b = sma_20 + 2.0 * std_20
    lower_b = sma_20 - 2.0 * std_20
    out['bollinger_pct_b'] = (close_s - lower_b) / (upper_b - lower_b + 1e-8)
    
    # ----------------------------------------------------
    # GROUP 4: VOLUME / LIQUIDITY
    # ----------------------------------------------------
    vol_s = pd.Series(vol, index=df.index)
    sma_vol_5 = vol_s.rolling(5).mean()
    sma_vol_21 = vol_s.rolling(21).mean()
    out['vol_ratio_5d'] = vol_s / (sma_vol_5 + 1e-8)
    out['vol_ratio_21d'] = vol_s / (sma_vol_21 + 1e-8)
    
    # Dollar turnover: close * volume
    dollar_turnover = close_s * vol_s
    out['log_turnover'] = np.log(dollar_turnover + 1.0)
    out['turnover_vol_21d'] = out['log_turnover'].rolling(21).std()
    
    # Amihud illiquidity: mean(|ret_1d| / dollar_turnover) over 21 days
    amihud_daily = np.abs(out['ret_1d']) / (dollar_turnover + 1e-8)
    out['amihud_illiq_21d'] = amihud_daily.rolling(21).mean()
    
    # OBV slope 10d
    price_change_sign = np.sign(delta).fillna(0)
    obv = (price_change_sign * vol_s).cumsum()
    out['obv_slope_10d'] = compute_obv_slope(obv, vol_s, window=10)
    
    # ----------------------------------------------------
    # GROUP 5: BAR GEOMETRY
    # ----------------------------------------------------
    close_arr = close_p + 1e-8
    out['hl_spread'] = (high_p - low_p) / close_arr
    out['oc_return'] = (close_p - open_p) / (open_p + 1e-8)
    out['overnight_gap'] = (open_p - prev_close.values) / (prev_close.values + 1e-8)
    out['upper_shadow'] = (high_p - np.maximum(open_p, close_p)) / close_arr
    out['lower_shadow'] = (np.minimum(open_p, close_p) - low_p) / close_arr
    out['bar_pressure'] = (close_p - low_p) / (high_p - low_p + 1e-8)
    out['roll_spread_21d'] = compute_roll_spread(close_s, window=21)
    
    return out
