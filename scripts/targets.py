"""
Target Construction Module.
Computes forward returns and cross-sectional target metrics.
Primary Horizon: H = 5 trading days.
Secondary Horizons: H = 1, H = 21 trading days (for robustness).
Primary Target: Daily cross-sectional z-score of 5-day forward return.
Secondary Targets: Raw 5-day forward return, Market-relative/excess 5-day forward return.
"""

import numpy as np
import pandas as pd

def compute_forward_returns(df: pd.DataFrame, horizons=[1, 5, 21]) -> pd.DataFrame:
    """
    Computes forward returns for a single stock time series sorted by date.
    Input df must contain 'adj_close' and 'date'.
    y(i, t, H) = (AdjClose_{t+H} - AdjClose_t) / AdjClose_t
    """
    df = df.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    out = pd.DataFrame(index=df.index)
    out['date'] = df['date']
    if 'ticker' in df.columns:
        out['ticker'] = df['ticker']
    
    adj = df['adj_close']
    for h in horizons:
        out[f'fwd_ret_{h}d'] = (adj.shift(-h) - adj) / adj
        
    return out

def compute_cross_sectional_targets(panel_df: pd.DataFrame, primary_horizon=5) -> pd.DataFrame:
    """
    Computes daily cross-sectional z-score and cross-sectional market-relative excess returns.
    Target panel_df must contain ['date', 'ticker', f'fwd_ret_{primary_horizon}d', ...].
    IMPORTANT: Calculated CROSS-SECTIONALLY across stocks for each date t.
    """
    df = panel_df.copy()
    
    for h in [1, 5, 21]:
        ret_col = f'fwd_ret_{h}d'
        if ret_col not in df.columns:
            continue
            
        # Group by date to perform cross-sectional transformation
        grouped = df.groupby('date')[ret_col]
        mean_t = grouped.transform('mean')
        std_t = grouped.transform('std')
        
        # 1. Market-relative / excess forward return: y(i,t) - mean_t(y)
        df[f'excess_fwd_ret_{h}d'] = df[ret_col] - mean_t
        
        # 2. Cross-sectional z-score: (y(i,t) - mean_t) / std_t
        df[f'zscore_fwd_ret_{h}d'] = (df[ret_col] - mean_t) / (std_t + 1e-8)
        
    return df
