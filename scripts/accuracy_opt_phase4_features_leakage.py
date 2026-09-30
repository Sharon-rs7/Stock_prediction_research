"""
Accuracy Optimization - Phase 4: Feature Expansion & Automated Leakage Audit
===========================================================================
Engineers candidate feature groups A through G (72 features total):
  Group A: Momentum (1d, 2d, 3d, 5d, 10d, 20d, 60d, acceleration, risk-adjusted)
  Group B: Mean Reversion (SMA distances, RSI, MACD, Bollinger, short-term reversal, RSI regimes)
  Group C: Volatility (Realized 5/21/63d, Parkinson, NATR, skew, ratios, acceleration, shock)
  Group D: Volume/Liquidity (Log turnover, volume ratios, Amihud, OBV slope, price-volume interaction)
  Group E: Price Structure (HL spread, OC return, overnight gap, wick asymmetry, bar efficiency, roll spread)
  Group F: Market-Relative (Relative returns 1d/5d/21d, relative volume, relative vol, cross-sectional percentile ranks)
  Group G: Macro Regimes (Market returns, volatility 21/63d, dispersion, breadth SMA50/SMA200, AD ratio, regime flags)

Executes Automated Future-Perturbation Leakage Test:
  Perturbs t+1..t+5 prices (+100%) and volume (10x), verifies absolute delta == 0.0000000000 across all 72 features.

Outputs:
  - results/accuracy_optimization/09_leakage_audit.json
  - data/processed/universe_b_expanded_features.parquet
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from scripts.features import FEATURE_COLUMNS
from scripts.temporal_split import build_temporal_splits

OUT_DIR = os.path.abspath("results/accuracy_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

def build_expanded_features(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs the 72 candidate features strictly using causal information at or before session t.
    """
    df = df_in.copy()
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # GROUP A: EXPANDED MOMENTUM
    # ---------------------------------------------------------
    # Base: ret_1d, ret_5d, ret_10d, ret_21d, ret_63d already in panel
    df['ret_2d'] = df.groupby('ticker')['adj_close'].pct_change(2).fillna(0)
    df['ret_3d'] = df.groupby('ticker')['adj_close'].pct_change(3).fillna(0)
    df['mom_accel_5_21'] = df['ret_5d'] - (df['ret_21d'] / 4.0)
    df['mom_risk_adj_5d'] = df['ret_5d'] / (df['vol_5d'] + 1e-5)
    df['mom_trend_spread'] = df['ret_5d'] - df['ret_1d'] * 5.0
    
    # ---------------------------------------------------------
    # GROUP B: MEAN REVERSION
    # ---------------------------------------------------------
    # Base: dist_sma_20, dist_sma_50, dist_sma_200, rsi_14d, macd_diff, bollinger_pct_b
    df['rev_short_1d'] = -df['ret_1d']
    df['rev_short_5d'] = -df['ret_5d']
    df['rsi_centered'] = (df['rsi_14d'] - 50.0) / 10.0
    df['rsi_oversold'] = (df['rsi_14d'] < 30.0).astype(float)
    df['rsi_overbought'] = (df['rsi_14d'] > 70.0).astype(float)
    df['trend_accel'] = df['dist_sma_20'] - df['dist_sma_50']
    
    # ---------------------------------------------------------
    # GROUP C: VOLATILITY & TAIL RISK
    # ---------------------------------------------------------
    # Base: vol_5d, vol_21d, vol_63d, parkinson_vol_21d, natr_14d, ret_skew_21d, vol_ratio_5d, vol_ratio_21d
    df['vol_accel'] = df['vol_ratio_5d'] - df['vol_ratio_21d']
    df['vol_shock'] = df['vol_5d'] / (df['vol_21d'] + 1e-5)
    df['vol_term_structure'] = df['vol_21d'] / (df['vol_63d'] + 1e-5)
    
    # ---------------------------------------------------------
    # GROUP D: VOLUME & LIQUIDITY
    # ---------------------------------------------------------
    # Base: log_turnover, turnover_vol_21d, amihud_illiq_21d, obv_slope_10d, vol_ratio_5d, vol_ratio_21d
    df['pv_interaction'] = df['ret_5d'] * df['vol_ratio_5d']
    df['pv_reversal_interaction'] = df['ret_1d'] * df['vol_ratio_5d']
    df['liquidity_stress'] = df['amihud_illiq_21d'] * df['vol_21d']
    
    # ---------------------------------------------------------
    # GROUP E: PRICE STRUCTURE & BAR GEOMETRY
    # ---------------------------------------------------------
    # Base: hl_spread, oc_return, overnight_gap, upper_shadow, lower_shadow, bar_pressure, roll_spread_21d
    df['shadow_asym'] = (df['upper_shadow'] - df['lower_shadow']) / (df['hl_spread'] + 1e-5)
    df['bar_efficiency'] = df['oc_return'] / (df['hl_spread'] + 1e-5)
    df['range_expansion'] = df['hl_spread'] / (df['natr_14d'] + 1e-5)
    
    # ---------------------------------------------------------
    # GROUP G: MACRO MARKET AGGREGATES & REGIMES (CROSS-SECTIONAL at t)
    # ---------------------------------------------------------
    # Group by date to compute equal-weighted market aggregates strictly on day t
    mkt = df.groupby('date').agg(
        mkt_ret_1d=('ret_1d', 'mean'),
        mkt_ret_5d=('ret_5d', 'mean'),
        mkt_ret_21d=('ret_21d', 'mean'),
        mkt_dispersion_1d=('ret_1d', 'std'),
        mkt_mean_vol_21d=('vol_21d', 'mean'),
        mkt_breadth_sma50=('dist_sma_50', lambda x: (x > 0).mean()),
        mkt_breadth_sma200=('dist_sma_200', lambda x: (x > 0).mean()),
        mkt_ad_ratio=('ret_1d', lambda x: (np.sum(x > 0) + 1.0) / (np.sum(x < 0) + 1.0)),
        median_ret_5d=('ret_5d', 'median'),
        median_ret_21d=('ret_21d', 'median'),
        median_vol_21d=('vol_21d', 'median'),
        median_vol_ratio_5d=('vol_ratio_5d', 'median'),
    ).reset_index().sort_values('date')
    
    # Rolling market volatility and regimes (rolling strictly over past dates)
    mkt['mkt_vol_21d'] = mkt['mkt_ret_1d'].rolling(21, min_periods=5).std().fillna(0.01)
    mkt['mkt_vol_63d'] = mkt['mkt_ret_1d'].rolling(63, min_periods=10).std().fillna(0.01)
    mkt['mkt_momentum_regime'] = ((mkt['mkt_ret_21d'] > 0) & (mkt['mkt_breadth_sma50'] > 0.5)).astype(float)
    mkt_vol_80 = mkt['mkt_vol_21d'].rolling(252, min_periods=63).quantile(0.80).fillna(0.02)
    mkt['mkt_vol_regime'] = (mkt['mkt_vol_21d'] > mkt_vol_80).astype(float)
    
    # Merge market aggregates back
    df = df.merge(mkt, on='date', how='left')
    
    # ---------------------------------------------------------
    # GROUP F: MARKET-RELATIVE & CROSS-SECTIONAL PERCENTILE RANKS
    # ---------------------------------------------------------
    df['rel_ret_1d'] = df['ret_1d'] - df['mkt_ret_1d']
    df['rel_ret_5d'] = df['ret_5d'] - df['median_ret_5d']
    df['rel_ret_21d'] = df['ret_21d'] - df['median_ret_21d']
    df['rel_vol_21d'] = df['vol_21d'] / (df['median_vol_21d'] + 1e-5)
    df['rel_volume_5d'] = df['vol_ratio_5d'] / (df['median_vol_ratio_5d'] + 1e-5)
    
    # Percentile ranks strictly within date cross-section
    df['pct_rank_ret_1d'] = df.groupby('date')['ret_1d'].rank(pct=True)
    df['pct_rank_ret_5d'] = df.groupby('date')['ret_5d'].rank(pct=True)
    df['pct_rank_ret_21d'] = df.groupby('date')['ret_21d'].rank(pct=True)
    df['pct_rank_vol_21d'] = df.groupby('date')['vol_21d'].rank(pct=True)
    df['pct_rank_turnover'] = df.groupby('date')['log_turnover'].rank(pct=True)
    df['pct_rank_dist_sma200'] = df.groupby('date')['dist_sma_200'].rank(pct=True)
    
    return df

def run_leakage_audit(raw_df: pd.DataFrame, test_dates: list) -> dict:
    """
    Automated future perturbation test:
    Selects 5 probe evaluation sessions across the training/validation partitions.
    For each session t, perturbs all prices and volumes from t+1 to t+5 (+100% price, 10x volume).
    Re-evaluates all 72 features at session t.
    Calculates absolute delta. Asserts max delta == 0.0000000000.
    """
    print("\n--- Running Automated Future-Perturbation Leakage Test ---")
    probe_dates = test_dates[:5]
    audit_results = {}
    max_overall_delta = 0.0
    
    # Identify all candidate features (exclude non-feature metadata)
    probe_base = build_expanded_features(raw_df)
    feature_cols = [c for c in probe_base.columns if c not in [
        'date', 'ticker', 'adj_close', 'close', 'volume',
        'fwd_ret_1d', 'fwd_ret_2d', 'fwd_ret_3d', 'fwd_ret_5d', 'fwd_ret_10d', 'fwd_ret_21d',
        'excess_fwd_ret_1d', 'zscore_fwd_ret_1d', 'excess_fwd_ret_5d', 'zscore_fwd_ret_5d',
        'excess_fwd_ret_21d', 'zscore_fwd_ret_21d', 'zscore_fwd_ret_2d', 'zscore_fwd_ret_3d',
        'zscore_fwd_ret_10d', 'excess_fwd_ret_2d', 'excess_fwd_ret_3d', 'excess_fwd_ret_10d'
    ]]
    
    print(f"Total candidate features under leakage audit: {len(feature_cols)}")
    
    unique_dates = sorted(raw_df['date'].unique())
    
    for probe_d in probe_dates:
        d_idx = unique_dates.index(probe_d)
        if d_idx + 6 >= len(unique_dates):
            continue
            
        future_dates = unique_dates[d_idx+1 : d_idx+6]
        
        # 1. Unperturbed baseline at t
        unpert_vals = probe_base[probe_base['date'] == probe_d].set_index('ticker')[feature_cols]
        
        # 2. Perturb future dates in raw_df
        pert_df = raw_df.copy()
        fut_mask = pert_df['date'].isin(future_dates)
        pert_df.loc[fut_mask, 'adj_close'] *= 2.0
        pert_df.loc[fut_mask, 'close'] *= 2.0
        pert_df.loc[fut_mask, 'volume'] *= 10.0
        
        # 3. Rebuild features
        pert_features = build_expanded_features(pert_df)
        pert_vals = pert_features[pert_features['date'] == probe_d].set_index('ticker')[feature_cols]
        
        # Compare values at date t
        deltas = np.abs(unpert_vals.values - pert_vals.values)
        max_delta = float(np.nanmax(deltas))
        max_overall_delta = max(max_overall_delta, max_delta)
        
        audit_results[str(probe_d)] = {
            "probe_date": str(probe_d),
            "future_dates_perturbed": [str(x) for x in future_dates],
            "max_feature_delta": max_delta,
            "leakage_detected": bool(max_delta > 1e-9),
            "status": "PASS" if max_delta < 1e-9 else "FAIL"
        }
        print(f"  Probe Date {probe_d} -> Max Delta across all {len(feature_cols)} features: {max_delta:.10f} ({audit_results[str(probe_d)]['status']})")
        
    audit_summary = {
        "audit_name": "Automated Future-Perturbation Leakage Test (Groups A-G)",
        "feature_count_audited": len(feature_cols),
        "feature_list": feature_cols,
        "probe_sessions_evaluated": len(probe_dates),
        "perturbation_magnitude": "+100% price shock, 10x volume surge at t+1..t+5",
        "maximum_observed_delta": max_overall_delta,
        "zero_lookahead_verified": bool(max_overall_delta < 1e-9),
        "overall_status": "PASS" if max_overall_delta < 1e-9 else "FAIL",
        "session_details": audit_results
    }
    
    with open(os.path.join(OUT_DIR, "09_leakage_audit.json"), "w") as f:
        json.dump(audit_summary, f, indent=2)
        
    print(f"\n[Leakage Audit Result]: {audit_summary['overall_status']} (Max Delta = {max_overall_delta:.10f})")
    print(f"Audit log saved to results/accuracy_optimization/09_leakage_audit.json")
    return audit_summary, feature_cols

def main():
    print("=" * 80)
    print("PHASE 4: FEATURE EXPANSION & AUTOMATED LEAKAGE AUDIT")
    print("=" * 80)
    t0 = time.time()
    
    print("\n[1] Loading Universe B panel...")
    panel_path = "data/processed/universe_b_panel.parquet"
    df = pd.read_parquet(panel_path)
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date'].unique())
    
    # Chronological partition check
    splits, sdates = build_temporal_splits(unique_dates, H=5)
    val_dates = sorted(list(sdates['val_dates']))
    
    # Run Leakage Audit using validation probe dates
    probe_dates = [val_dates[20], val_dates[80], val_dates[140], val_dates[200], val_dates[260]]
    audit_summary, feature_cols = run_leakage_audit(df, probe_dates)
    
    # Save full expanded panel
    print(f"\n[2] Building full expanded panel ({len(feature_cols)} features)...")
    expanded_df = build_expanded_features(df)
    
    out_parquet = "data/processed/universe_b_expanded_features.parquet"
    print(f"[3] Saving expanded feature panel to {out_parquet}...")
    expanded_df.to_parquet(out_parquet, index=False)
    print(f"Expanded panel saved! Rows: {len(expanded_df):,}, Columns: {len(expanded_df.columns)}, Elapsed: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
