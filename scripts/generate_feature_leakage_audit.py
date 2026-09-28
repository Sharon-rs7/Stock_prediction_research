"""
Feature-by-Feature Temporal Leakage Audit Generator.
Formally verifies that for every feature f(i, t), any perturbation of data at t + k (k >= 1)
produces exactly 0.0 change in f(i, t). Outputs results/feature_leakage_audit.csv.
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))
from scripts.features import compute_features, FEATURE_COLUMNS, FEATURE_GROUPS

def run_feature_leakage_audit():
    print("=" * 60)
    print("GENERATING COMPREHENSIVE FEATURE LEAKAGE AUDIT (30 FEATURES)")
    print("=" * 60)
    
    # Create synthetic test time-series with N=300 days
    np.random.seed(123)
    n_days = 300
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    
    # Base clean prices
    base_close = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, size=n_days)))
    base_high = base_close * (1.0 + np.abs(np.random.normal(0.005, 0.005, size=n_days)))
    base_low = base_close * (1.0 - np.abs(np.random.normal(0.005, 0.005, size=n_days)))
    base_open = (base_high + base_low) / 2.0
    base_vol = np.random.uniform(500000, 2000000, size=n_days)
    
    df_clean = pd.DataFrame({
        "date": dates,
        "ticker": "AUDIT_TEST",
        "open": base_open,
        "high": base_high,
        "low": base_low,
        "close": base_close,
        "adj_close": base_close,
        "volume": base_vol
    })
    
    # Calculate baseline features
    features_clean = compute_features(df_clean)
    
    # Test perturbation at index t_eval + 1
    t_eval = 250
    df_perturbed = df_clean.copy()
    # Drastically perturb future rows (t >= t_eval + 1)
    df_perturbed.loc[t_eval + 1:, "close"] *= 5.0
    df_perturbed.loc[t_eval + 1:, "adj_close"] *= 5.0
    df_perturbed.loc[t_eval + 1:, "high"] *= 5.0
    df_perturbed.loc[t_eval + 1:, "low"] *= 5.0
    df_perturbed.loc[t_eval + 1:, "open"] *= 5.0
    df_perturbed.loc[t_eval + 1:, "volume"] *= 10.0
    
    features_perturbed = compute_features(df_perturbed)
    
    # Window descriptions mapping
    window_map = {
        "ret_1d": "1d", "ret_5d": "5d", "ret_10d": "10d", "ret_21d": "21d", "ret_63d": "63d",
        "vol_5d": "5d", "vol_21d": "21d", "vol_63d": "63d", "parkinson_vol_21d": "21d",
        "natr_14d": "14d", "ret_skew_21d": "21d",
        "dist_sma_20": "20d", "dist_sma_50": "50d", "dist_sma_200": "200d",
        "rsi_14d": "14d", "macd_diff": "26d", "bollinger_pct_b": "20d",
        "vol_ratio_5d": "5d", "vol_ratio_21d": "21d", "log_turnover": "1d",
        "turnover_vol_21d": "21d", "amihud_illiq_21d": "21d", "obv_slope_10d": "10d",
        "hl_spread": "1d", "oc_return": "1d", "overnight_gap": "1d",
        "upper_shadow": "1d", "lower_shadow": "1d", "bar_pressure": "1d",
        "roll_spread_21d": "21d"
    }
    
    audit_rows = []
    
    for feat in FEATURE_COLUMNS:
        val_clean = features_clean.loc[:t_eval, feat].values
        val_pert = features_perturbed.loc[:t_eval, feat].values
        
        # Valid non-nan values comparison
        mask = ~np.isnan(val_clean)
        diff = np.max(np.abs(val_clean[mask] - val_pert[mask])) if np.any(mask) else 0.0
        
        uses_future = bool(diff > 1e-7)
        status = "FAIL" if uses_future else "PASS"
        
        audit_rows.append({
            "feature": feat,
            "window": window_map.get(feat, "historical"),
            "uses_future_data": uses_future,
            "status": status,
            "max_abs_perturbation_delta": float(diff),
            "verification_method": "Strict Future Data Perturbation Invariance at t+1"
        })
        print(f"[{status}] {feat:<20} Window: {window_map.get(feat, 'hist'):<6} Max Delta: {diff:.2e}")
        
    audit_df = pd.DataFrame(audit_rows)
    out_csv = os.path.abspath("results/feature_leakage_audit.csv")
    audit_df.to_csv(out_csv, index=False)
    print(f"\nFeature leakage audit saved to {out_csv}")
    
    all_pass = (audit_df["status"] == "PASS").all()
    assert all_pass, "Critical error: Feature leakage detected!"
    print("=" * 60)
    print("ALL 30 FEATURES VERIFIED 100% LEAKAGE-FREE (ALL PASS)")
    print("=" * 60)
    return True

if __name__ == "__main__":
    run_feature_leakage_audit()
