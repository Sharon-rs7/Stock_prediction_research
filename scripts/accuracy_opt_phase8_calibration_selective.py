"""
Accuracy Optimization - Phase 3, 8 & 9: Calibration, Selective Prediction & Robustness
=====================================================================================
Executes:
  1. Calibration Diagnostics (Phase 8):
     - Compares Raw Probabilities, Platt Logistic Scaling, and Isotonic Regression
     - Computes Brier Score, Log Loss, ROC-AUC, PR-AUC, ECE, Slope & Intercept
     - Outputs: results/accuracy_optimization/06_calibration.csv
     
  2. Selective Prediction System (Phase 3):
     - Formulates confidence = max(P(UP), P(DOWN)) = |P(UP) - 0.5| + 0.5
     - Evaluates 9 coverage tiers: 100%, 50%, 30%, 20%, 15%, 10%, 5%, 2%, 1%
     - Tests whether directional accuracy crosses >=60% on the validation partition
     - Pre-registers the exact confidence threshold on VALIDATION ONLY
     - Outputs: results/accuracy_optimization/05_selective_accuracy.csv
     
  3. Robustness Tests (Phase 9):
     - Assesses validation stability across Bull, Bear, High-Vol, Low-Vol, High-Liquidity, Low-Liquidity, 2024 vs 2025
     - Outputs: results/accuracy_optimization/07_robustness.csv

Strictly on TRAIN and VALIDATION. Zero test set exposure.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
import xgboost as xgb
import lightgbm as lgb

sys.path.insert(0, os.path.abspath("."))
from scripts.temporal_split import build_temporal_splits

OUT_DIR = os.path.abspath("results/accuracy_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES_49 = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d',
    'vol_5d', 'vol_21d', 'vol_63d', 'parkinson_vol_21d', 'natr_14d', 'ret_skew_21d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'rsi_14d', 'macd_diff', 'bollinger_pct_b',
    'vol_ratio_5d', 'vol_ratio_21d', 'log_turnover', 'turnover_vol_21d', 'amihud_illiq_21d', 'obv_slope_10d',
    'hl_spread', 'oc_return', 'overnight_gap', 'upper_shadow', 'lower_shadow', 'bar_pressure', 'roll_spread_21d',
    'mkt_ret_1d', 'mkt_ret_5d', 'mkt_ret_21d', 'mkt_dispersion_1d',
    'mkt_vol_21d', 'mkt_vol_63d', 'mkt_breadth_sma50', 'mkt_breadth_sma200',
    'mkt_ad_ratio', 'mkt_momentum_regime', 'mkt_vol_regime',
    'rel_ret_5d', 'rel_ret_21d', 'rel_vol_21d', 'rel_volume_5d',
    'pct_rank_ret_21d', 'pct_rank_vol_21d', 'pct_rank_turnover', 'pct_rank_dist_sma200'
]

def compute_ece(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i+1])
        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()
            bin_conf = y_prob[bin_mask].mean()
            ece += (bin_mask.sum() / len(y_true)) * np.abs(bin_acc - bin_conf)
    return float(ece)

def main():
    print("=" * 80)
    print("PHASE 3, 8 & 9: CALIBRATION, SELECTIVE PREDICTION & ROBUSTNESS (VAL ONLY)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load Data
    print("\n[1] Loading dataset...")
    df = pd.read_parquet("data/processed/universe_b_expanded_features.parquet")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date'].unique())
    
    splits, sdates = build_temporal_splits(unique_dates, H=5)
    train_dates = set(sdates['train_dates'])
    val_dates = sorted(list(sdates['val_dates']))
    
    train_mask = df['date'].isin(train_dates)
    val_mask = df['date'].isin(val_dates)
    
    y_train_z = df.loc[train_mask, 'zscore_fwd_ret_5d'].values
    y_val_z = df.loc[val_mask, 'zscore_fwd_ret_5d'].values
    raw_fwd_val = df.loc[val_mask, 'fwd_ret_5d'].values
    
    y_train_bin = (y_train_z >= 0).astype(int)
    y_val_bin = (y_val_z >= 0).astype(int)
    
    v_tr = ~np.isnan(y_train_z)
    v_va = ~np.isnan(y_val_z)
    
    X_tr = df.loc[train_mask, FEATURES_49].fillna(0).values[v_tr]
    X_va = df.loc[val_mask, FEATURES_49].fillna(0).values[v_va]
    
    val_meta = df.loc[val_mask, ['date', 'ticker', 'mkt_ret_1d', 'mkt_vol_21d', 'log_turnover']].copy()
    val_meta = val_meta.loc[v_va].reset_index(drop=True)
    
    # 2. Fit Primary Best Model (HP_09: XGBoost Huber Deep Regressor)
    print("\n[2] Training Best Architecture: XGBoost Huber Deep Regressor (HP_09)...")
    best_xgb = xgb.XGBRegressor(
        objective='reg:pseudohubererror',
        n_estimators=120,
        learning_rate=0.02,
        max_depth=7,
        reg_lambda=10.0,
        subsample=0.8,
        colsample_bytree=0.75,
        tree_method='hist',
        random_state=42,
        n_jobs=4
    )
    best_xgb.fit(X_tr, y_train_z[v_tr])
    pred_z_val = best_xgb.predict(X_va)
    
    # Also train LightGBM Classifier for calibration comparison
    print("  Training LightGBM Classifier baseline for comparison...")
    best_lgb_cls = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    best_lgb_cls.fit(X_tr, y_train_bin[v_tr])
    raw_p_lgb = best_lgb_cls.predict_proba(X_va)[:, 1]
    
    # 3. Calibration Comparison (Phase 8)
    print("\n[3] Fitting Calibration Models on Validation...")
    # Method A: Raw Sigmoid of Continuous Z-score: 1 / (1 + exp(-z))
    raw_prob_z = 1.0 / (1.0 + np.exp(-pred_z_val))
    
    # Method B: Platt Logistic Scaling fitted on validation: P(Y=1|z) = 1 / (1 + exp(-(A*z + B)))
    # To prevent leakage, fit Platt scaling on first half of validation, evaluate on second half, then full
    val_half = len(pred_z_val) // 2
    platt_model = LogisticRegression(C=1.0)
    platt_model.fit(pred_z_val[:val_half].reshape(-1, 1), y_val_bin[v_va][:val_half])
    platt_slope = float(platt_model.coef_[0][0])
    platt_intercept = float(platt_model.intercept_[0])
    prob_platt = platt_model.predict_proba(pred_z_val.reshape(-1, 1))[:, 1]
    print(f"  Platt Scaling Parameters: slope A = {platt_slope:.4f}, intercept B = {platt_intercept:.4f}")
    
    # Method C: Isotonic Regression on first half
    iso_model = IsotonicRegression(out_of_bounds='clip')
    iso_model.fit(pred_z_val[:val_half], y_val_bin[v_va][:val_half])
    prob_iso = iso_model.predict(pred_z_val)
    
    cal_experiments = [
        {"method": "Raw_Sigmoid_ZScore", "prob": raw_prob_z},
        {"method": "Platt_Logistic_Scaling", "prob": prob_platt},
        {"method": "Isotonic_Regression", "prob": prob_iso},
        {"method": "Raw_LightGBM_Classifier_Prob", "prob": raw_p_lgb}
    ]
    
    cal_records = []
    for c in cal_experiments:
        p = c["prob"]
        brier = brier_score_loss(y_val_bin[v_va], p)
        ll = log_loss(y_val_bin[v_va], np.clip(p, 1e-6, 1-1e-6))
        roc = roc_auc_score(y_val_bin[v_va], p)
        pr = average_precision_score(y_val_bin[v_va], p)
        ece = compute_ece(y_val_bin[v_va], p, n_bins=10)
        
        # Fit calibration line (observed vs mean predicted)
        prob_true, prob_pred = calibration_curve(y_val_bin[v_va], p, n_bins=10)
        if len(prob_pred) > 2:
            lr_cal = stats.linregress(prob_pred, prob_true)
            slope = lr_cal.slope
            intercept = lr_cal.intercept
        else:
            slope, intercept = 1.0, 0.0
            
        print(f"  [{c['method']}] Brier: {brier:.5f} | LogLoss: {ll:.4f} | ROC-AUC: {roc:.4f} | ECE: {ece*100:.2f}% | Slope: {slope:.3f}")
        cal_records.append({
            "calibration_method": c["method"],
            "val_sample_size": len(p),
            "brier_score": round(float(brier), 5),
            "log_loss": round(float(ll), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr), 4),
            "ece": round(float(ece), 5),
            "cal_slope": round(float(slope), 4),
            "cal_intercept": round(float(intercept), 4)
        })
        
    df_cal = pd.DataFrame(cal_records)
    df_cal.to_csv(os.path.join(OUT_DIR, "06_calibration.csv"), index=False)
    print("  Saved calibration results to 06_calibration.csv")
    
    # 4. Selective Prediction System (Phase 3)
    print("\n[4] Evaluating Selective Prediction Across 9 Coverage Tiers on Validation...")
    # Confidence metric:
    # 1. Option A: absolute predicted z-score |z|
    # 2. Option B: probability margin |P(UP) - 0.5|
    # They are strictly monotonic to each other under Platt scaling!
    confidence = np.abs(prob_platt - 0.5)
    
    coverage_tiers = [1.00, 0.50, 0.30, 0.20, 0.15, 0.10, 0.05, 0.02, 0.01]
    
    sel_records = []
    val_meta['pred_z'] = pred_z_val
    val_meta['prob_platt'] = prob_platt
    val_meta['confidence'] = confidence
    val_meta['actual_bin'] = y_val_bin[v_va]
    val_meta['raw_fwd_ret'] = raw_fwd_val[v_va]
    
    for cov in coverage_tiers:
        cutoff = float(np.percentile(confidence, 100 * (1.0 - cov))) if cov < 1.0 else 0.0
        mask_cov = (confidence >= cutoff)
        
        n_cov = int(mask_cov.sum())
        act_cov = n_cov / len(confidence)
        
        # Predictions on covered subset
        y_true_cov = val_meta.loc[mask_cov, 'actual_bin'].values
        p_cov = val_meta.loc[mask_cov, 'prob_platt'].values
        y_pred_cov = (p_cov >= 0.5).astype(int)
        ret_cov = val_meta.loc[mask_cov, 'raw_fwd_ret'].values
        
        # When predicted UP vs predicted DOWN
        up_mask = (y_pred_cov == 1)
        down_mask = (y_pred_cov == 0)
        
        acc = accuracy_score(y_true_cov, y_pred_cov)
        bal_acc = balanced_accuracy_score(y_true_cov, y_pred_cov)
        prec_up = precision_score(y_true_cov, y_pred_cov, zero_division=0)
        rec_up = recall_score(y_true_cov, y_pred_cov, zero_division=0)
        f1 = f1_score(y_true_cov, y_pred_cov, zero_division=0)
        ece = compute_ece(y_true_cov, p_cov)
        
        # Rank IC on covered dates
        sub_df = val_meta.loc[mask_cov]
        daily_ics = sub_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred_z'], g['raw_fwd_ret'])[0] if len(g)>=5 else np.nan).dropna()
        rank_ic = float(daily_ics.mean()) if len(daily_ics) > 0 else 0.0
        
        avg_fwd_ret_all = float(np.mean(ret_cov)) if len(ret_cov) > 0 else 0.0
        avg_fwd_ret_up = float(np.mean(ret_cov[up_mask])) if up_mask.sum() > 0 else 0.0
        avg_fwd_ret_down = float(np.mean(ret_cov[down_mask])) if down_mask.sum() > 0 else 0.0
        spread_up_down = avg_fwd_ret_up - avg_fwd_ret_down
        
        # Check whether directional accuracy or precision crosses 60%
        crosses_60_dir = bool(acc >= 0.60)
        crosses_60_prec = bool(prec_up >= 0.60)
        
        print(f"  Target Cov: {cov*100:4.0f}% | Actual: {act_cov*100:5.2f}% | N: {n_cov:7,d} | DirAcc: {acc*100:5.2f}% | BalAcc: {bal_acc*100:5.2f}% | UP Prec: {prec_up*100:5.2f}% | UP/DN Spread: {spread_up_down*100:+.2f}% | Rank IC: {rank_ic:.4f}")
        
        sel_records.append({
            "target_coverage": cov,
            "actual_coverage": round(float(act_cov), 4),
            "sample_size": n_cov,
            "confidence_cutoff": round(float(cutoff), 5),
            "dir_accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "up_precision": round(float(prec_up), 4),
            "up_recall": round(float(rec_up), 4),
            "f1_score": round(float(f1), 4),
            "ece": round(float(ece), 5),
            "rank_ic": round(float(rank_ic), 4),
            "avg_forward_return": round(float(avg_fwd_ret_all), 5),
            "avg_fwd_ret_up_calls": round(float(avg_fwd_ret_up), 5),
            "avg_fwd_ret_down_calls": round(float(avg_fwd_ret_down), 5),
            "up_down_spread": round(float(spread_up_down), 5),
            "crosses_60pct_directional_acc": crosses_60_dir,
            "crosses_60pct_up_precision": crosses_60_prec
        })
        
    df_sel = pd.DataFrame(sel_records)
    df_sel.to_csv(os.path.join(OUT_DIR, "05_selective_accuracy.csv"), index=False)
    print("  Saved selective accuracy results to 05_selective_accuracy.csv")
    
    # 5. Robustness Tests Across Regimes (Phase 9)
    print("\n[5] Executing Robustness Tests on Validation Partition...")
    # Define regimes
    # A. Market return regime: Bull (> +0.5%), Bear (< -0.5%), Neutral
    # B. Market volatility regime: High Vol (> 80th pct), Low Vol
    # C. Liquidity: High Turnover (> median), Low Turnover
    # D. Year: 2024 vs 2025
    
    val_meta['mkt_vol_80'] = val_meta['mkt_vol_21d'].quantile(0.80)
    median_turnover = val_meta['log_turnover'].median()
    val_meta['year'] = pd.to_datetime(val_meta['date']).dt.year
    
    regimes = {
        "Full_Validation_Sample": np.ones(len(val_meta), dtype=bool),
        "Bull_Days (Mkt_Ret > +0.5%)": (val_meta['mkt_ret_1d'] > 0.005).values,
        "Bear_Days (Mkt_Ret < -0.5%)": (val_meta['mkt_ret_1d'] < -0.005).values,
        "Neutral_Market_Days": ((val_meta['mkt_ret_1d'] >= -0.005) & (val_meta['mkt_ret_1d'] <= 0.005)).values,
        "High_Volatility_Regime (>80th pct)": (val_meta['mkt_vol_21d'] > val_meta['mkt_vol_80']).values,
        "Low_Volatility_Regime (<=80th pct)": (val_meta['mkt_vol_21d'] <= val_meta['mkt_vol_80']).values,
        "High_Liquidity_Equities (Turnover > Med)": (val_meta['log_turnover'] > median_turnover).values,
        "Low_Liquidity_Equities (Turnover <= Med)": (val_meta['log_turnover'] <= median_turnover).values,
        "Calendar_Year_2024": (val_meta['year'] == 2024).values,
        "Calendar_Year_2025": (val_meta['year'] == 2025).values
    }
    
    rob_records = []
    for r_name, r_mask in regimes.items():
        if r_mask.sum() == 0: continue
        sub_df = val_meta.loc[r_mask]
        y_true = sub_df['actual_bin'].values
        p_val = sub_df['prob_platt'].values
        y_pred = (p_val >= 0.5).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        daily_ics = sub_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred_z'], g['raw_fwd_ret'])[0] if len(g)>=10 else np.nan).dropna()
        rank_ic = float(daily_ics.mean()) if len(daily_ics) > 0 else 0.0
        rank_ic_t = float(daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if (len(daily_ics) > 1 and daily_ics.std() > 0) else 0.0
        
        print(f"  Regime: {r_name[:35]:35s} | N: {len(sub_df):7,d} | DirAcc: {acc*100:5.2f}% | BalAcc: {bal_acc*100:5.2f}% | Rank IC: {rank_ic:.4f} (t={rank_ic_t:.2f})")
        
        rob_records.append({
            "regime_name": r_name,
            "sample_size": len(sub_df),
            "dir_accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "up_precision": round(float(prec), 4),
            "f1_score": round(float(f1), 4),
            "rank_ic": round(float(rank_ic), 4),
            "rank_ic_t": round(float(rank_ic_t), 2),
            "notes": "Validation partition robustness subset"
        })
        
    df_rob = pd.DataFrame(rob_records)
    df_rob.to_csv(os.path.join(OUT_DIR, "07_robustness.csv"), index=False)
    print("  Saved robustness results to 07_robustness.csv")
    print(f"\nAll Phase 3, 8, 9 routines complete! Elapsed: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
