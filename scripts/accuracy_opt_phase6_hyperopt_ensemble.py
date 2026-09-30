"""
Accuracy Optimization - Phase 6 & 7: Hyperparameter Tuning & Ensembling
========================================================================
Executes systematic hyperparameter optimization across time-series validation windows:
  - 10 distinct architectural configurations across LightGBM Huber, LightGBM Classifier, and XGBoost Huber
  - Evaluated on 3 distinct chronological validation sub-windows + full validation partition
  - Measures temporal stability across regimes
  - Tests 4 multi-model ensembling techniques: Probability Averaging, Rank Averaging, 3-Way Blend, and Weighted Blend

Strictly on TRAIN and VALIDATION. Zero test set exposure.
Outputs: results/accuracy_optimization/04_hyperparameter_search.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss
)
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.abspath("."))
from scripts.temporal_split import build_temporal_splits

OUT_DIR = os.path.abspath("results/accuracy_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

# 49 market-aware features (confirmed optimal in Phase 5)
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

def main():
    print("=" * 80)
    print("PHASE 6 & 7: HYPERPARAMETER TUNING & ENSEMBLING (TRAIN & VAL ONLY)")
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
    
    # Define 3 chronological validation windows (early, mid, late)
    n_val = len(val_dates)
    w1_dates = set(val_dates[: n_val // 3])
    w2_dates = set(val_dates[n_val // 3 : 2 * (n_val // 3)])
    w3_dates = set(val_dates[2 * (n_val // 3) :])
    
    train_mask = df['date'].isin(train_dates)
    val_mask = df['date'].isin(val_dates)
    
    y_train_z = df.loc[train_mask, 'zscore_fwd_ret_5d'].values
    y_val_z = df.loc[val_mask, 'zscore_fwd_ret_5d'].values
    
    y_train_bin = (y_train_z >= 0).astype(int)
    y_val_bin = (y_val_z >= 0).astype(int)
    
    v_tr = ~np.isnan(y_train_z)
    v_va = ~np.isnan(y_val_z)
    
    X_tr = df.loc[train_mask, FEATURES_49].fillna(0).values[v_tr]
    X_va = df.loc[val_mask, FEATURES_49].fillna(0).values[v_va]
    
    val_dates_arr = df.loc[val_mask, 'date'].values[v_va]
    
    mask_w1 = np.isin(val_dates_arr, list(w1_dates))
    mask_w2 = np.isin(val_dates_arr, list(w2_dates))
    mask_w3 = np.isin(val_dates_arr, list(w3_dates))
    
    print(f"Train samples: {v_tr.sum():,}, Full Val: {v_va.sum():,}")
    print(f"Window 1: {mask_w1.sum():,} samples ({len(w1_dates)} days)")
    print(f"Window 2: {mask_w2.sum():,} samples ({len(w2_dates)} days)")
    print(f"Window 3: {mask_w3.sum():,} samples ({len(w3_dates)} days)")
    
    # 2. Hyperparameter Grid Configurations
    configs = [
        {
            "name": "HP_01_LGBM_Huber_Baseline",
            "type": "lgbm_reg",
            "params": {"objective": "huber", "huber_alpha": 1.0, "n_estimators": 100, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_02_LGBM_Huber_Deep_Reg",
            "type": "lgbm_reg",
            "params": {"objective": "huber", "huber_alpha": 1.0, "n_estimators": 120, "learning_rate": 0.02, "max_depth": 8, "num_leaves": 63, "min_child_samples": 50, "reg_alpha": 0.1, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_03_LGBM_Huber_Shallow",
            "type": "lgbm_reg",
            "params": {"objective": "huber", "huber_alpha": 0.8, "n_estimators": 100, "learning_rate": 0.05, "max_depth": 4, "num_leaves": 15, "min_child_samples": 20, "reg_lambda": 1.0, "subsample": 0.9, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_04_LGBM_Huber_Tail_Robust",
            "type": "lgbm_reg",
            "params": {"objective": "huber", "huber_alpha": 0.5, "n_estimators": 100, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "min_child_samples": 100, "reg_lambda": 5.0, "subsample": 0.8, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_05_LGBM_Huber_Slow_Learn",
            "type": "lgbm_reg",
            "params": {"objective": "huber", "huber_alpha": 1.2, "n_estimators": 150, "learning_rate": 0.015, "max_depth": 6, "num_leaves": 31, "min_child_samples": 50, "reg_lambda": 2.0, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_06_LGBM_Binary_Classifier",
            "type": "lgbm_cls",
            "params": {"objective": "binary", "n_estimators": 100, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 4, "verbose": -1}
        },
        {
            "name": "HP_07_XGB_Huber_Baseline",
            "type": "xgb_reg",
            "params": {"objective": "reg:pseudohubererror", "n_estimators": 100, "learning_rate": 0.03, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "random_state": 42, "n_jobs": 4}
        },
        {
            "name": "HP_08_XGB_Huber_Shallow",
            "type": "xgb_reg",
            "params": {"objective": "reg:pseudohubererror", "n_estimators": 100, "learning_rate": 0.04, "max_depth": 4, "reg_lambda": 5.0, "subsample": 0.85, "colsample_bytree": 0.7, "tree_method": "hist", "random_state": 42, "n_jobs": 4}
        },
        {
            "name": "HP_09_XGB_Huber_Deep_Reg",
            "type": "xgb_reg",
            "params": {"objective": "reg:pseudohubererror", "n_estimators": 120, "learning_rate": 0.02, "max_depth": 7, "reg_lambda": 10.0, "subsample": 0.8, "colsample_bytree": 0.75, "tree_method": "hist", "random_state": 42, "n_jobs": 4}
        },
        {
            "name": "HP_10_LGBM_Weighted_Classifier",
            "type": "lgbm_cls",
            "params": {"objective": "binary", "scale_pos_weight": 1.05, "n_estimators": 100, "learning_rate": 0.03, "max_depth": 6, "num_leaves": 31, "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42, "n_jobs": 4, "verbose": -1}
        }
    ]
    
    results = []
    trained_models = {}
    val_preds_dict = {}
    
    print("\n[2] Executing Hyperparameter Grid Searches...")
    for cfg in configs:
        t_cfg = time.time()
        c_name = cfg["name"]
        c_type = cfg["type"]
        c_params = cfg["params"]
        
        print(f"  Fitting {c_name}...")
        if c_type == "lgbm_reg":
            model = lgb.LGBMRegressor(**c_params)
            model.fit(X_tr, y_train_z[v_tr])
            preds_val = model.predict(X_va)
            bin_preds = (preds_val >= 0).astype(int)
        elif c_type == "lgbm_cls":
            model = lgb.LGBMClassifier(**c_params)
            model.fit(X_tr, y_train_bin[v_tr])
            preds_val = model.predict_proba(X_va)[:, 1]
            bin_preds = (preds_val >= 0.5).astype(int)
        elif c_type == "xgb_reg":
            model = xgb.XGBRegressor(**c_params)
            model.fit(X_tr, y_train_z[v_tr])
            preds_val = model.predict(X_va)
            bin_preds = (preds_val >= 0).astype(int)
            
        trained_models[c_name] = model
        val_preds_dict[c_name] = preds_val
        
        # Calculate full validation metrics
        acc_full = accuracy_score(y_val_bin[v_va], bin_preds)
        bal_full = balanced_accuracy_score(y_val_bin[v_va], bin_preds)
        prec_full = precision_score(y_val_bin[v_va], bin_preds, zero_division=0)
        f1_full = f1_score(y_val_bin[v_va], bin_preds, zero_division=0)
        
        # Sub-window metrics
        acc_w1 = accuracy_score(y_val_bin[v_va][mask_w1], bin_preds[mask_w1])
        acc_w2 = accuracy_score(y_val_bin[v_va][mask_w2], bin_preds[mask_w2])
        acc_w3 = accuracy_score(y_val_bin[v_va][mask_w3], bin_preds[mask_w3])
        stability_std = float(np.std([acc_w1, acc_w2, acc_w3]))
        
        # Rank IC across dates
        eval_df = pd.DataFrame({'date': val_dates_arr, 'pred': preds_val, 'actual': y_val_z[v_va]})
        daily_ics = eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        rank_ic = float(daily_ics.mean())
        rank_ic_t = float(daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0.0
        
        print(f"    Full Acc: {acc_full*100:.2f}% | W1: {acc_w1*100:.2f}% | W2: {acc_w2*100:.2f}% | W3: {acc_w3*100:.2f}% | Std: {stability_std*100:.2f}% | Rank IC: {rank_ic:.4f} (t={rank_ic_t:.2f}) [{time.time()-t_cfg:.1f}s]")
        
        results.append({
            "config_name": c_name,
            "architecture_type": c_type,
            "val_dir_accuracy": round(float(acc_full), 4),
            "window_1_acc": round(float(acc_w1), 4),
            "window_2_acc": round(float(acc_w2), 4),
            "window_3_acc": round(float(acc_w3), 4),
            "stability_std": round(float(stability_std), 4),
            "balanced_accuracy": round(float(bal_full), 4),
            "up_precision": round(float(prec_full), 4),
            "f1_score": round(float(f1_full), 4),
            "rank_ic": round(float(rank_ic), 4),
            "rank_ic_t": round(float(rank_ic_t), 2),
            "notes": "Single model configuration"
        })
        
    # 3. Model Ensembling Experiments
    print("\n[3] Testing Ensemble Combinations on Validation Partition...")
    # Best LightGBM: HP_01 (Baseline) / HP_04
    # Best XGBoost: HP_07 (Baseline)
    p_lgb = val_preds_dict["HP_01_LGBM_Huber_Baseline"]
    p_xgb = val_preds_dict["HP_07_XGB_Huber_Baseline"]
    p_lgb_cls = val_preds_dict["HP_06_LGBM_Binary_Classifier"]
    
    # Logistic baseline
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    lr = LogisticRegression(C=0.01, max_iter=200, random_state=42)
    lr.fit(X_tr_s, y_train_bin[v_tr])
    p_lr = lr.predict_proba(X_va_s)[:, 1]
    
    # Normalize regression preds to percentile ranks for ensembling
    eval_df['rank_lgb'] = eval_df.groupby('date')['pred'].rank(pct=True) # temp
    
    def to_daily_rank(scores, dates):
        s_df = pd.DataFrame({'date': dates, 's': scores})
        return s_df.groupby('date')['s'].rank(pct=True).values
        
    rank_lgb = to_daily_rank(p_lgb, val_dates_arr)
    rank_xgb = to_daily_rank(p_xgb, val_dates_arr)
    rank_lr = to_daily_rank(p_lr, val_dates_arr)
    
    ensembles = {
        "ENS_01_Dual_Huber_Rank_Avg (50% LGBM + 50% XGB)": 0.5 * rank_lgb + 0.5 * rank_xgb,
        "ENS_02_Dual_Huber_Score_Avg": 0.5 * p_lgb + 0.5 * p_xgb,
        "ENS_03_Tri_Model_Blend (45% LGB + 45% XGB + 10% LR)": 0.45 * rank_lgb + 0.45 * rank_xgb + 0.10 * rank_lr,
        "ENS_04_Classification_Regression_Hybrid (50% Reg + 50% Cls)": 0.5 * rank_lgb + 0.5 * to_daily_rank(p_lgb_cls, val_dates_arr)
    }
    
    for ens_name, ens_score in ensembles.items():
        # Direction from rank > 0.5 or score > 0
        bin_preds = (ens_score >= np.median(ens_score)).astype(int) if "Rank" in ens_name or "Blend" in ens_name or "Hybrid" in ens_name else (ens_score >= 0).astype(int)
        
        acc_full = accuracy_score(y_val_bin[v_va], bin_preds)
        bal_full = balanced_accuracy_score(y_val_bin[v_va], bin_preds)
        prec_full = precision_score(y_val_bin[v_va], bin_preds, zero_division=0)
        f1_full = f1_score(y_val_bin[v_va], bin_preds, zero_division=0)
        
        acc_w1 = accuracy_score(y_val_bin[v_va][mask_w1], bin_preds[mask_w1])
        acc_w2 = accuracy_score(y_val_bin[v_va][mask_w2], bin_preds[mask_w2])
        acc_w3 = accuracy_score(y_val_bin[v_va][mask_w3], bin_preds[mask_w3])
        stability_std = float(np.std([acc_w1, acc_w2, acc_w3]))
        
        eval_df['pred_ens'] = ens_score
        daily_ics = eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred_ens'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        rank_ic = float(daily_ics.mean())
        rank_ic_t = float(daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0.0
        
        print(f"  [{ens_name[:32]}] Full Acc: {acc_full*100:.2f}% | W1: {acc_w1*100:.2f}% | W2: {acc_w2*100:.2f}% | W3: {acc_w3*100:.2f}% | Rank IC: {rank_ic:.4f} (t={rank_ic_t:.2f})")
        
        results.append({
            "config_name": ens_name,
            "architecture_type": "ensemble_blend",
            "val_dir_accuracy": round(float(acc_full), 4),
            "window_1_acc": round(float(acc_w1), 4),
            "window_2_acc": round(float(acc_w2), 4),
            "window_3_acc": round(float(acc_w3), 4),
            "stability_std": round(float(stability_std), 4),
            "balanced_accuracy": round(float(bal_full), 4),
            "up_precision": round(float(prec_full), 4),
            "f1_score": round(float(f1_full), 4),
            "rank_ic": round(float(rank_ic), 4),
            "rank_ic_t": round(float(rank_ic_t), 2),
            "notes": "Multi-model ensemble on validation partition"
        })
        
    out_df = pd.DataFrame(results)
    out_path = os.path.join(OUT_DIR, "04_hyperparameter_search.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\n[4] Hyperparameter search & ensemble results saved to {out_path}")
    print(f"Elapsed time: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
