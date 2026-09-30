"""
Accuracy Optimization - Phase 5: Feature Selection & Model Comparison
=====================================================================
Evaluates 3 Feature Sets:
  1. Current 49-Feature Market-Aware Model
  2. Expanded 77-Feature Model (Groups A-G)
  3. Validation-Selected Feature Model (~40 top features after collinearity pruning)

Across Candidate Model Architectures:
  - Logistic Regression (Standardized Baseline)
  - LightGBM Huber Regressor (Continuous z-score)
  - LightGBM Binary Classifier (Cross-Entropy)
  - XGBoost Huber Regressor (CUDA/Hist)
  - XGBoost Binary Classifier (CUDA/Hist)
  - Calibrated LightGBM (Platt Scaling)

Strictly on TRAIN and VALIDATION. Zero test set exposure.
Outputs:
  - results/accuracy_optimization/02_feature_experiments.csv
  - results/accuracy_optimization/03_model_comparison.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss
)
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.abspath("."))
from scripts.features import FEATURE_COLUMNS
from scripts.temporal_split import build_temporal_splits

OUT_DIR = os.path.abspath("results/accuracy_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Feature Set Definitions
SET_1_49_FEATURES = [
    # 30 base OHLCV
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d',
    'vol_5d', 'vol_21d', 'vol_63d', 'parkinson_vol_21d', 'natr_14d', 'ret_skew_21d',
    'dist_sma_20', 'dist_sma_50', 'dist_sma_200', 'rsi_14d', 'macd_diff', 'bollinger_pct_b',
    'vol_ratio_5d', 'vol_ratio_21d', 'log_turnover', 'turnover_vol_21d', 'amihud_illiq_21d', 'obv_slope_10d',
    'hl_spread', 'oc_return', 'overnight_gap', 'upper_shadow', 'lower_shadow', 'bar_pressure', 'roll_spread_21d',
    # 19 market-aware
    'mkt_ret_1d', 'mkt_ret_5d', 'mkt_ret_21d', 'mkt_dispersion_1d',
    'mkt_vol_21d', 'mkt_vol_63d', 'mkt_breadth_sma50', 'mkt_breadth_sma200',
    'mkt_ad_ratio', 'mkt_momentum_regime', 'mkt_vol_regime',
    'rel_ret_5d', 'rel_ret_21d', 'rel_vol_21d', 'rel_volume_5d',
    'pct_rank_ret_21d', 'pct_rank_vol_21d', 'pct_rank_turnover', 'pct_rank_dist_sma200'
]

def main():
    print("=" * 80)
    print("PHASE 5: FEATURE SELECTION & MODEL COMPARISON (TRAIN & VAL ONLY)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load Expanded Feature Panel
    print("\n[1] Loading expanded feature panel...")
    df = pd.read_parquet("data/processed/universe_b_expanded_features.parquet")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date'].unique())
    
    splits, sdates = build_temporal_splits(unique_dates, H=5)
    train_dates = set(sdates['train_dates'])
    val_dates = set(sdates['val_dates'])
    
    train_mask = df['date'].isin(train_dates)
    val_mask = df['date'].isin(val_dates)
    
    # Target definition (H=5 pre-registered primary horizon)
    # Forward return z-score and binary direction
    y_train_z = df.loc[train_mask, 'zscore_fwd_ret_5d'].values
    y_val_z = df.loc[val_mask, 'zscore_fwd_ret_5d'].values
    
    y_train_bin = (y_train_z >= 0).astype(int)
    y_val_bin = (y_val_z >= 0).astype(int)
    
    # Valid rows mask
    v_tr = ~np.isnan(y_train_z)
    v_va = ~np.isnan(y_val_z)
    
    val_dates_series = df.loc[val_mask, 'date'].values[v_va]
    
    # All 77 candidate features
    with open("results/accuracy_optimization/09_leakage_audit.json") as f:
        leak_meta = json.load(f)
    SET_2_77_FEATURES = leak_meta['feature_list']
    print(f"Set 1 (Current Market-Aware): {len(SET_1_49_FEATURES)} features")
    print(f"Set 2 (Expanded Groups A-G): {len(SET_2_77_FEATURES)} features")
    
    # 2. Validation-Only Feature Selection (Set 3)
    print("\n[2] Performing Validation-Only Feature Selection on Set 2...")
    X_tr_full = df.loc[train_mask, SET_2_77_FEATURES].fillna(0).values[v_tr]
    X_va_full = df.loc[val_mask, SET_2_77_FEATURES].fillna(0).values[v_va]
    
    # Fit initial screening model on Train to measure split gain
    screener = lgb.LGBMRegressor(
        objective='huber',
        n_estimators=100,
        learning_rate=0.03,
        num_leaves=31,
        random_state=42,
        n_jobs=4,
        verbose=-1
    )
    screener.fit(X_tr_full, y_train_z[v_tr])
    importances = screener.feature_importances_
    
    # Check collinearity on Train sample (sample 50,000 rows for fast correlation matrix)
    sample_idx = np.random.choice(len(X_tr_full), min(50000, len(X_tr_full)), replace=False)
    corr_matrix = pd.DataFrame(X_tr_full[sample_idx], columns=SET_2_77_FEATURES).corr().abs()
    
    # Rank features by importance
    ranked_feats = [x for _, x in sorted(zip(importances, SET_2_77_FEATURES), reverse=True)]
    
    # Prune collinear features (|r| > 0.90)
    selected_features = []
    for feat in ranked_feats:
        # Check correlation with already selected features
        too_correlated = False
        for s_feat in selected_features:
            if corr_matrix.loc[feat, s_feat] > 0.90:
                too_correlated = True
                break
        if not too_correlated:
            selected_features.append(feat)
        if len(selected_features) >= 42:
            break
            
    print(f"Set 3 (Selected & Pruned): {len(selected_features)} features")
    print(f"Top 10 selected features: {selected_features[:10]}")
    
    feature_sets = {
        "Set_1_Current_49": SET_1_49_FEATURES,
        "Set_2_Expanded_77": SET_2_77_FEATURES,
        "Set_3_Selected_42": selected_features
    }
    
    # Save feature experiments overview
    feat_exp_records = []
    
    # 3. Model Benchmark Evaluation on Validation Partition
    model_records = []
    
    for set_name, f_cols in feature_sets.items():
        print(f"\n--- Evaluating Feature Configuration: {set_name} ({len(f_cols)} features) ---")
        
        X_tr = df.loc[train_mask, f_cols].fillna(0).values[v_tr]
        X_va = df.loc[val_mask, f_cols].fillna(0).values[v_va]
        
        # Standard scaler fitted on TRAIN ONLY
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_va_scaled = scaler.transform(X_va)
        
        # -------------------------------------------------------------
        # MODEL 1: Logistic Regression Baseline
        # -------------------------------------------------------------
        lr = LogisticRegression(C=0.01, max_iter=200, random_state=42)
        lr.fit(X_tr_scaled, y_train_bin[v_tr])
        p_lr = lr.predict_proba(X_va_scaled)[:, 1]
        pred_lr = (p_lr >= 0.5).astype(int)
        
        acc_lr = accuracy_score(y_val_bin[v_va], pred_lr)
        bal_lr = balanced_accuracy_score(y_val_bin[v_va], pred_lr)
        prec_lr = precision_score(y_val_bin[v_va], pred_lr, zero_division=0)
        rec_lr = recall_score(y_val_bin[v_va], pred_lr, zero_division=0)
        f1_lr = f1_score(y_val_bin[v_va], pred_lr, zero_division=0)
        roc_lr = roc_auc_score(y_val_bin[v_va], p_lr)
        pr_lr = average_precision_score(y_val_bin[v_va], p_lr)
        brier_lr = brier_score_loss(y_val_bin[v_va], p_lr)
        
        # Rank IC
        val_eval_df = pd.DataFrame({'date': val_dates_series, 'pred': p_lr, 'actual': y_val_z[v_va]})
        daily_ics = val_eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        ic_lr = daily_ics.mean()
        ic_t_lr = (daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0
        
        print(f"  [Logistic Regression] DirAcc: {acc_lr*100:.2f}% | BalAcc: {bal_lr*100:.2f}% | UP Prec: {prec_lr*100:.2f}% | Rank IC: {ic_lr:.4f} (t={ic_t_lr:.2f})")
        
        model_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "model_architecture": "Logistic_Regression",
            "val_sample_size": int(v_va.sum()),
            "dir_accuracy": round(float(acc_lr), 4),
            "balanced_accuracy": round(float(bal_lr), 4),
            "up_precision": round(float(prec_lr), 4),
            "up_recall": round(float(rec_lr), 4),
            "f1_score": round(float(f1_lr), 4),
            "roc_auc": round(float(roc_lr), 4),
            "pr_auc": round(float(pr_lr), 4),
            "brier_score": round(float(brier_lr), 5),
            "rank_ic": round(float(ic_lr), 4),
            "rank_ic_t": round(float(ic_t_lr), 2)
        })
        
        # -------------------------------------------------------------
        # MODEL 2: LightGBM Huber Regressor (Continuous z-score)
        # -------------------------------------------------------------
        lgb_reg = lgb.LGBMRegressor(
            objective='huber',
            huber_alpha=1.0,
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
        lgb_reg.fit(X_tr, y_train_z[v_tr])
        pred_z_val = lgb_reg.predict(X_va)
        pred_lgb_reg = (pred_z_val >= 0).astype(int)
        
        acc_lgb_r = accuracy_score(y_val_bin[v_va], pred_lgb_reg)
        bal_lgb_r = balanced_accuracy_score(y_val_bin[v_va], pred_lgb_reg)
        prec_lgb_r = precision_score(y_val_bin[v_va], pred_lgb_reg, zero_division=0)
        rec_lgb_r = recall_score(y_val_bin[v_va], pred_lgb_reg, zero_division=0)
        f1_lgb_r = f1_score(y_val_bin[v_va], pred_lgb_reg, zero_division=0)
        roc_lgb_r = roc_auc_score(y_val_bin[v_va], pred_z_val)
        pr_lgb_r = average_precision_score(y_val_bin[v_va], pred_z_val)
        
        val_eval_df['pred'] = pred_z_val
        daily_ics = val_eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        ic_lgb_r = daily_ics.mean()
        ic_t_lgb_r = (daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0
        
        # Platt calibration on regression predictions
        platt_lr = LogisticRegression(C=1.0)
        platt_lr.fit(pred_z_val.reshape(-1, 1), y_val_bin[v_va])
        prob_calibrated = platt_lr.predict_proba(pred_z_val.reshape(-1, 1))[:, 1]
        brier_lgb_r = brier_score_loss(y_val_bin[v_va], prob_calibrated)
        
        print(f"  [LightGBM Huber]     DirAcc: {acc_lgb_r*100:.2f}% | BalAcc: {bal_lgb_r*100:.2f}% | UP Prec: {prec_lgb_r*100:.2f}% | Rank IC: {ic_lgb_r:.4f} (t={ic_t_lgb_r:.2f})")
        
        model_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "model_architecture": "LightGBM_Huber_Regressor",
            "val_sample_size": int(v_va.sum()),
            "dir_accuracy": round(float(acc_lgb_r), 4),
            "balanced_accuracy": round(float(bal_lgb_r), 4),
            "up_precision": round(float(prec_lgb_r), 4),
            "up_recall": round(float(rec_lgb_r), 4),
            "f1_score": round(float(f1_lgb_r), 4),
            "roc_auc": round(float(roc_lgb_r), 4),
            "pr_auc": round(float(pr_lgb_r), 4),
            "brier_score": round(float(brier_lgb_r), 5),
            "rank_ic": round(float(ic_lgb_r), 4),
            "rank_ic_t": round(float(ic_t_lgb_r), 2)
        })
        
        # -------------------------------------------------------------
        # MODEL 3: LightGBM Binary Classifier
        # -------------------------------------------------------------
        lgb_cls = lgb.LGBMClassifier(
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
        lgb_cls.fit(X_tr, y_train_bin[v_tr])
        p_lgb_c = lgb_cls.predict_proba(X_va)[:, 1]
        pred_lgb_c = (p_lgb_c >= 0.5).astype(int)
        
        acc_lgb_c = accuracy_score(y_val_bin[v_va], pred_lgb_c)
        bal_lgb_c = balanced_accuracy_score(y_val_bin[v_va], pred_lgb_c)
        prec_lgb_c = precision_score(y_val_bin[v_va], pred_lgb_c, zero_division=0)
        rec_lgb_c = recall_score(y_val_bin[v_va], pred_lgb_c, zero_division=0)
        f1_lgb_c = f1_score(y_val_bin[v_va], pred_lgb_c, zero_division=0)
        roc_lgb_c = roc_auc_score(y_val_bin[v_va], p_lgb_c)
        pr_lgb_c = average_precision_score(y_val_bin[v_va], p_lgb_c)
        brier_lgb_c = brier_score_loss(y_val_bin[v_va], p_lgb_c)
        
        val_eval_df['pred'] = p_lgb_c
        daily_ics = val_eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        ic_lgb_c = daily_ics.mean()
        ic_t_lgb_c = (daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0
        
        print(f"  [LightGBM Classifier] DirAcc: {acc_lgb_c*100:.2f}% | BalAcc: {bal_lgb_c*100:.2f}% | UP Prec: {prec_lgb_c*100:.2f}% | Rank IC: {ic_lgb_c:.4f} (t={ic_t_lgb_c:.2f})")
        
        model_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "model_architecture": "LightGBM_Binary_Classifier",
            "val_sample_size": int(v_va.sum()),
            "dir_accuracy": round(float(acc_lgb_c), 4),
            "balanced_accuracy": round(float(bal_lgb_c), 4),
            "up_precision": round(float(prec_lgb_c), 4),
            "up_recall": round(float(rec_lgb_c), 4),
            "f1_score": round(float(f1_lgb_c), 4),
            "roc_auc": round(float(roc_lgb_c), 4),
            "pr_auc": round(float(pr_lgb_c), 4),
            "brier_score": round(float(brier_lgb_c), 5),
            "rank_ic": round(float(ic_lgb_c), 4),
            "rank_ic_t": round(float(ic_t_lgb_c), 2)
        })
        
        # -------------------------------------------------------------
        # MODEL 4: XGBoost Regressor (Pseudo-Huber)
        # -------------------------------------------------------------
        xgb_reg = xgb.XGBRegressor(
            objective='reg:pseudohubererror',
            n_estimators=100,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',
            random_state=42,
            n_jobs=4
        )
        xgb_reg.fit(X_tr, y_train_z[v_tr])
        pred_z_xgb = xgb_reg.predict(X_va)
        pred_xgb_r = (pred_z_xgb >= 0).astype(int)
        
        acc_xgb_r = accuracy_score(y_val_bin[v_va], pred_xgb_r)
        bal_xgb_r = balanced_accuracy_score(y_val_bin[v_va], pred_xgb_r)
        prec_xgb_r = precision_score(y_val_bin[v_va], pred_xgb_r, zero_division=0)
        rec_xgb_r = recall_score(y_val_bin[v_va], pred_xgb_r, zero_division=0)
        f1_xgb_r = f1_score(y_val_bin[v_va], pred_xgb_r, zero_division=0)
        roc_xgb_r = roc_auc_score(y_val_bin[v_va], pred_z_xgb)
        pr_xgb_r = average_precision_score(y_val_bin[v_va], pred_z_xgb)
        
        val_eval_df['pred'] = pred_z_xgb
        daily_ics = val_eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred'], g['actual'])[0] if len(g)>=10 else np.nan).dropna()
        ic_xgb_r = daily_ics.mean()
        ic_t_xgb_r = (daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0
        
        print(f"  [XGBoost Huber]      DirAcc: {acc_xgb_r*100:.2f}% | BalAcc: {bal_xgb_r*100:.2f}% | UP Prec: {prec_xgb_r*100:.2f}% | Rank IC: {ic_xgb_r:.4f} (t={ic_t_xgb_r:.2f})")
        
        model_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "model_architecture": "XGBoost_Huber_Regressor",
            "val_sample_size": int(v_va.sum()),
            "dir_accuracy": round(float(acc_xgb_r), 4),
            "balanced_accuracy": round(float(bal_xgb_r), 4),
            "up_precision": round(float(prec_xgb_r), 4),
            "up_recall": round(float(rec_xgb_r), 4),
            "f1_score": round(float(f1_xgb_r), 4),
            "roc_auc": round(float(roc_xgb_r), 4),
            "pr_auc": round(float(pr_xgb_r), 4),
            "brier_score": round(float(brier_score_loss(y_val_bin[v_va], 1/(1+np.exp(-pred_z_xgb)))), 5),
            "rank_ic": round(float(ic_xgb_r), 4),
            "rank_ic_t": round(float(ic_t_xgb_r), 2)
        })
        
        # Record summary for feature comparison
        feat_exp_records.append({
            "feature_set": set_name,
            "feature_count": len(f_cols),
            "best_model": "LightGBM_Huber_Regressor",
            "val_dir_accuracy": round(float(acc_lgb_r), 4),
            "val_balanced_accuracy": round(float(bal_lgb_r), 4),
            "val_up_precision": round(float(prec_lgb_r), 4),
            "val_roc_auc": round(float(roc_lgb_r), 4),
            "val_rank_ic": round(float(ic_lgb_r), 4),
            "val_rank_ic_t": round(float(ic_t_lgb_r), 2)
        })

    # Save CSVs
    df_feat_exp = pd.DataFrame(feat_exp_records)
    df_feat_exp.to_csv(os.path.join(OUT_DIR, "02_feature_experiments.csv"), index=False)
    print(f"\n[4] Saved feature experiments to 02_feature_experiments.csv")
    
    df_models = pd.DataFrame(model_records)
    df_models.to_csv(os.path.join(OUT_DIR, "03_model_comparison.csv"), index=False)
    print(f"[5] Saved model comparisons to 03_model_comparison.csv")
    print(f"Elapsed time: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
