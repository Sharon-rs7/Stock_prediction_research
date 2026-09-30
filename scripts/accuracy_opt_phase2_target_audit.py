"""
Accuracy Optimization - Phase 2: Target & Horizon Audit
=======================================================
Evaluates candidate horizons (H=1, 2, 3, 5, 10, 21) and target formulations
(Binary, Three-Class, Selective Absolute Threshold, Cross-Sectional Z-Score)
strictly on TRAIN and VALIDATION partitions.

Zero test set exposure. Test set is strictly locked.
Outputs: results/accuracy_optimization/01_target_analysis.csv
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score
)

sys.path.insert(0, os.path.abspath("."))
from scripts.features import FEATURE_COLUMNS
from scripts.temporal_split import build_temporal_splits

OUT_DIR = os.path.abspath("results/accuracy_optimization")
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    print("=" * 80)
    print("PHASE 2: TARGET & HORIZON AUDIT (TRAIN & VALIDATION ONLY)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load Panel
    print("\n[1] Loading Universe B panel...")
    df = pd.read_parquet("data/processed/universe_b_panel.parquet")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date'].unique())
    
    # Strictly respect chronological splits
    splits, sdates = build_temporal_splits(unique_dates, H=5)
    train_dates = set(sdates['train_dates'])
    val_dates = set(sdates['val_dates'])
    # TEST DATES ARE UNTOUCHED AND LOCKED!
    
    print(f"Total panel rows: {len(df):,}, Unique dates: {len(unique_dates)}")
    print(f"Train dates: {len(train_dates)} (rows: {df['date'].isin(train_dates).sum():,})")
    print(f"Val dates: {len(val_dates)} (rows: {df['date'].isin(val_dates).sum():,})")
    
    # 2. Compute Additional Forward Horizons if not present (H=2, 3, 10)
    print("\n[2] Computing forward return targets for H in [1, 2, 3, 5, 10, 21]...")
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    for h in [1, 2, 3, 5, 10, 21]:
        ret_col = f'fwd_ret_{h}d'
        if ret_col not in df.columns:
            print(f"  Generating {ret_col}...")
            df[ret_col] = df.groupby('ticker')['adj_close'].shift(-h) / df['adj_close'] - 1.0
            
        # Cross-sectional z-score
        z_col = f'zscore_fwd_ret_{h}d'
        if z_col not in df.columns:
            print(f"  Generating {z_col}...")
            m_t = df.groupby('date')[ret_col].transform('mean')
            s_t = df.groupby('date')[ret_col].transform('std')
            df[z_col] = (df[ret_col] - m_t) / (s_t + 1e-8)
            
        # Market-relative excess
        exc_col = f'excess_fwd_ret_{h}d'
        if exc_col not in df.columns:
            m_t = df.groupby('date')[ret_col].transform('mean')
            df[exc_col] = df[ret_col] - m_t

    # 3. Base Features
    base_features = FEATURE_COLUMNS
    print(f"\n[3] Using base feature set ({len(base_features)} features)...")
    
    # Split train and validation masks
    train_mask = df['date'].isin(train_dates)
    val_mask = df['date'].isin(val_dates)
    
    X_train = df.loc[train_mask, base_features].fillna(0).values
    X_val = df.loc[val_mask, base_features].fillna(0).values
    val_df_meta = df.loc[val_mask, ['date', 'ticker']].copy()
    
    results = []
    
    horizons = [1, 2, 3, 5, 10, 21]
    
    for h in horizons:
        print(f"\n--- Auditing Horizon H = {h} Days ---")
        ret_train = df.loc[train_mask, f'fwd_ret_{h}d'].values
        ret_val = df.loc[val_mask, f'fwd_ret_{h}d'].values
        
        # Valid mask for target (drop terminal NaN)
        v_train = ~np.isnan(ret_train)
        v_val = ~np.isnan(ret_val)
        
        # A. Binary Target: Raw Return > 0
        y_train_bin = (ret_train[v_train] > 0).astype(int)
        y_val_bin = (ret_val[v_val] > 0).astype(int)
        
        base_rate_train = y_train_bin.mean()
        base_rate_val = y_val_bin.mean()
        
        clf = lgb.LGBMClassifier(
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
        clf.fit(X_train[v_train], y_train_bin)
        
        p_val = clf.predict_proba(X_val[v_val])[:, 1]
        y_pred = (p_val >= 0.5).astype(int)
        
        acc = accuracy_score(y_val_bin, y_pred)
        bal_acc = balanced_accuracy_score(y_val_bin, y_pred)
        prec = precision_score(y_val_bin, y_pred, zero_division=0)
        rec = recall_score(y_val_bin, y_pred, zero_division=0)
        f1 = f1_score(y_val_bin, y_pred, zero_division=0)
        roc = roc_auc_score(y_val_bin, p_val)
        pr_auc = average_precision_score(y_val_bin, p_val)
        
        # Rank IC across validation dates
        val_eval_df = val_df_meta.loc[v_val].copy()
        val_eval_df['pred'] = p_val
        val_eval_df['actual'] = ret_val[v_val]
        
        def calc_rank_ic(g):
            if len(g) < 10: return np.nan
            return stats.spearmanr(g['pred'], g['actual'])[0]
            
        daily_ics = val_eval_df.groupby('date').apply(calc_rank_ic).dropna()
        rank_ic = daily_ics.mean()
        ic_t = (daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics)))) if daily_ics.std() > 0 else 0
        
        print(f"  Target: Binary (R > 0) | Val DirAcc: {acc*100:.2f}% | BalAcc: {bal_acc*100:.2f}% | UP Prec: {prec*100:.2f}% | ROC-AUC: {roc:.4f} | Rank IC: {rank_ic:.4f} (t={ic_t:.2f})")
        
        results.append({
            "horizon": h,
            "target_type": "Binary_Return_GT_0",
            "val_sample_size": int(v_val.sum()),
            "val_up_base_rate": round(float(base_rate_val), 4),
            "dir_accuracy": round(float(acc), 4),
            "balanced_accuracy": round(float(bal_acc), 4),
            "up_precision": round(float(prec), 4),
            "up_recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "roc_auc": round(float(roc), 4),
            "pr_auc": round(float(pr_auc), 4),
            "rank_ic": round(float(rank_ic), 4),
            "rank_ic_t": round(float(ic_t), 2),
            "notes": "Unconditional binary classification on raw forward return"
        })
        
        # B. Cross-Sectional Z-Score Regression Direction
        z_train = df.loc[train_mask, f'zscore_fwd_ret_{h}d'].values[v_train]
        z_val = df.loc[val_mask, f'zscore_fwd_ret_{h}d'].values[v_val]
        
        reg = lgb.LGBMRegressor(
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
        reg.fit(X_train[v_train], z_train)
        pred_z = reg.predict(X_val[v_val])
        pred_z_dir = (pred_z >= 0).astype(int)
        actual_z_dir = (z_val >= 0).astype(int)
        
        acc_z = accuracy_score(actual_z_dir, pred_z_dir)
        bal_acc_z = balanced_accuracy_score(actual_z_dir, pred_z_dir)
        prec_z = precision_score(actual_z_dir, pred_z_dir, zero_division=0)
        rec_z = recall_score(actual_z_dir, pred_z_dir, zero_division=0)
        f1_z = f1_score(actual_z_dir, pred_z_dir, zero_division=0)
        roc_z = roc_auc_score(actual_z_dir, pred_z)
        pr_auc_z = average_precision_score(actual_z_dir, pred_z)
        
        val_eval_df['pred_z'] = pred_z
        val_eval_df['actual_z'] = z_val
        daily_ics_z = val_eval_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred_z'], g['actual_z'])[0] if len(g)>=10 else np.nan).dropna()
        rank_ic_z = daily_ics_z.mean()
        ic_t_z = (daily_ics_z.mean() / (daily_ics_z.std() / np.sqrt(len(daily_ics_z)))) if daily_ics_z.std() > 0 else 0
        
        print(f"  Target: Cross-Sectional Z-Score | Val DirAcc: {acc_z*100:.2f}% | BalAcc: {bal_acc_z*100:.2f}% | UP Prec: {prec_z*100:.2f}% | Rank IC: {rank_ic_z:.4f} (t={ic_t_z:.2f})")
        
        results.append({
            "horizon": h,
            "target_type": "CrossSectional_ZScore_Huber",
            "val_sample_size": int(v_val.sum()),
            "val_up_base_rate": round(float(actual_z_dir.mean()), 4),
            "dir_accuracy": round(float(acc_z), 4),
            "balanced_accuracy": round(float(bal_acc_z), 4),
            "up_precision": round(float(prec_z), 4),
            "up_recall": round(float(rec_z), 4),
            "f1_score": round(float(f1_z), 4),
            "roc_auc": round(float(roc_z), 4),
            "pr_auc": round(float(pr_auc_z), 4),
            "rank_ic": round(float(rank_ic_z), 4),
            "rank_ic_t": round(float(ic_t_z), 2),
            "notes": "Direction predicted from continuous cross-sectional z-score Huber regressor"
        })
        
        # C. Three-Class Target (UP / NO-MOVE / DOWN)
        # Learn threshold exclusively from train partition (e.g. 25th percentile of absolute return)
        tau = np.percentile(np.abs(ret_train[v_train]), 25)
        print(f"  Train-derived NO-MOVE threshold tau: {tau*100:.3f}%")
        
        # Train class: 0: Down (R < -tau), 1: No-move (|R| <= tau), 2: Up (R > tau)
        def to_3class(r):
            c = np.ones_like(r, dtype=int)
            c[r < -tau] = 0
            c[r > tau] = 2
            return c
            
        y_train_3c = to_3class(ret_train[v_train])
        y_val_3c = to_3class(ret_val[v_val])
        
        clf_3c = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=3,
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
        clf_3c.fit(X_train[v_train], y_train_3c)
        p_val_3c = clf_3c.predict_proba(X_val[v_val])
        
        # Directional prediction when predicting active move (Up vs Down)
        # Margin: P(Up) - P(Down)
        margin = p_val_3c[:, 2] - p_val_3c[:, 0]
        # Evaluate active calls where model conviction is highest
        active_pred = np.where(margin > 0, 1, 0)
        actual_active = np.where(ret_val[v_val] > 0, 1, 0)
        
        acc_3c = accuracy_score(actual_active, active_pred)
        bal_acc_3c = balanced_accuracy_score(actual_active, active_pred)
        prec_3c = precision_score(actual_active, active_pred, zero_division=0)
        rec_3c = recall_score(actual_active, active_pred, zero_division=0)
        f1_3c = f1_score(actual_active, active_pred, zero_division=0)
        roc_3c = roc_auc_score(actual_active, margin)
        pr_auc_3c = average_precision_score(actual_active, margin)
        
        results.append({
            "horizon": h,
            "target_type": "ThreeClass_ActiveMargin",
            "val_sample_size": int(v_val.sum()),
            "val_up_base_rate": round(float(actual_active.mean()), 4),
            "dir_accuracy": round(float(acc_3c), 4),
            "balanced_accuracy": round(float(bal_acc_3c), 4),
            "up_precision": round(float(prec_3c), 4),
            "up_recall": round(float(rec_3c), 4),
            "f1_score": round(float(f1_3c), 4),
            "roc_auc": round(float(roc_3c), 4),
            "pr_auc": round(float(pr_auc_3c), 4),
            "rank_ic": round(float(rank_ic), 4),
            "rank_ic_t": round(float(ic_t), 2),
            "notes": f"Three-class model with train-derived threshold tau={tau*100:.3f}%"
        })
        
        # D. Selective Target: Filtered to Decisive Moves (|R| > tau_train)
        decisive_mask_val = np.abs(ret_val[v_val]) > tau
        if decisive_mask_val.sum() > 0:
            y_val_decisive = (ret_val[v_val][decisive_mask_val] > 0).astype(int)
            p_val_decisive = p_val[decisive_mask_val]
            y_pred_decisive = (p_val_decisive >= 0.5).astype(int)
            
            acc_dec = accuracy_score(y_val_decisive, y_pred_decisive)
            bal_acc_dec = balanced_accuracy_score(y_val_decisive, y_pred_decisive)
            prec_dec = precision_score(y_val_decisive, y_pred_decisive, zero_division=0)
            rec_dec = recall_score(y_val_decisive, y_pred_decisive, zero_division=0)
            f1_dec = f1_score(y_val_decisive, y_pred_decisive, zero_division=0)
            roc_dec = roc_auc_score(y_val_decisive, p_val_decisive)
            pr_auc_dec = average_precision_score(y_val_decisive, p_val_decisive)
            
            print(f"  Target: Selective (|R| > tau) | Val DirAcc: {acc_dec*100:.2f}% | UP Prec: {prec_dec*100:.2f}% | Samples: {decisive_mask_val.sum():,}")
            
            results.append({
                "horizon": h,
                "target_type": "Selective_Decisive_Move",
                "val_sample_size": int(decisive_mask_val.sum()),
                "val_up_base_rate": round(float(y_val_decisive.mean()), 4),
                "dir_accuracy": round(float(acc_dec), 4),
                "balanced_accuracy": round(float(bal_acc_dec), 4),
                "up_precision": round(float(prec_dec), 4),
                "up_recall": round(float(rec_dec), 4),
                "f1_score": round(float(f1_dec), 4),
                "roc_auc": round(float(roc_dec), 4),
                "pr_auc": round(float(pr_auc_dec), 4),
                "rank_ic": round(float(rank_ic), 4),
                "rank_ic_t": round(float(ic_t), 2),
                "notes": f"Evaluation restricted to decisive moves (|R| > {tau*100:.3f}%)"
            })
            
    # 4. Save CSV
    out_df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, "01_target_analysis.csv")
    out_df.to_csv(csv_path, index=False)
    print(f"\n[4] Target analysis complete! Saved to {csv_path}")
    print(f"Total audit elapsed time: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
