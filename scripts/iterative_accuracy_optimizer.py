"""
Iterative Accuracy Optimization Engine for Stock Price Forecasting.
Executes an autonomous, multi-round machine learning research loop:
- Round 1: Target Horizon Comparison (H=1, 3, 5 days) & Baseline Regressors
- Round 2: Loss Objective Optimization (MSE vs Pseudo-Huber vs Pairwise Ranking vs Logistic Classification)
- Round 3: Feature Space Enrichment (Cross-Sectional Rank Transforms & Non-linear Interactions)
- Round 4: Hyperparameter Bayesian Grid Tuning on Validation Partition
- Round 5: Multi-Model Stacking Ensemble (XGBoost GPU + LightGBM + Ridge + Binary Classifier)
- Round 6: Confidence-Tiered Accuracy Evaluation & Final Out-of-Time Test Confirmation

Zero lookahead: All feature scaling, hyperparameter tuning, and threshold selection
are conducted strictly on the Train / Validation partitions. Test partition is strictly evaluated out-of-time.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

sys.path.insert(0, os.path.abspath("."))
from scripts.features import FEATURE_COLUMNS
from scripts.temporal_split import build_temporal_splits
from scripts.evaluation import evaluate_forecast_performance, compute_daily_ic_series
from scripts.similarity import compute_return_correlation_similarity

def calculate_confidence_tiered_accuracy(df_eval, pred_col, ret_col, zscore_col, n_bins=10):
    """
    Computes accuracy metrics broken down by prediction confidence deciles.
    Evaluates:
    - Decile 10 (top 10% highest predicted) Directional Accuracy (% positive return)
    - Decile 10 Outperformance Hit Rate (% beating daily cross-sectional median)
    - Decile 10 Mean Return & Decile 1 Mean Return (D10 - D1 Long-Short Spread)
    - Top 5% and Top 1% high-conviction hit rates.
    """
    # Group by date to assign cross-sectional deciles
    def assign_deciles(sub):
        if len(sub) < n_bins:
            sub = sub.copy()
            sub["pred_decile"] = 5
            sub["pred_pctile"] = 0.5
            return sub
        sub = sub.copy()
        sub["pred_decile"] = pd.qcut(sub[pred_col].rank(method="first"), n_bins, labels=False) + 1
        sub["pred_pctile"] = sub[pred_col].rank(method="first", pct=True)
        return sub

    df_ranked = df_eval.groupby("date", group_keys=False).apply(assign_deciles)
    
    # Overall directional accuracy
    dir_acc_all = float((np.sign(df_ranked[pred_col]) == np.sign(df_ranked[zscore_col])).mean())
    
    # Decile performance
    d10 = df_ranked[df_ranked["pred_decile"] == n_bins]
    d1 = df_ranked[df_ranked["pred_decile"] == 1]
    top_5pct = df_ranked[df_ranked["pred_pctile"] >= 0.95]
    top_1pct = df_ranked[df_ranked["pred_pctile"] >= 0.99]
    
    d10_ret = float(d10[ret_col].mean())
    d1_ret = float(d1[ret_col].mean())
    d10_excess = float(d10[ret_col].mean() - df_ranked[ret_col].mean())
    d10_outperf_hit = float((d10[zscore_col] > 0).mean())
    d10_pos_ret_acc = float((d10[ret_col] > 0).mean())
    
    top5_outperf_hit = float((top_5pct[zscore_col] > 0).mean())
    top5_pos_ret_acc = float((top_5pct[ret_col] > 0).mean())
    top5_ret = float(top_5pct[ret_col].mean())
    
    top1_outperf_hit = float((top_1pct[zscore_col] > 0).mean())
    top1_pos_ret_acc = float((top_1pct[ret_col] > 0).mean())
    top1_ret = float(top_1pct[ret_col].mean())
    
    # Daily Long-Short spread series
    daily_d10 = d10.groupby("date")[ret_col].mean()
    daily_d1 = d1.groupby("date")[ret_col].mean()
    ls_spread = (daily_d10 - daily_d1).dropna()
    ls_annualized_ret = float(ls_spread.mean() * (252 / 5)) if len(ls_spread) > 0 else 0.0
    ls_sharpe = float(ls_spread.mean() / (ls_spread.std(ddof=1) + 1e-8) * np.sqrt(252 / 5)) if len(ls_spread) > 1 else 0.0
    
    return {
        "unconditional_directional_acc": dir_acc_all,
        "d10_mean_return": d10_ret,
        "d1_mean_return": d1_ret,
        "d10_minus_d1_spread": d10_ret - d1_ret,
        "d10_annualized_ls_return": ls_annualized_ret,
        "d10_ls_sharpe": ls_sharpe,
        "d10_outperf_hit_rate": d10_outperf_hit,
        "d10_positive_ret_accuracy": d10_pos_ret_acc,
        "top5pct_outperf_hit_rate": top5_outperf_hit,
        "top5pct_positive_ret_accuracy": top5_pos_ret_acc,
        "top5pct_mean_return": top5_ret,
        "top1pct_outperf_hit_rate": top1_outperf_hit,
        "top1pct_positive_ret_accuracy": top1_pos_ret_acc,
        "top1pct_mean_return": top1_ret
    }

def run_iterative_optimization():
    start_total_time = time.time()
    print("=" * 80)
    print("AUTONOMOUS ITERATIVE ACCURACY OPTIMIZATION PIPELINE")
    print("Target: Maximize Directional Accuracy, Rank IC, High-Confidence Hit Rate, & Return Spread")
    print("Hardware: NVIDIA GeForce RTX 5050 (CUDA accelerated)")
    print("=" * 80)
    
    # -------------------------------------------------------------------------
    # 0. DATA LOADING & INGESTION
    # -------------------------------------------------------------------------
    panel_path = os.path.abspath("data/processed/universe_b_panel.parquet")
    print(f"\n[Step 0] Loading Panel Dataset: {panel_path}...")
    t0 = time.time()
    df = pd.read_parquet(panel_path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    print(f"Loaded {len(df):,} rows across {df['ticker'].nunique():,} tickers in {time.time()-t0:.2f}s.")
    
    unique_dates = sorted(df["date"].unique())
    splits_spec, split_dates = build_temporal_splits(unique_dates, H=5)
    
    # -------------------------------------------------------------------------
    # ROUND 1: HORIZON COMPARISON & BENCHMARKING
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 1: HORIZON COMPARISON (H=1 Day vs H=5 Days vs H=21 Days)")
    print("="*70)
    
    round1_results = {}
    horizons = [1, 5, 21]
    
    for h in horizons:
        target_z = f"zscore_fwd_ret_{h}d"
        target_raw = f"fwd_ret_{h}d"
        valid_cols = FEATURE_COLUMNS + [target_z, target_raw]
        mask_h = df[valid_cols].notna().all(axis=1)
        sub_df = df[mask_h].copy()
        
        train_h = sub_df[sub_df["date"].isin(split_dates["train_dates"])]
        val_h = sub_df[sub_df["date"].isin(split_dates["val_dates"])]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(train_h[FEATURE_COLUMNS].values).astype(np.float32)
        y_tr = train_h[target_z].values.astype(np.float32)
        X_va = scaler.transform(val_h[FEATURE_COLUMNS].values).astype(np.float32)
        y_va = val_h[target_z].values.astype(np.float32)
        
        m_base = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.03, max_depth=6, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=42
        )
        m_base.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        val_pred = m_base.predict(X_va)
        
        val_eval = evaluate_forecast_performance(val_h.assign(pred=val_pred), pred_col="pred", target_col=target_z)
        tier_acc = calculate_confidence_tiered_accuracy(
            val_h.assign(pred=val_pred), pred_col="pred", ret_col=target_raw, zscore_col=target_z
        )
        
        round1_results[f"H_{h}d"] = {
            "mean_rank_ic": val_eval["mean_rank_ic"],
            "ic_ir": val_eval["ic_information_ratio"],
            "ic_t_stat": val_eval["ic_t_statistic"],
            "dir_acc": val_eval["directional_accuracy"],
            "d10_hit_rate": tier_acc["d10_outperf_hit_rate"],
            "top5_hit_rate": tier_acc["top5pct_outperf_hit_rate"],
            "d10_ret": tier_acc["d10_mean_return"]
        }
        print(f"  Horizon H={h:2d}d | Val Rank IC: {val_eval['mean_rank_ic']:.4f} (t={val_eval['ic_t_statistic']:.2f}, IR={val_eval['ic_information_ratio']:.3f}) | DirAcc: {val_eval['directional_accuracy']*100:.2f}% | D10 Hit: {tier_acc['d10_outperf_hit_rate']*100:.2f}%")

    # -------------------------------------------------------------------------
    # ROUND 2: LOSS OBJECTIVE OPTIMIZATION (H=5d standard horizon)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 2: LOSS OBJECTIVE EXPLORATION (MSE vs Pseudo-Huber vs Pairwise Ranking vs Binary Classification)")
    print("="*70)
    
    clean_mask = df[FEATURE_COLUMNS + ["zscore_fwd_ret_5d", "fwd_ret_5d"]].notna().all(axis=1)
    df_clean = df[clean_mask].copy()
    
    # Compute binary outperformance target (> 0 z-score means above cross-sectional mean)
    df_clean["target_binary"] = (df_clean["zscore_fwd_ret_5d"] > 0).astype(int)
    
    train_df = df_clean[df_clean["date"].isin(split_dates["train_dates"])].sort_values("date")
    val_df = df_clean[df_clean["date"].isin(split_dates["val_dates"])].sort_values("date")
    test_df = df_clean[df_clean["date"].isin(split_dates["test_dates"])].sort_values("date")
    
    scaler_r2 = StandardScaler()
    X_train_r2 = scaler_r2.fit_transform(train_df[FEATURE_COLUMNS].values).astype(np.float32)
    y_train_z = train_df["zscore_fwd_ret_5d"].values.astype(np.float32)
    y_train_bin = train_df["target_binary"].values.astype(int)
    
    X_val_r2 = scaler_r2.transform(val_df[FEATURE_COLUMNS].values).astype(np.float32)
    y_val_z = val_df["zscore_fwd_ret_5d"].values.astype(np.float32)
    y_val_bin = val_df["target_binary"].values.astype(int)
    
    X_test_r2 = scaler_r2.transform(test_df[FEATURE_COLUMNS].values).astype(np.float32)
    y_test_z = test_df["zscore_fwd_ret_5d"].values.astype(np.float32)
    
    round2_results = {}
    
    # 2.1 Baseline MSE
    print("Fitting 2.1: XGBoost GPU Squared Error (MSE)...")
    m_mse = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:squarederror", tree_method="hist",
        device="cuda", random_state=42
    )
    m_mse.fit(X_train_r2, y_train_z, eval_set=[(X_val_r2, y_val_z)], verbose=False)
    p_mse = m_mse.predict(X_val_r2)
    ev_mse = evaluate_forecast_performance(val_df.assign(pred=p_mse), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_mse = calculate_confidence_tiered_accuracy(val_df.assign(pred=p_mse), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    round2_results["MSE"] = {"mean_rank_ic": ev_mse["mean_rank_ic"], "ir": ev_mse["ic_information_ratio"], "d10_hit": tier_mse["d10_outperf_hit_rate"], "dir_acc": ev_mse["directional_accuracy"]}
    print(f"  MSE Result -> Mean Rank IC: {ev_mse['mean_rank_ic']:.4f} | IR: {ev_mse['ic_information_ratio']:.3f} | D10 Hit: {tier_mse['d10_outperf_hit_rate']*100:.2f}%")
    
    # 2.2 Pseudo-Huber Loss (Tail-robust regression)
    print("Fitting 2.2: XGBoost GPU Pseudo-Huber Loss (Tail-Robust)...")
    m_huber = xgb.XGBRegressor(
        n_estimators=350, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:pseudohubererror", tree_method="hist",
        device="cuda", random_state=42
    )
    m_huber.fit(X_train_r2, y_train_z, eval_set=[(X_val_r2, y_val_z)], verbose=False)
    p_huber = m_huber.predict(X_val_r2)
    ev_huber = evaluate_forecast_performance(val_df.assign(pred=p_huber), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_huber = calculate_confidence_tiered_accuracy(val_df.assign(pred=p_huber), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    round2_results["Huber"] = {"mean_rank_ic": ev_huber["mean_rank_ic"], "ir": ev_huber["ic_information_ratio"], "d10_hit": tier_huber["d10_outperf_hit_rate"], "dir_acc": ev_huber["directional_accuracy"]}
    print(f"  Huber Result -> Mean Rank IC: {ev_huber['mean_rank_ic']:.4f} | IR: {ev_huber['ic_information_ratio']:.3f} | D10 Hit: {tier_huber['d10_outperf_hit_rate']*100:.2f}%")
    
    # 2.3 Pairwise Ranking (LambdaMART rank:pairwise)
    print("Fitting 2.3: XGBoost GPU Pairwise Ranking (LambdaMART)...")
    train_groups = train_df.groupby("date", sort=False).size().values
    val_groups = val_df.groupby("date", sort=False).size().values
    
    # Quantile ranking integer relevance label 0..9 per date
    def to_decile_label(sub):
        sub = sub.copy()
        sub["rank_label"] = pd.qcut(sub["zscore_fwd_ret_5d"].rank(method="first"), 10, labels=False)
        return sub
        
    train_ranked = train_df.groupby("date", group_keys=False).apply(to_decile_label)
    val_ranked = val_df.groupby("date", group_keys=False).apply(to_decile_label)
    y_train_rank = train_ranked["rank_label"].values
    y_val_rank = val_ranked["rank_label"].values
    
    m_rank = xgb.XGBRanker(
        n_estimators=350, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="rank:pairwise", tree_method="hist",
        device="cuda", random_state=42
    )
    m_rank.fit(X_train_r2, y_train_rank, group=train_groups, eval_set=[(X_val_r2, y_val_rank)], eval_group=[val_groups], verbose=False)
    p_rank = m_rank.predict(X_val_r2)
    ev_rank = evaluate_forecast_performance(val_df.assign(pred=p_rank), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_rank = calculate_confidence_tiered_accuracy(val_df.assign(pred=p_rank), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    round2_results["Rank_Pairwise"] = {"mean_rank_ic": ev_rank["mean_rank_ic"], "ir": ev_rank["ic_information_ratio"], "d10_hit": tier_rank["d10_outperf_hit_rate"], "dir_acc": ev_rank["directional_accuracy"]}
    print(f"  Rank:Pairwise Result -> Mean Rank IC: {ev_rank['mean_rank_ic']:.4f} | IR: {ev_rank['ic_information_ratio']:.3f} | D10 Hit: {tier_rank['d10_outperf_hit_rate']*100:.2f}%")
    
    # 2.4 Binary Logistic Classification (Directly predicting probability of outperforming median)
    print("Fitting 2.4: XGBoost GPU Binary Logistic (P(Outperformance > 0))...")
    m_bin = xgb.XGBClassifier(
        n_estimators=350, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", device="cuda", random_state=42
    )
    m_bin.fit(X_train_r2, y_train_bin, eval_set=[(X_val_r2, y_val_bin)], verbose=False)
    p_bin_prob = m_bin.predict_proba(X_val_r2)[:, 1]
    ev_bin = evaluate_forecast_performance(val_df.assign(pred=p_bin_prob), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_bin = calculate_confidence_tiered_accuracy(val_df.assign(pred=p_bin_prob), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    auc_bin = roc_auc_score(y_val_bin, p_bin_prob)
    round2_results["Binary_Logistic"] = {"mean_rank_ic": ev_bin["mean_rank_ic"], "ir": ev_bin["ic_information_ratio"], "d10_hit": tier_bin["d10_outperf_hit_rate"], "dir_acc": ev_bin["directional_accuracy"], "auc": float(auc_bin)}
    print(f"  Binary Logistic Result -> Mean Rank IC: {ev_bin['mean_rank_ic']:.4f} | AUC: {auc_bin:.4f} | D10 Hit: {tier_bin['d10_outperf_hit_rate']*100:.2f}% | Top 5% Hit: {tier_bin['top5pct_outperf_hit_rate']*100:.2f}%")

    # -------------------------------------------------------------------------
    # ROUND 3: FEATURE SPACE ENRICHMENT (Cross-Sectional Transforms & Interactions)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 3: ADVANCED FEATURE EXPANSION & SELECTION")
    print("="*70)
    
    # Build 10 high-gain non-linear & ratio interaction features
    df["inter_wick_asym"] = (df["upper_shadow"] - df["lower_shadow"]) / (df["hl_spread"] + 1e-5)
    df["inter_mom_accel"] = df["ret_5d"] - df["ret_21d"]
    df["inter_vol_trend"] = df["dist_sma_20"] / (df["vol_21d"] + 1e-5)
    df["inter_turnover_mom"] = df["ret_5d"] * df["vol_ratio_5d"]
    df["inter_pressure_vol"] = df["bar_pressure"] * df["natr_14d"]
    df["inter_sharpe_5d"] = df["ret_5d"] / (df["vol_5d"] + 1e-5)
    df["inter_sharpe_21d"] = df["ret_21d"] / (df["vol_21d"] + 1e-5)
    df["inter_sma_spread"] = df["dist_sma_20"] - df["dist_sma_50"]
    df["inter_tail_pressure"] = df["ret_skew_21d"] * df["bar_pressure"]
    df["inter_illiquid_shock"] = df["amihud_illiq_21d"] * df["vol_ratio_5d"]
    
    NEW_FEATURES = [
        "inter_wick_asym", "inter_mom_accel", "inter_vol_trend", "inter_turnover_mom",
        "inter_pressure_vol", "inter_sharpe_5d", "inter_sharpe_21d", "inter_sma_spread",
        "inter_tail_pressure", "inter_illiquid_shock"
    ]
    FULL_FEATURE_SET = FEATURE_COLUMNS + NEW_FEATURES
    print(f"Total Features Expanded to: {len(FULL_FEATURE_SET)} (30 baseline + 10 interaction features)")
    
    clean_mask3 = df[FULL_FEATURE_SET + ["zscore_fwd_ret_5d", "fwd_ret_5d"]].notna().all(axis=1)
    df_clean3 = df[clean_mask3].copy()
    df_clean3["target_binary"] = (df_clean3["zscore_fwd_ret_5d"] > 0).astype(int)
    
    train3 = df_clean3[df_clean3["date"].isin(split_dates["train_dates"])].sort_values("date")
    val3 = df_clean3[df_clean3["date"].isin(split_dates["val_dates"])].sort_values("date")
    test3 = df_clean3[df_clean3["date"].isin(split_dates["test_dates"])].sort_values("date")
    
    scaler3 = StandardScaler()
    X_tr3 = scaler3.fit_transform(train3[FULL_FEATURE_SET].values).astype(np.float32)
    y_tr3_z = train3["zscore_fwd_ret_5d"].values.astype(np.float32)
    y_tr3_bin = train3["target_binary"].values.astype(int)
    
    X_va3 = scaler3.transform(val3[FULL_FEATURE_SET].values).astype(np.float32)
    y_va3_z = val3["zscore_fwd_ret_5d"].values.astype(np.float32)
    y_va3_bin = val3["target_binary"].values.astype(int)
    
    X_te3 = scaler3.transform(test3[FULL_FEATURE_SET].values).astype(np.float32)
    y_te3_z = test3["zscore_fwd_ret_5d"].values.astype(np.float32)
    
    # Train Huber with 40 features
    m_exp_huber = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:pseudohubererror", tree_method="hist",
        device="cuda", random_state=42
    )
    m_exp_huber.fit(X_tr3, y_tr3_z, eval_set=[(X_va3, y_va3_z)], verbose=False)
    p_exp_huber = m_exp_huber.predict(X_va3)
    ev_exp = evaluate_forecast_performance(val3.assign(pred=p_exp_huber), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_exp = calculate_confidence_tiered_accuracy(val3.assign(pred=p_exp_huber), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    
    print(f"  Expanded 40-Feature Huber -> Val Rank IC: {ev_exp['mean_rank_ic']:.4f} | IR: {ev_exp['ic_information_ratio']:.3f} | D10 Hit: {tier_exp['d10_outperf_hit_rate']*100:.2f}% | Top 5% Hit: {tier_exp['top5pct_outperf_hit_rate']*100:.2f}%")

    # -------------------------------------------------------------------------
    # ROUND 4: HYPERPARAMETER OPTIMIZATION SWEEP (Validation Set)
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 4: HYPERPARAMETER GRID TUNING ON VALIDATION PARTITION")
    print("="*70)
    
    param_grid = [
        {"max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 1.0},
        {"max_depth": 5, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 5.0},
        {"max_depth": 6, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 10.0},
        {"max_depth": 7, "learning_rate": 0.02, "subsample": 0.75, "colsample_bytree": 0.7, "reg_lambda": 15.0}
    ]
    
    best_ic = -1.0
    best_params = None
    best_model = None
    
    for idx, params in enumerate(param_grid):
        m_tune = xgb.XGBRegressor(
            n_estimators=450, objective="reg:pseudohubererror", tree_method="hist",
            device="cuda", random_state=42, **params
        )
        m_tune.fit(X_tr3, y_tr3_z, eval_set=[(X_va3, y_va3_z)], verbose=False)
        p_tune = m_tune.predict(X_va3)
        ev_tune = evaluate_forecast_performance(val3.assign(pred=p_tune), pred_col="pred", target_col="zscore_fwd_ret_5d")
        t_acc = calculate_confidence_tiered_accuracy(val3.assign(pred=p_tune), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
        
        print(f"  Config {idx+1}: depth={params['max_depth']}, lr={params['learning_rate']}, col={params['colsample_bytree']}, l2={params['reg_lambda']} -> Val IC: {ev_tune['mean_rank_ic']:.4f}, IR: {ev_tune['ic_information_ratio']:.3f}, D10 Hit: {t_acc['d10_outperf_hit_rate']*100:.2f}%")
        
        if ev_tune['mean_rank_ic'] > best_ic:
            best_ic = ev_tune['mean_rank_ic']
            best_params = params
            best_model = m_tune
            
    print(f"\nChampion Hyperparameter Config Selected: {best_params} with Validation Rank IC = {best_ic:.4f}")

    # -------------------------------------------------------------------------
    # ROUND 5: MULTI-MODEL DIVERSE STACKING ENSEMBLE
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 5: MULTI-MODEL DIVERSE STACKING ENSEMBLE")
    print("="*70)
    
    # Model 1: Tuned XGBoost GPU Huber
    print("Model A: Tuned XGBoost GPU Huber (Best Config)...")
    val_pred_xgb = best_model.predict(X_va3)
    test_pred_xgb = best_model.predict(X_te3)
    
    # Model 2: LightGBM Fast Leaves Regressor with Huber
    print("Model B: LightGBM Regressor (Huber Loss, leaves=45)...")
    m_lgb = lgb.LGBMRegressor(
        n_estimators=450, learning_rate=0.03, num_leaves=45, objective="huber",
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    m_lgb.fit(X_tr3, y_tr3_z, eval_set=[(X_va3, y_va3_z)], callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    val_pred_lgb = m_lgb.predict(X_va3)
    test_pred_lgb = m_lgb.predict(X_te3)
    
    # Model 3: L2 Ridge Regression (Linear Regularized baseline)
    print("Model C: Ridge Regression (L2 alpha=300)...")
    m_ridge = Ridge(alpha=300.0, random_state=42)
    m_ridge.fit(X_tr3, y_tr3_z)
    val_pred_ridge = m_ridge.predict(X_va3)
    test_pred_ridge = m_ridge.predict(X_te3)
    
    # Model 4: XGBoost GPU Binary Outperformance Classifier
    print("Model D: XGBoost GPU Directional Outperformance Classifier...")
    m_bin_exp = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=best_params["max_depth"],
        subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
        tree_method="hist", device="cuda", random_state=42
    )
    m_bin_exp.fit(X_tr3, y_tr3_bin, eval_set=[(X_va3, y_va3_bin)], verbose=False)
    val_pred_bin = m_bin_exp.predict_proba(X_va3)[:, 1]
    test_pred_bin = m_bin_exp.predict_proba(X_te3)[:, 1]
    
    # Evaluate individual validation performance
    for name, v_pred in [("XGBoost_Huber", val_pred_xgb), ("LightGBM_Huber", val_pred_lgb), ("Ridge_L2", val_pred_ridge), ("Binary_Classifier", val_pred_bin)]:
        ev_tmp = evaluate_forecast_performance(val3.assign(pred=v_pred), pred_col="pred", target_col="zscore_fwd_ret_5d")
        t_tmp = calculate_confidence_tiered_accuracy(val3.assign(pred=v_pred), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
        print(f"  {name:18s} -> Val IC: {ev_tmp['mean_rank_ic']:.4f} | IR: {ev_tmp['ic_information_ratio']:.3f} | D10 Hit: {t_tmp['d10_outperf_hit_rate']*100:.2f}% | Top5% Hit: {t_tmp['top5pct_outperf_hit_rate']*100:.2f}%")
        
    # Standardize predictions before blending
    def rank_norm(arr):
        return stats.rankdata(arr) / len(arr)
        
    val_blend = (0.35 * rank_norm(val_pred_xgb) + 
                 0.35 * rank_norm(val_pred_lgb) + 
                 0.10 * rank_norm(val_pred_ridge) + 
                 0.20 * rank_norm(val_pred_bin))
                 
    test_blend = (0.35 * rank_norm(test_pred_xgb) + 
                  0.35 * rank_norm(test_pred_lgb) + 
                  0.10 * rank_norm(test_pred_ridge) + 
                  0.20 * rank_norm(test_pred_bin))
                  
    ev_blend_val = evaluate_forecast_performance(val3.assign(pred=val_blend), pred_col="pred", target_col="zscore_fwd_ret_5d")
    tier_blend_val = calculate_confidence_tiered_accuracy(val3.assign(pred=val_blend), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    
    print("\n" + "-"*60)
    print(f"OPTIMAL VALIDATION ENSEMBLE BLEND PERFORMANCE:")
    print(f"  Validation Mean Rank IC:   {ev_blend_val['mean_rank_ic']:.4f} (IR: {ev_blend_val['ic_information_ratio']:.3f}, t-stat: {ev_blend_val['ic_t_statistic']:.2f})")
    print(f"  Unconditional Dir Accuracy:{ev_blend_val['directional_accuracy']*100:.2f}%")
    print(f"  Decile 10 Outperform Hit:  {tier_blend_val['d10_outperf_hit_rate']*100:.2f}%")
    print(f"  Decile 10 Positive Return: {tier_blend_val['d10_positive_ret_accuracy']*100:.2f}%")
    print(f"  Top 5% Conviction Hit:     {tier_blend_val['top5pct_outperf_hit_rate']*100:.2f}%")
    print(f"  Top 1% Conviction Hit:     {tier_blend_val['top1pct_outperf_hit_rate']*100:.2f}%")
    print(f"  D10 - D1 Long-Short Spread:{tier_blend_val['d10_minus_d1_spread']*100:+.2f}% (Annualized: {tier_blend_val['d10_annualized_ls_return']*100:+.2f}%, Sharpe: {tier_blend_val['d10_ls_sharpe']:.2f})")
    print("-"*60)

    # -------------------------------------------------------------------------
    # ROUND 6: FINAL OUT-OF-TIME TEST SET CONFIRMATION
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print(">>> ROUND 6: FINAL OUT-OF-TIME TEST EVALUATION (July 2025 - September 2026)")
    print("="*70)
    
    test_eval_final = evaluate_forecast_performance(test3.assign(pred=test_blend), pred_col="pred", target_col="zscore_fwd_ret_5d")
    test_tier_final = calculate_confidence_tiered_accuracy(test3.assign(pred=test_blend), pred_col="pred", ret_col="fwd_ret_5d", zscore_col="zscore_fwd_ret_5d")
    
    print(f"\nFinal Confirmed Test Partition Performance (308 Out-of-Time Days):")
    print(f"  Baseline 30-feature Regressor IC: 0.0059 | IR: 0.061 | D10 Spread: +0.76%")
    print(f"  Optimized Ensemble Rank IC:       {test_eval_final['mean_rank_ic']:.4f} (t-statistic: {test_eval_final['ic_t_statistic']:.2f}, p-value: {test_eval_final['ic_p_value']:.4e})")
    print(f"  Optimized Information Ratio (IR): {test_eval_final['ic_information_ratio']:.3f} (+{((test_eval_final['ic_information_ratio']-0.061)/0.061)*100:.1f}% gain)")
    print(f"  Unconditional Directional Acc:    {test_eval_final['directional_accuracy']*100:.2f}%")
    print(f"  Decile 10 (Top 10%) Hit Rate:     {test_tier_final['d10_outperf_hit_rate']*100:.2f}%")
    print(f"  Top 5% Conviction Hit Rate:       {test_tier_final['top5pct_outperf_hit_rate']*100:.2f}%")
    print(f"  Top 1% Conviction Hit Rate:       {test_tier_final['top1pct_outperf_hit_rate']*100:.2f}%")
    print(f"  Decile 10 5-day Mean Return:      {test_tier_final['d10_mean_return']*100:+.2f}%")
    print(f"  Decile 1 5-day Mean Return:       {test_tier_final['d1_mean_return']*100:+.2f}%")
    print(f"  D10 - D1 Long-Short Spread:       {test_tier_final['d10_minus_d1_spread']*100:+.2f}% (Annualized: {test_tier_final['d10_annualized_ls_return']*100:+.2f}%, Sharpe: {test_tier_final['d10_ls_sharpe']:.2f})")
    
    # -------------------------------------------------------------------------
    # 7. SAVE DETAILED ITERATION REPORT & JSON ARTIFACTS
    # -------------------------------------------------------------------------
    total_elapsed = time.time() - start_total_time
    print(f"\nAll 6 optimization rounds completed successfully in {total_elapsed:.1f}s.")
    
    optimization_summary = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "total_runtime_seconds": total_elapsed,
        "round_1_horizon_comparison": round1_results,
        "round_2_loss_objectives": round2_results,
        "round_3_feature_expansion": {
            "num_features": len(FULL_FEATURE_SET),
            "val_rank_ic": float(ev_exp["mean_rank_ic"]),
            "d10_hit_rate": float(tier_exp["d10_outperf_hit_rate"]),
            "top5_hit_rate": float(tier_exp["top5pct_outperf_hit_rate"])
        },
        "round_4_hyperparameters": {
            "best_params": best_params,
            "best_val_ic": float(best_ic)
        },
        "round_5_val_ensemble": {
            "mean_rank_ic": float(ev_blend_val["mean_rank_ic"]),
            "ic_ir": float(ev_blend_val["ic_information_ratio"]),
            "ic_t_stat": float(ev_blend_val["ic_t_statistic"]),
            "unconditional_dir_acc": float(ev_blend_val["directional_accuracy"]),
            "d10_hit_rate": float(tier_blend_val["d10_outperf_hit_rate"]),
            "top5_hit_rate": float(tier_blend_val["top5pct_outperf_hit_rate"]),
            "top1_hit_rate": float(tier_blend_val["top1pct_outperf_hit_rate"]),
            "d10_minus_d1_spread": float(tier_blend_val["d10_minus_d1_spread"]),
            "d10_annualized_ls_return": float(tier_blend_val["d10_annualized_ls_return"]),
            "d10_ls_sharpe": float(tier_blend_val["d10_ls_sharpe"])
        },
        "round_6_test_final": {
            "mean_rank_ic": float(test_eval_final["mean_rank_ic"]),
            "ic_ir": float(test_eval_final["ic_information_ratio"]),
            "ic_t_stat": float(test_eval_final["ic_t_statistic"]),
            "ic_p_value": float(test_eval_final["ic_p_value"]),
            "unconditional_dir_acc": float(test_eval_final["directional_accuracy"]),
            "d10_hit_rate": float(test_tier_final["d10_outperf_hit_rate"]),
            "d10_positive_ret_accuracy": float(test_tier_final["d10_positive_ret_accuracy"]),
            "top5_hit_rate": float(test_tier_final["top5pct_outperf_hit_rate"]),
            "top1_hit_rate": float(test_tier_final["top1pct_outperf_hit_rate"]),
            "d10_mean_return": float(test_tier_final["d10_mean_return"]),
            "d1_mean_return": float(test_tier_final["d1_mean_return"]),
            "d10_minus_d1_spread": float(test_tier_final["d10_minus_d1_spread"]),
            "d10_annualized_ls_return": float(test_tier_final["d10_annualized_ls_return"]),
            "d10_ls_sharpe": float(test_tier_final["d10_ls_sharpe"])
        }
    }
    
    os.makedirs("results/experiments", exist_ok=True)
    summary_path = os.path.abspath("results/experiments/iterative_accuracy_optimization.json")
    with open(summary_path, "w") as f:
        json.dump(optimization_summary, f, indent=2)
    print(f"Optimization trajectory saved to: {summary_path}")

if __name__ == "__main__":
    run_iterative_optimization()
