import os
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from scripts.features import FEATURE_COLUMNS
from scripts.temporal_split import build_temporal_splits
from scripts.evaluation import evaluate_forecast_performance

def main():
    print("Loading panel for H=1 Day Champion Testing...")
    df = pd.read_parquet("data/processed/universe_b_panel.parquet")
    df["date"] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d')
    splits, sdates = build_temporal_splits(sorted(df["date"].unique()), H=1)

    # Clean mask for H=1
    mask = df[FEATURE_COLUMNS + ["zscore_fwd_ret_1d", "fwd_ret_1d"]].notna().all(axis=1)
    df_clean = df[mask].copy()

    tr = df_clean[df_clean["date"].isin(sdates["train_dates"])]
    va = df_clean[df_clean["date"].isin(sdates["val_dates"])]
    te = df_clean[df_clean["date"].isin(sdates["test_dates"])]

    print(f"Train: {len(tr):,}, Val: {len(va):,}, Test: {len(te):,}")

    X_tr = tr[FEATURE_COLUMNS].values
    y_tr = tr["zscore_fwd_ret_1d"].values
    X_va = va[FEATURE_COLUMNS].values
    y_va = va["zscore_fwd_ret_1d"].values
    X_te = te[FEATURE_COLUMNS].values
    y_te = te["zscore_fwd_ret_1d"].values

    # 1. LightGBM Huber
    print("\nFitting LightGBM Huber (H=1d)...")
    m_lgb = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31, objective="huber",
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
    )
    m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(30, verbose=False)])

    # 2. XGBoost GPU Huber
    print("Fitting XGBoost GPU Huber (H=1d)...")
    m_xgb = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.8, objective="reg:pseudohubererror", tree_method="hist",
        device="cuda", random_state=42
    )
    m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    p_va_lgb = m_lgb.predict(X_va)
    p_te_lgb = m_lgb.predict(X_te)
    p_va_xgb = m_xgb.predict(X_va)
    p_te_xgb = m_xgb.predict(X_te)

    # 50/50 Ensemble
    p_va_ens = 0.5 * p_va_lgb + 0.5 * p_va_xgb
    p_te_ens = 0.5 * p_te_lgb + 0.5 * p_te_xgb

    print("\n" + "="*60)
    print("EVALUATION ON VALIDATION (2024-04 to 2025-06):")
    ev_va_lgb = evaluate_forecast_performance(va.assign(pred=p_va_lgb), pred_col="pred", target_col="zscore_fwd_ret_1d")
    ev_va_xgb = evaluate_forecast_performance(va.assign(pred=p_va_xgb), pred_col="pred", target_col="zscore_fwd_ret_1d")
    ev_va_ens = evaluate_forecast_performance(va.assign(pred=p_va_ens), pred_col="pred", target_col="zscore_fwd_ret_1d")
    print(f"  LightGBM  -> Mean Rank IC: {ev_va_lgb['mean_rank_ic']:.4f} | t-stat: {ev_va_lgb['ic_t_statistic']:.2f} | IR: {ev_va_lgb['ic_information_ratio']:.3f} | DirAcc: {ev_va_lgb['directional_accuracy']*100:.2f}%")
    print(f"  XGBoost   -> Mean Rank IC: {ev_va_xgb['mean_rank_ic']:.4f} | t-stat: {ev_va_xgb['ic_t_statistic']:.2f} | IR: {ev_va_xgb['ic_information_ratio']:.3f} | DirAcc: {ev_va_xgb['directional_accuracy']*100:.2f}%")
    print(f"  Ensemble  -> Mean Rank IC: {ev_va_ens['mean_rank_ic']:.4f} | t-stat: {ev_va_ens['ic_t_statistic']:.2f} | IR: {ev_va_ens['ic_information_ratio']:.3f} | DirAcc: {ev_va_ens['directional_accuracy']*100:.2f}%")

    print("\n" + "="*60)
    print("EVALUATION ON TEST PARTITION (2025-07 to 2026-09):")
    ev_te_lgb = evaluate_forecast_performance(te.assign(pred=p_te_lgb), pred_col="pred", target_col="zscore_fwd_ret_1d")
    ev_te_xgb = evaluate_forecast_performance(te.assign(pred=p_te_xgb), pred_col="pred", target_col="zscore_fwd_ret_1d")
    ev_te_ens = evaluate_forecast_performance(te.assign(pred=p_te_ens), pred_col="pred", target_col="zscore_fwd_ret_1d")
    print(f"  LightGBM  -> Mean Rank IC: {ev_te_lgb['mean_rank_ic']:.4f} | t-stat: {ev_te_lgb['ic_t_statistic']:.2f} | IR: {ev_te_lgb['ic_information_ratio']:.3f} | DirAcc: {ev_te_lgb['directional_accuracy']*100:.2f}%")
    print(f"  XGBoost   -> Mean Rank IC: {ev_te_xgb['mean_rank_ic']:.4f} | t-stat: {ev_te_xgb['ic_t_statistic']:.2f} | IR: {ev_te_xgb['ic_information_ratio']:.3f} | DirAcc: {ev_te_xgb['directional_accuracy']*100:.2f}%")
    print(f"  Ensemble  -> Mean Rank IC: {ev_te_ens['mean_rank_ic']:.4f} | t-stat: {ev_te_ens['ic_t_statistic']:.2f} | IR: {ev_te_ens['ic_information_ratio']:.3f} | DirAcc: {ev_te_ens['directional_accuracy']*100:.2f}%")

    # Let's inspect deciles and high-confidence returns on Test
    def get_decile_spread(df_in, pred_col, ret_col):
        def dec(s):
            s = s.copy()
            s["d"] = pd.qcut(s[pred_col].rank(method="first"), 10, labels=False) + 1
            return s
        ranked = df_in.groupby("date", group_keys=False).apply(dec)
        d10 = ranked[ranked["d"] == 10]
        d1 = ranked[ranked["d"] == 1]
        daily_d10 = d10.groupby("date")[ret_col].mean()
        daily_d1 = d1.groupby("date")[ret_col].mean()
        spread = (daily_d10 - daily_d1).dropna()
        ann_ret = spread.mean() * 252
        sharpe = spread.mean() / (spread.std(ddof=1) + 1e-8) * np.sqrt(252)
        d10_ret = d10[ret_col].mean()
        d1_ret = d1[ret_col].mean()
        d10_hit = (d10["zscore_fwd_ret_1d"] > 0).mean()
        d10_pos = (d10[ret_col] > 0).mean()
        return d10_ret, d1_ret, ann_ret, sharpe, d10_hit, d10_pos

    d10_ret, d1_ret, ann_ret, sharpe, d10_hit, d10_pos = get_decile_spread(te.assign(pred=p_te_ens), "pred", "fwd_ret_1d")
    print("\n" + "="*60)
    print("TEST DECILE & TRADING ACCURACY (H=1d Ensemble):")
    print(f"  Decile 10 (Top 10% Pred) Daily Return:  {d10_ret*100:+.3f}%")
    print(f"  Decile 1  (Bottom 10%) Daily Return:    {d1_ret*100:+.3f}%")
    print(f"  D10 - D1 Long-Short Spread (Daily):     {(d10_ret-d1_ret)*100:+.3f}%")
    print(f"  Annualized Long-Short Return:           {ann_ret*100:+.2f}%")
    print(f"  Long-Short Sharpe Ratio:                {sharpe:.2f}")
    print(f"  Decile 10 Outperformance Hit Rate:     {d10_hit*100:.2f}%")
    print(f"  Decile 10 Positive Day Win Rate:        {d10_pos*100:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
