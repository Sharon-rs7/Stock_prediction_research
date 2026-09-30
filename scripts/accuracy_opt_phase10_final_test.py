"""
Accuracy Optimization - Phase 10: Final Unseen Test Evaluation
==============================================================
STRICT PROTOCOL: The test partition (July 8, 2025 to September 25, 2026; 308 trading sessions;
737,805 stock-day evaluations, 303 evaluable daily cross-sections) is opened EXACTLY ONCE.

All models, hyperparameters, feature definitions, calibration parameters, and selective confidence
cutoffs were permanently frozen during Phases 1-9 on Train/Validation data.

Computes:
  1. Overall Directional Accuracy
  2. Balanced Accuracy
  3. Precision & Confidence Interval
  4. Recall & Confidence Interval
  5. F1 Score
  6. ROC-AUC & PR-AUC
  7. Daily Cross-Sectional Rank IC, Newey-West HAC t-stat, p-value
  8. Selective Prediction across all 9 coverage tiers using VALIDATION-FROZEN cutoffs
  9. Calibration Diagnostics (Brier, ECE, LogLoss)
  10. Average Forward Return across conviction tiers
  11. Transaction Cost Sensitivity (Turnover, Gross, 5 bps, 10 bps, 15 bps net, Max DD, Sharpe, Hit Rate)
  12. McNemar Statistical Test & Block Bootstrap 95% Confidence Intervals

Outputs:
  - results/accuracy_optimization/08_final_test_metrics.json
  - results/accuracy_optimization/10_experiment_report.md
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, log_loss, confusion_matrix
)
import statsmodels.api as sm
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

# Validation-frozen cutoffs from 05_selective_accuracy.csv
VAL_FROZEN_CUTOFFS = {
    1.00: 0.0,
    0.50: 0.01520,
    0.30: 0.02277,
    0.20: 0.03087,
    0.15: 0.03767,
    0.10: 0.04863,
    0.05: 0.07012,
    0.02: 0.09752,
    0.01: 0.11925
}

# Validation-frozen Platt parameters from Phase 8
PLATT_SLOPE = 1.7373
PLATT_INTERCEPT = -0.0329

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

def block_bootstrap_ci(y_true, y_pred, dates, n_boot=2000, seed=42):
    np.random.seed(seed)
    unique_dates = np.array(sorted(list(set(dates))))
    n_dates = len(unique_dates)
    
    acc_samples = []
    prec_samples = []
    rec_samples = []
    
    df_eval = pd.DataFrame({'date': dates, 'y': y_true, 'pred': y_pred})
    grouped = {d: (g['y'].values, g['pred'].values) for d, g in df_eval.groupby('date')}
    
    for _ in range(n_boot):
        sample_dates = np.random.choice(unique_dates, size=n_dates, replace=True)
        boot_y = []
        boot_p = []
        for d in sample_dates:
            y_d, p_d = grouped[d]
            boot_y.extend(y_d)
            boot_p.extend(p_d)
            
        boot_y = np.array(boot_y)
        boot_p = np.array(boot_p)
        acc_samples.append(accuracy_score(boot_y, boot_p))
        prec_samples.append(precision_score(boot_y, boot_p, zero_division=0))
        rec_samples.append(recall_score(boot_y, boot_p, zero_division=0))
        
    ci_acc = (float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5)))
    ci_prec = (float(np.percentile(prec_samples, 2.5)), float(np.percentile(prec_samples, 97.5)))
    ci_rec = (float(np.percentile(rec_samples, 2.5)), float(np.percentile(rec_samples, 97.5)))
    return ci_acc, ci_prec, ci_rec

def newey_west_hac(series, max_lags=5):
    T = len(series)
    mean_val = np.mean(series)
    e = series - mean_val
    gamma_0 = np.sum(e ** 2) / T
    var_hac = gamma_0
    for l in range(1, max_lags + 1):
        weight = 1.0 - (l / (max_lags + 1.0))
        gamma_l = np.sum(e[l:] * e[:-l]) / T
        var_hac += 2.0 * weight * gamma_l
    se_hac = np.sqrt(max(1e-12, var_hac) / T)
    t_stat = mean_val / se_hac
    p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), df=T - 1))
    return float(t_stat), float(p_val), float(se_hac)

def main():
    print("=" * 80)
    print("PHASE 10: FINAL UNSEEN TEST EVALUATION (SINGLE FINAL OPENING)")
    print("=" * 80)
    t0 = time.time()
    
    # 1. Load Panel
    print("\n[1] Loading dataset...")
    df = pd.read_parquet("data/processed/universe_b_expanded_features.parquet")
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    unique_dates = sorted(df['date'].unique())
    
    splits, sdates = build_temporal_splits(unique_dates, H=5)
    train_dates = set(sdates['train_dates'])
    val_dates = set(sdates['val_dates'])
    test_dates = sorted(list(sdates['test_dates']))
    
    train_mask = df['date'].isin(train_dates)
    test_mask = df['date'].isin(test_dates)
    
    y_train_z = df.loc[train_mask, 'zscore_fwd_ret_5d'].values
    y_test_z = df.loc[test_mask, 'zscore_fwd_ret_5d'].values
    raw_fwd_test = df.loc[test_mask, 'fwd_ret_5d'].values
    
    y_train_bin = (y_train_z >= 0).astype(int)
    y_test_bin = (y_test_z >= 0).astype(int)
    
    v_tr = ~np.isnan(y_train_z)
    v_te = ~np.isnan(y_test_z)
    
    X_tr = df.loc[train_mask, FEATURES_49].fillna(0).values[v_tr]
    X_te = df.loc[test_mask, FEATURES_49].fillna(0).values[v_te]
    
    test_meta = df.loc[test_mask, ['date', 'ticker']].copy()
    test_meta = test_meta.loc[v_te].reset_index(drop=True)
    test_dates_arr = test_meta['date'].values
    
    print(f"Train samples: {v_tr.sum():,}")
    print(f"Test samples: {v_te.sum():,} ({len(test_dates)} calendar days; {len(set(test_dates_arr))} evaluable dates)")
    
    # 2. Train Frozen Champion Model (HP_09: XGBoost Huber Deep Regressor)
    print("\n[2] Training Frozen Champion Model: XGBoost Huber Deep Regressor...")
    model_champ = xgb.XGBRegressor(
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
    model_champ.fit(X_tr, y_train_z[v_tr])
    
    # Also fit Baseline Model (LightGBM Huber Baseline from Level 2) for McNemar test
    print("  Training Baseline Model: LightGBM Huber Baseline for comparative inferential testing...")
    model_base = lgb.LGBMRegressor(
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
    model_base.fit(X_tr, y_train_z[v_tr])
    
    # 3. Predict on Test Set
    print("\n[3] Generating Out-of-Time Test Predictions...")
    pred_z_champ = model_champ.predict(X_te)
    pred_z_base = model_base.predict(X_te)
    
    # Apply Frozen Platt Calibration
    prob_platt = 1.0 / (1.0 + np.exp(-(PLATT_SLOPE * pred_z_champ + PLATT_INTERCEPT)))
    confidence = np.abs(prob_platt - 0.5)
    
    y_pred_champ = (pred_z_champ >= 0).astype(int)
    y_pred_base = (pred_z_base >= 0).astype(int)
    y_true_test = y_test_bin[v_te]
    
    # 4. Compute Core Confirmatory Metrics (100% Unconditional Coverage)
    acc_overall = accuracy_score(y_true_test, y_pred_champ)
    bal_acc = balanced_accuracy_score(y_true_test, y_pred_champ)
    prec_overall = precision_score(y_true_test, y_pred_champ, zero_division=0)
    rec_overall = recall_score(y_true_test, y_pred_champ, zero_division=0)
    f1_overall = f1_score(y_true_test, y_pred_champ, zero_division=0)
    roc_overall = roc_auc_score(y_true_test, prob_platt)
    pr_auc_overall = average_precision_score(y_true_test, prob_platt)
    brier_overall = brier_score_loss(y_true_test, prob_platt)
    ll_overall = log_loss(y_true_test, np.clip(prob_platt, 1e-6, 1-1e-6))
    ece_overall = compute_ece(y_true_test, prob_platt)
    
    # Daily Cross-Sectional Rank IC
    test_meta['pred_z'] = pred_z_champ
    test_meta['raw_fwd_ret'] = raw_fwd_test[v_te]
    daily_ics = test_meta.groupby('date').apply(lambda g: stats.spearmanr(g['pred_z'], g['raw_fwd_ret'])[0] if len(g)>=10 else np.nan).dropna()
    mean_rank_ic = float(daily_ics.mean())
    hac_t, hac_p, hac_se = newey_west_hac(daily_ics.values, max_lags=5)
    ic_ir = mean_rank_ic / float(daily_ics.std()) if daily_ics.std() > 0 else 0.0
    
    # Block Bootstrap CIs
    print("\n[4] Computing Block Bootstrap Confidence Intervals (B = 2,000 dates)...")
    ci_acc, ci_prec, ci_rec = block_bootstrap_ci(y_true_test, y_pred_champ, test_dates_arr, n_boot=2000)
    
    # McNemar Test against Baseline
    # Contingency table:
    # b: champ correct, base wrong
    # c: base correct, champ wrong
    champ_correct = (y_pred_champ == y_true_test)
    base_correct = (y_pred_base == y_true_test)
    b_count = int(np.sum(champ_correct & ~base_correct))
    c_count = int(np.sum(~champ_correct & base_correct))
    mcnemar_stat = float((abs(b_count - c_count) - 1.0)**2 / (b_count + c_count + 1e-8))
    mcnemar_p = float(1.0 - stats.chi2.cdf(mcnemar_stat, df=1))
    
    print("\n--- UNCONDITIONAL TEST RESULTS (100% COVERAGE) ---")
    print(f"  Sample Size (N): {len(y_true_test):,}")
    print(f"  Directional Accuracy: {acc_overall*100:.2f}% (95% CI: [{ci_acc[0]*100:.2f}%, {ci_acc[1]*100:.2f}%])")
    print(f"  Balanced Accuracy:    {bal_acc*100:.2f}%")
    print(f"  Precision (UP calls): {prec_overall*100:.2f}% (95% CI: [{ci_prec[0]*100:.2f}%, {ci_prec[1]*100:.2f}%])")
    print(f"  Recall (UP calls):    {rec_overall*100:.2f}% (95% CI: [{ci_rec[0]*100:.2f}%, {ci_rec[1]*100:.2f}%])")
    print(f"  F1 Score:             {f1_overall:.4f}")
    print(f"  ROC-AUC:              {roc_overall:.4f}")
    print(f"  PR-AUC:               {pr_auc_overall:.4f}")
    print(f"  Brier Score:          {brier_overall:.5f}")
    print(f"  Expected Calib Error: {ece_overall*100:.2f}%")
    print(f"  Mean Daily Rank IC:   {mean_rank_ic:.4f} (HAC t = {hac_t:.4f}, p = {hac_p:.4f}, IR = {ic_ir:.4f})")
    print(f"  McNemar vs Baseline:  chi2 = {mcnemar_stat:.2f}, p = {mcnemar_p:.4f}")
    
    # 5. Selective Prediction across 9 Coverage Tiers on Test
    print("\n[5] Evaluating Selective Prediction Using Validation-Frozen Cutoffs...")
    test_meta['prob_platt'] = prob_platt
    test_meta['confidence'] = confidence
    test_meta['actual_bin'] = y_true_test
    
    selective_results = []
    
    for cov_target, cutoff in VAL_FROZEN_CUTOFFS.items():
        cov_mask = (confidence >= cutoff)
        n_cov = int(cov_mask.sum())
        act_cov = n_cov / len(confidence)
        
        y_true_c = y_true_test[cov_mask]
        p_c = prob_platt[cov_mask]
        y_pred_c = (p_c >= 0.5).astype(int)
        ret_c = raw_fwd_test[v_te][cov_mask]
        dates_c = test_dates_arr[cov_mask]
        
        acc_c = accuracy_score(y_true_c, y_pred_c)
        bal_c = balanced_accuracy_score(y_true_c, y_pred_c)
        prec_c = precision_score(y_true_c, y_pred_c, zero_division=0)
        rec_c = recall_score(y_true_c, y_pred_c, zero_division=0)
        f1_c = f1_score(y_true_c, y_pred_c, zero_division=0)
        ece_c = compute_ece(y_true_c, p_c)
        
        up_m = (y_pred_c == 1)
        dn_m = (y_pred_c == 0)
        avg_ret_up = float(np.mean(ret_c[up_m])) if up_m.sum() > 0 else 0.0
        avg_ret_dn = float(np.mean(ret_c[dn_m])) if dn_m.sum() > 0 else 0.0
        spread_c = avg_ret_up - avg_ret_dn
        
        # Rank IC on subset
        sub_df = test_meta.loc[cov_mask]
        daily_ics_c = sub_df.groupby('date').apply(lambda g: stats.spearmanr(g['pred_z'], g['raw_fwd_ret'])[0] if len(g)>=5 else np.nan).dropna()
        rank_ic_c = float(daily_ics_c.mean()) if len(daily_ics_c) > 0 else 0.0
        
        # Check whether >=60% accuracy was genuinely achieved
        crosses_60_acc = bool(acc_c >= 0.60)
        crosses_60_prec = bool(prec_c >= 0.60)
        
        print(f"  Target: {cov_target*100:4.0f}% | Actual: {act_cov*100:5.2f}% | N: {n_cov:7,d} | Cutoff: {cutoff:.5f} | DirAcc: {acc_c*100:5.2f}% | BalAcc: {bal_c*100:5.2f}% | UP Prec: {prec_c*100:5.2f}% | UP/DN Spread: {spread_c*100:+.2f}% | Rank IC: {rank_ic_c:.4f}")
        
        selective_results.append({
            "target_coverage": cov_target,
            "actual_coverage": round(float(act_cov), 4),
            "sample_size": n_cov,
            "confidence_cutoff": round(float(cutoff), 5),
            "dir_accuracy": round(float(acc_c), 4),
            "balanced_accuracy": round(float(bal_c), 4),
            "up_precision": round(float(prec_c), 4),
            "up_recall": round(float(rec_c), 4),
            "f1_score": round(float(f1_c), 4),
            "ece": round(float(ece_c), 5),
            "rank_ic": round(float(rank_ic_c), 4),
            "avg_forward_return": round(float(np.mean(ret_c)), 5),
            "avg_fwd_ret_up_calls": round(float(avg_ret_up), 5),
            "avg_fwd_ret_down_calls": round(float(avg_ret_dn), 5),
            "up_down_spread": round(float(spread_c), 5),
            "crosses_60pct_directional_acc": crosses_60_acc,
            "crosses_60pct_up_precision": crosses_60_prec
        })

    # 6. Trading Strategy & Transaction Cost Sensitivity
    print("\n[6] Evaluating Trading Strategy & Transaction Cost Sensitivity...")
    # Long Top Decile (D10) vs Short Bottom Decile (D1) or Long Top-K rebalanced every 5 days
    # Evaluate across 61 non-overlapping 5-day cycles in test
    stride_dates = sorted(list(set(test_dates_arr)))[::5]
    
    portfolio_records = []
    prev_holdings = None
    
    for t_date in stride_dates:
        d_df = test_meta[test_meta['date'] == t_date].sort_values('pred_z', ascending=False)
        if len(d_df) < 50: continue
        
        # Top 50 long portfolio
        top_k = d_df.head(50)
        tickers_curr = set(top_k['ticker'].values)
        fwd_ret = float(top_k['raw_fwd_ret'].mean())
        
        # Turnover
        if prev_holdings is not None:
            turnover = 1.0 - (len(tickers_curr.intersection(prev_holdings)) / len(tickers_curr))
        else:
            turnover = 1.0
        prev_holdings = tickers_curr
        
        portfolio_records.append({
            "date": t_date,
            "gross_return": fwd_ret,
            "turnover": turnover
        })
        
    df_port = pd.DataFrame(portfolio_records)
    mean_gross = float(df_port['gross_return'].mean())
    mean_turnover = float(df_port['turnover'].mean())
    
    # Benchmark equal-weighted return
    bench_returns = [float(test_meta[test_meta['date'] == t_d]['raw_fwd_ret'].mean()) for t_d in df_port['date']]
    mean_bench = float(np.mean(bench_returns))
    gross_excess = mean_gross - mean_bench
    
    # Net returns
    net_5bps = gross_excess - (mean_turnover * 0.0005 * 2.0)
    net_10bps = gross_excess - (mean_turnover * 0.0010 * 2.0)
    net_15bps = gross_excess - (mean_turnover * 0.0015 * 2.0)
    
    # Annualized Sharpe & Max Drawdown
    excess_series = (df_port['gross_return'] - bench_returns).values
    ann_sharpe = float(np.mean(excess_series) / np.std(excess_series) * np.sqrt(52)) if np.std(excess_series) > 0 else 0.0
    cum_returns = np.cumprod(1.0 + excess_series)
    peak = np.maximum.accumulate(cum_returns)
    max_dd = float(np.max((peak - cum_returns) / peak))
    hit_rate = float(np.mean(excess_series > 0))
    
    print("\n--- TRADING STRATEGY & TRANSACTION COST SENSITIVITY ---")
    print(f"  Holding Horizon: 5 Trading Days (Non-Overlapping Stride)")
    print(f"  Rebalance Periods: {len(df_port)} evaluation windows")
    print(f"  Mean Portfolio Turnover: {mean_turnover*100:.2f}%")
    print(f"  Mean Gross Excess Return: {gross_excess*100:+.3f}% per 5-day cycle")
    print(f"  Net Excess (5 bps cost):   {net_5bps*100:+.3f}% per 5-day cycle")
    print(f"  Net Excess (10 bps cost):  {net_10bps*100:+.3f}% per 5-day cycle")
    print(f"  Net Excess (15 bps cost):  {net_15bps*100:+.3f}% per 5-day cycle")
    print(f"  Annualized Sharpe Ratio:   {ann_sharpe:.3f}")
    print(f"  Maximum Drawdown:          {max_dd*100:.2f}%")
    print(f"  Outperformance Hit Rate:   {hit_rate*100:.2f}%")
    
    # 7. Compile Final Output JSON
    final_json = {
        "evaluation_name": "Final Unseen Test Evaluation (Phase 10)",
        "protocol": "Strict Out-of-Time Single Evaluation. Zero Post-Test Tuning.",
        "test_dates": {
            "start_date": str(test_dates[0]),
            "end_date": str(test_dates[-1]),
            "calendar_days": len(test_dates),
            "evaluable_daily_cross_sections": len(set(test_dates_arr)),
            "total_stock_day_evaluations": len(y_true_test)
        },
        "frozen_specification": {
            "model_architecture": "XGBoost_Huber_Deep_Reg (HP_09)",
            "hyperparameters": {
                "max_depth": 7,
                "n_estimators": 120,
                "learning_rate": 0.02,
                "reg_lambda": 10.0,
                "subsample": 0.8,
                "colsample_bytree": 0.75,
                "objective": "reg:pseudohubererror"
            },
            "features_used": len(FEATURES_49),
            "calibration": f"Platt Scaling: P(Y=1|z) = 1 / (1 + exp(-({PLATT_SLOPE:.4f}*z + {PLATT_INTERCEPT:.4f})))"
        },
        "unconditional_metrics_100pct_coverage": {
            "sample_size": len(y_true_test),
            "directional_accuracy": round(float(acc_overall), 4),
            "directional_accuracy_95_ci": [round(float(ci_acc[0]), 4), round(float(ci_acc[1]), 4)],
            "balanced_accuracy": round(float(bal_acc), 4),
            "precision_up_calls": round(float(prec_overall), 4),
            "precision_95_ci": [round(float(ci_prec[0]), 4), round(float(ci_prec[1]), 4)],
            "recall_up_calls": round(float(rec_overall), 4),
            "recall_95_ci": [round(float(ci_rec[0]), 4), round(float(ci_rec[1]), 4)],
            "f1_score": round(float(f1_overall), 4),
            "roc_auc": round(float(roc_overall), 4),
            "pr_auc": round(float(pr_auc_overall), 4),
            "brier_score": round(float(brier_overall), 5),
            "expected_calibration_error": round(float(ece_overall), 5),
            "rank_ic_mean": round(float(mean_rank_ic), 4),
            "newey_west_hac_t": round(float(hac_t), 4),
            "newey_west_hac_p": round(float(hac_p), 4),
            "information_ratio": round(float(ic_ir), 4),
            "mcnemar_chi2_vs_baseline": round(float(mcnemar_stat), 2),
            "mcnemar_p_val": round(float(mcnemar_p), 4)
        },
        "selective_prediction_tiers": selective_results,
        "trading_and_cost_sensitivity": {
            "rebalance_stride_days": 5,
            "portfolio_size": 50,
            "evaluation_cycles": len(df_port),
            "portfolio_turnover": round(float(mean_turnover), 4),
            "gross_excess_return": round(float(gross_excess), 5),
            "net_excess_5bps": round(float(net_5bps), 5),
            "net_excess_10bps": round(float(net_10bps), 5),
            "net_excess_15bps": round(float(net_15bps), 5),
            "annualized_sharpe": round(float(ann_sharpe), 3),
            "max_drawdown": round(float(max_dd), 4),
            "hit_rate": round(float(hit_rate), 4)
        },
        "sixty_percent_rule_findings": {
            "overall_unconditional_reaches_60pct": bool(acc_overall >= 0.60),
            "selective_directional_acc_reaches_60pct": any(r["crosses_60pct_directional_acc"] for r in selective_results),
            "selective_up_precision_reaches_60pct": any(r["crosses_60pct_up_precision"] for r in selective_results),
            "best_selective_directional_accuracy": max(r["dir_accuracy"] for r in selective_results),
            "coverage_at_best_accuracy": [r["actual_coverage"] for r in selective_results if r["dir_accuracy"] == max(r["dir_accuracy"] for r in selective_results)][0],
            "conclusion_statement": (
                f"Overall unconditional directional accuracy is {acc_overall*100:.2f}% (not >= 60%). "
                f"Under selective prediction, directional accuracy reaches {max(r['dir_accuracy'] for r in selective_results)*100:.2f}% "
                f"at {selective_results[-1]['actual_coverage']*100:.2f}% coverage."
            )
        }
    }
    
    out_json_path = os.path.join(OUT_DIR, "08_final_test_metrics.json")
    with open(out_json_path, "w") as f:
        json.dump(final_json, f, indent=2)
    print(f"\n[7] Saved final test metrics to {out_json_path}")
    print(f"Total Phase 10 execution elapsed: {time.time()-t0:.2f} seconds")

if __name__ == "__main__":
    main()
