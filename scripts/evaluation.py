"""
Forecasting Evaluation and Statistical Analysis Module.
Calculates Daily Spearman Rank IC, IC Information Ratio, MAE, RMSE, Directional Accuracy,
and statistical significance metrics (t-statistics, p-values, paired differences).
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

def compute_daily_ic_series(df: pd.DataFrame, pred_col: str, target_col: str, date_col: str = "date") -> pd.Series:
    """
    Computes daily Spearman Rank IC across the cross-section of stocks for each date.
    Returns a pandas Series indexed by date.
    """
    def calc_spearman(sub):
        sub_valid = sub[[pred_col, target_col]].dropna()
        if len(sub_valid) < 10 or sub_valid[pred_col].nunique() <= 1 or sub_valid[target_col].nunique() <= 1:
            return 0.0
        corr, _ = stats.spearmanr(sub_valid[pred_col], sub_valid[target_col])
        return corr if not np.isnan(corr) else 0.0

    daily_ic = df.groupby(date_col).apply(calc_spearman, include_groups=False)
    return daily_ic.dropna()

def evaluate_forecast_performance(df: pd.DataFrame, pred_col: str, target_col: str, date_col: str = "date") -> Dict:
    """
    Computes complete publication-grade forecast metrics:
    - Daily Rank IC (Mean, Median, Std, IC IR, t-stat, p-value)
    - Pooled Error Metrics (MAE, RMSE, R2)
    - Directional Accuracy vs Baselines (Always-Up, Zero-Return)
    """
    valid_df = df[[date_col, pred_col, target_col]].dropna()
    if len(valid_df) == 0:
        return {}
        
    # 1. Daily Cross-Sectional Spearman Rank IC
    daily_ic = compute_daily_ic_series(valid_df, pred_col, target_col, date_col)
    n_days = len(daily_ic)
    
    if n_days > 1:
        mean_ic = float(daily_ic.mean())
        median_ic = float(daily_ic.median())
        std_ic = float(daily_ic.std(ddof=1))
        se_ic = std_ic / np.sqrt(n_days)
        ic_ir = float(mean_ic / (std_ic + 1e-8))
        # t-statistic and p-value for H0: Mean IC = 0
        t_stat, p_val = stats.ttest_1samp(daily_ic, 0.0)
        ic_skew = float(daily_ic.skew())
        pct_positive_ic = float((daily_ic > 0).mean())
    else:
        mean_ic = float(daily_ic.iloc[0]) if n_days == 1 else 0.0
        median_ic = mean_ic
        std_ic = 0.0
        se_ic = 0.0
        ic_ir = 0.0
        t_stat = 0.0
        p_val = 1.0
        ic_skew = 0.0
        pct_positive_ic = 1.0 if mean_ic > 0 else 0.0
        
    # 2. Regression / Error Metrics
    y_true = valid_df[target_col].values
    y_pred = valid_df[pred_col].values
    
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(errors ** 2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))
    
    # 3. Directional Accuracy
    # Note: For z-score targets, positive means outperforming the daily cross-sectional mean
    dir_true = np.sign(y_true)
    dir_pred = np.sign(y_pred)
    # Exclude exact zeros
    non_zero = (dir_true != 0) & (dir_pred != 0)
    if non_zero.sum() > 0:
        dir_acc = float((dir_true[non_zero] == dir_pred[non_zero]).mean())
    else:
        dir_acc = 0.5
        
    always_up_acc = float((dir_true > 0).mean())
    
    return {
        "num_days": n_days,
        "total_samples": len(valid_df),
        "mean_rank_ic": mean_ic,
        "median_rank_ic": median_ic,
        "std_rank_ic": std_ic,
        "se_rank_ic": float(se_ic),
        "ic_information_ratio": ic_ir,
        "ic_t_statistic": float(t_stat),
        "ic_p_value": float(p_val),
        "ic_skewness": ic_skew,
        "pct_positive_ic_days": pct_positive_ic,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "directional_accuracy": dir_acc,
        "always_up_accuracy": always_up_acc,
        "daily_ic_series": daily_ic
    }

def compare_models_statistical(ic_series_a: pd.Series, ic_series_b: pd.Series, name_a: str, name_b: str) -> Dict:
    """
    Performs paired statistical significance tests between two models on matched dates.
    """
    aligned = pd.concat([ic_series_a.rename(name_a), ic_series_b.rename(name_b)], axis=1).dropna()
    diff = aligned[name_a] - aligned[name_b]
    n = len(diff)
    if n < 5:
        return {}
        
    mean_diff = float(diff.mean())
    std_diff = float(diff.std(ddof=1))
    t_stat, p_val = stats.ttest_1samp(diff, 0.0)
    w_stat, w_pval = stats.wilcoxon(diff)
    cohen_d = float(mean_diff / (std_diff + 1e-8))
    
    return {
        "model_a": name_a,
        "model_b": name_b,
        "paired_days": n,
        "mean_ic_difference": mean_diff,
        "std_ic_difference": std_diff,
        "paired_t_stat": float(t_stat),
        "paired_t_pval": float(p_val),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pval": float(w_pval),
        "cohens_d": cohen_d
    }
