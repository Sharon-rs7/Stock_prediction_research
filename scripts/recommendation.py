"""
Top-5 Recommendation Engine and Evaluation Module.
Implements the 3 pre-specified recommendation methods:
Method A: Prediction-Only Top-5 (Highest predicted return scores across universe)
Method B: Similarity-Only Top-5 (Highest historical behavioral similarity to target stock)
Method C: Combined Top-5 (Pre-specified rank-fusion of similarity and predicted score)

Scoring Rule for Combined:
Rank_Sim(j) = Percentile Rank of similarity(target, j) in [0, 1]
Rank_Pred(j) = Percentile Rank of predicted_score(j) in [0, 1]
Score_Combined(j) = 0.5 * Rank_Sim(j) + 0.5 * Rank_Pred(j)
Top-5 stocks with highest Score_Combined(j), j != target.
Evaluated strictly against eligible cross-sectional universe benchmark return at date t.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

def run_recommendations_for_date(
    date: str,
    target_ticker: str,
    universe_tickers: List[str],
    pred_scores: np.ndarray,      # shape (N,)
    sim_scores: np.ndarray,       # shape (N,) similarity to target_ticker
    realized_returns: np.ndarray, # shape (N,) realized 5-day forward return
    k: int = 5,
    alpha: float = 0.5
) -> Dict:
    """
    Computes Method A, Method B, and Method C recommendations for a target stock on a given date.
    Excludes the target stock itself from its recommended peer set.
    """
    N = len(universe_tickers)
    if target_ticker not in universe_tickers:
        return None
        
    target_idx = universe_tickers.index(target_ticker)
    
    # Mask out target ticker itself
    eligible_mask = np.ones(N, dtype=bool)
    eligible_mask[target_idx] = False
    
    eligible_indices = np.where(eligible_mask)[0]
    eligible_tickers = [universe_tickers[i] for i in eligible_indices]
    
    sub_preds = pred_scores[eligible_indices]
    sub_sims = sim_scores[eligible_indices]
    sub_realized = realized_returns[eligible_indices]
    
    benchmark_return = float(np.mean(sub_realized))
    
    # ----------------------------------------------------
    # METHOD A: PREDICTION ONLY
    # ----------------------------------------------------
    pred_rank_indices = np.argsort(sub_preds)[::-1][:k]
    top_a_tickers = [eligible_tickers[i] for i in pred_rank_indices]
    top_a_returns = sub_realized[pred_rank_indices]
    
    # ----------------------------------------------------
    # METHOD B: SIMILARITY ONLY
    # ----------------------------------------------------
    sim_rank_indices = np.argsort(sub_sims)[::-1][:k]
    top_b_tickers = [eligible_tickers[i] for i in sim_rank_indices]
    top_b_returns = sub_realized[sim_rank_indices]
    
    # ----------------------------------------------------
    # METHOD C: COMBINED (Pre-specified 50/50 Rank Fusion)
    # ----------------------------------------------------
    # Compute normalized rank percentiles in [0, 1]
    n_sub = len(eligible_indices)
    pred_ranks = np.argsort(np.argsort(sub_preds)) / float(n_sub - 1 + 1e-8)
    sim_ranks = np.argsort(np.argsort(sub_sims)) / float(n_sub - 1 + 1e-8)
    
    combined_scores = alpha * sim_ranks + (1.0 - alpha) * pred_ranks
    combined_rank_indices = np.argsort(combined_scores)[::-1][:k]
    top_c_tickers = [eligible_tickers[i] for i in combined_rank_indices]
    top_c_returns = sub_realized[combined_rank_indices]
    
    return {
        "date": date,
        "target_ticker": target_ticker,
        "benchmark_return": benchmark_return,
        
        "method_a_tickers": top_a_tickers,
        "method_a_mean_return": float(np.mean(top_a_returns)),
        "method_a_median_return": float(np.median(top_a_returns)),
        "method_a_excess_return": float(np.mean(top_a_returns) - benchmark_return),
        "method_a_hit_rate": float(np.mean(top_a_returns > benchmark_return)),
        
        "method_b_tickers": top_b_tickers,
        "method_b_mean_return": float(np.mean(top_b_returns)),
        "method_b_median_return": float(np.median(top_b_returns)),
        "method_b_excess_return": float(np.mean(top_b_returns) - benchmark_return),
        "method_b_hit_rate": float(np.mean(top_b_returns > benchmark_return)),
        
        "method_c_tickers": top_c_tickers,
        "method_c_mean_return": float(np.mean(top_c_returns)),
        "method_c_median_return": float(np.median(top_c_returns)),
        "method_c_excess_return": float(np.mean(top_c_returns) - benchmark_return),
        "method_c_hit_rate": float(np.mean(top_c_returns > benchmark_return))
    }

def aggregate_recommendation_performance(rec_df: pd.DataFrame) -> Dict:
    """
    Summarizes performance across all evaluated recommendation instances.
    """
    summary = {}
    for prefix, method_name in [
        ("method_a", "Prediction_Only"),
        ("method_b", "Similarity_Only"),
        ("method_c", "Combined_Fusion")
    ]:
        excess = rec_df[f"{prefix}_excess_return"].values
        raw_ret = rec_df[f"{prefix}_mean_return"].values
        hit_rates = rec_df[f"{prefix}_hit_rate"].values
        
        # Paired t-statistic against benchmark
        mean_excess = np.mean(excess)
        std_excess = np.std(excess, ddof=1)
        se_excess = std_excess / np.sqrt(len(excess)) if len(excess) > 1 else 1e-8
        t_stat = mean_excess / (se_excess + 1e-8)
        
        summary[method_name] = {
            "mean_forward_return": float(np.mean(raw_ret)),
            "median_forward_return": float(np.median(raw_ret)),
            "mean_excess_return": float(mean_excess),
            "median_excess_return": float(np.median(excess)),
            "std_excess_return": float(std_excess),
            "se_excess_return": float(se_excess),
            "t_statistic": float(t_stat),
            "mean_hit_rate": float(np.mean(hit_rates)),
            "total_evaluations": len(rec_df)
        }
        
    benchmark_rets = rec_df["benchmark_return"].values
    summary["Universe_Benchmark"] = {
        "mean_forward_return": float(np.mean(benchmark_rets)),
        "median_forward_return": float(np.median(benchmark_rets)),
        "total_evaluations": len(rec_df)
    }
    
    return summary
