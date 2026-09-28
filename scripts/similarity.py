"""
Similarity Engine Module.
Implements historical similarity algorithms with strictly backward-looking windows.
Supports lookback horizons L = 252 (1-year) and L = 504 (2-year).
Candidate similarity metrics:
1. Pearson Correlation on historical daily returns:
   r_ij = corr(r_i[t-L+1:t], r_j[t-L+1:t])
2. Cosine Similarity on historical normalized multi-factor representation:
   cos_ij = (x_i . x_j) / (||x_i|| * ||x_j||)
3. Trajectory Euclidean Proximity:
   Normalized cumulative return trajectory distance:
   d_ij = 1 / (1 + ||traj_i - traj_j||_2)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def compute_return_correlation_similarity(return_matrix: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Pearson correlation between stock return vectors over lookback L.
    Input return_matrix shape: (L, N_stocks)
    Returns: (N_stocks, N_stocks) pairwise correlation matrix in [-1, 1].
    """
    # Center each column
    mean = np.mean(return_matrix, axis=0, keepdims=True)
    centered = return_matrix - mean
    # Standard deviation
    std = np.std(centered, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    normed = centered / std
    # Covariance / correlation = (1 / L) * normed.T @ normed
    L = return_matrix.shape[0]
    corr_matrix = (normed.T @ normed) / float(L)
    # Clip numerical precision
    return np.clip(corr_matrix, -1.0, 1.0)

def compute_factor_cosine_similarity(factor_matrix: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Cosine similarity between stock factor vectors.
    Input factor_matrix shape: (N_stocks, N_factors)
    Returns: (N_stocks, N_stocks) pairwise cosine matrix in [-1, 1].
    """
    norms = np.linalg.norm(factor_matrix, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1e-8, norms)
    normed_factors = factor_matrix / norms
    return np.clip(normed_factors @ normed_factors.T, -1.0, 1.0)

def compute_trajectory_similarity(return_matrix: np.ndarray) -> np.ndarray:
    """
    Computes trajectory proximity based on Euclidean distance between normalized
    cumulative return trajectories over lookback L.
    Input return_matrix shape: (L, N_stocks)
    """
    # Cumulative return paths starting at 0
    cum_paths = np.vstack([np.zeros((1, return_matrix.shape[1])), np.cumsum(return_matrix, axis=0)]) # shape (L+1, N)
    # Normalized by path standard deviation to make trajectories scale-invariant
    path_std = np.std(cum_paths, axis=0, keepdims=True)
    path_std = np.where(path_std < 1e-8, 1e-8, path_std)
    normed_paths = cum_paths / path_std # (L+1, N)
    
    # Pairwise squared Euclidean distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    dot = normed_paths.T @ normed_paths # (N, N)
    diag = np.diag(dot)
    dist_sq = np.maximum(0.0, diag[:, None] + diag[None, :] - 2.0 * dot)
    dist = np.sqrt(dist_sq)
    # Proximity score in (0, 1]
    proximity = 1.0 / (1.0 + dist / np.sqrt(return_matrix.shape[0]))
    return proximity

def get_top_k_similar_stocks(sim_matrix: np.ndarray, ticker_list: List[str], target_ticker: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Returns top-k most similar peers for a target stock, excluding itself.
    """
    if target_ticker not in ticker_list:
        return []
    idx = ticker_list.index(target_ticker)
    sim_scores = sim_matrix[idx].copy()
    sim_scores[idx] = -np.inf # Exclude self
    
    top_indices = np.argsort(sim_scores)[::-1][:k]
    return [(ticker_list[i], float(sim_scores[i])) for i in top_indices]
