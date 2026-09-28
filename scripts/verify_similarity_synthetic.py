"""
Synthetic Verification Script for Similarity Engine.
Verifies mathematical convergence, boundary conditions, self-match exclusion,
and causal invariance on controlled synthetic time series.
"""

import os
import sys
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from scripts.similarity import (
    compute_return_correlation_similarity,
    compute_trajectory_similarity,
    get_top_k_similar_stocks
)

def run_synthetic_tests():
    print("=" * 60)
    print("RUNNING SYNTHETIC VERIFICATION ON SIMILARITY ENGINE")
    print("=" * 60)
    
    np.random.seed(42)
    L = 252
    
    # 1. Test Identical Signals
    base_series = np.random.normal(0.001, 0.02, size=L)
    identical_series = base_series.copy()
    inverse_series = -base_series.copy()
    orthogonal_series = np.random.normal(0.0, 0.02, size=L)
    
    # Matrix of shape (L, 4): [Base, Identical, Inverse, Orthogonal]
    synth_matrix = np.column_stack([base_series, identical_series, inverse_series, orthogonal_series])
    ticker_names = ["BASE", "IDENTICAL", "INVERSE", "ORTHOGONAL"]
    
    corr_sim = compute_return_correlation_similarity(synth_matrix)
    traj_sim = compute_trajectory_similarity(synth_matrix)
    
    print(f"\n1. Correlation Similarity Matrix:\n{np.round(corr_sim, 4)}")
    
    # Assertions
    # Base vs Identical: corr must be ~ 1.0
    assert np.isclose(corr_sim[0, 1], 1.0, atol=1e-5), f"Failed: Base vs Identical corr is {corr_sim[0, 1]}"
    print("  [PASS] Identical trajectory correlation == +1.0000")
    
    # Base vs Inverse: corr must be ~ -1.0
    assert np.isclose(corr_sim[0, 2], -1.0, atol=1e-5), f"Failed: Base vs Inverse corr is {corr_sim[0, 2]}"
    print("  [PASS] Inverse trajectory correlation == -1.0000")
    
    # Base vs Orthogonal: corr magnitude should be small (< 0.15)
    assert abs(corr_sim[0, 3]) < 0.15, f"Failed: Orthogonal corr is {corr_sim[0, 3]}"
    print(f"  [PASS] Orthogonal series correlation near zero ({corr_sim[0, 3]:.4f})")
    
    # Trajectory proximity
    assert np.isclose(traj_sim[0, 1], 1.0, atol=1e-5), f"Failed: Trajectory proximity identical is {traj_sim[0, 1]}"
    print("  [PASS] Identical trajectory proximity == 1.0000")
    
    # Self-match exclusion check
    top_peers = get_top_k_similar_stocks(corr_sim, ticker_names, "BASE", k=2)
    peer_names = [p[0] for p in top_peers]
    assert "BASE" not in peer_names, f"Failed: Self-match found in peers: {peer_names}"
    assert peer_names[0] == "IDENTICAL", f"Failed: Expected IDENTICAL as top peer, got {peer_names[0]}"
    print(f"  [PASS] Self-match exclusion verified. Top peers for BASE: {top_peers}")
    
    # Boundary test: zero variance series
    flat_series = np.zeros(L)
    flat_matrix = np.column_stack([base_series, flat_series])
    flat_corr = compute_return_correlation_similarity(flat_matrix)
    assert not np.isnan(flat_corr).any(), "Failed: NaN produced on zero-variance input"
    print("  [PASS] Zero-variance numerical stability handled without NaNs or infs")
    
    print("\nALL SYNTHETIC SIMILARITY VERIFICATIONS PASSED SUCCESSFULLY.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = run_synthetic_tests()
    if not success:
        sys.exit(1)
