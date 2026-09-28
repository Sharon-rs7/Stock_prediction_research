"""
Reproducibility Package Generator.
Generates machine-readable manifest and environment logs for academic verification.
"""

import os
import sys
import json
import platform
import hashlib
from datetime import datetime
import pandas as pd
import numpy as np

METADATA_DIR = os.path.abspath("metadata")
RESULTS_DIR = os.path.abspath("results")

def get_installed_packages():
    import importlib.metadata
    packages = {}
    for dist in sorted(importlib.metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
        packages[dist.metadata['Name']] = dist.version
    return packages

def generate_reproducibility_manifest():
    manifest = {
        "project": "Machine Learning Framework for Stock Price Forecasting and Similar-Stock Recommendation Using Historical OHLCV Data",
        "protocol_version": "Phase 4 Execution Spec v1.0",
        "timestamp": datetime.now().isoformat(),
        "environment": {
            "os": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.machine(),
            "cpu_count": os.cpu_count()
        },
        "data_provenance": {
            "dataset_id": "AmirTrader/YahooFinance",
            "pinned_commit": "c3c01ff2fc62e02c338d2e03bdfd71016da09701",
            "expected_raw_parquet_count": 6708,
            "synchronized_period_start": "2019-09-26",
            "synchronized_period_end": "2026-09-25",
            "calendar_rows": 1759,
            "universe_name": "Universe B — Liquid Core",
            "target_universe_count": 2435
        },
        "target_specifications": {
            "primary_horizon": 5,
            "primary_target": "Daily Cross-Sectional Z-Score of 5-day forward return",
            "secondary_targets": ["Raw forward return", "Cross-sectional excess return"],
            "robustness_horizons": [1, 5, 21]
        },
        "feature_specification": {
            "num_features": 30,
            "groups": ["G1_MOMENTUM", "G2_VOLATILITY", "G3_TREND", "G4_VOLUME", "G5_BAR_GEOMETRY"],
            "lookback_windows": [1, 5, 10, 14, 20, 21, 50, 63, 200]
        },
        "temporal_partitions": {
            "train_end": "2024-03-28",
            "purge_gap_1": 5,
            "validation_end": "2025-06-27",
            "purge_gap_2": 5,
            "test_end": "2026-09-25",
            "random_split": False
        },
        "similarity_specifications": {
            "lookback_windows": [252, 504],
            "metrics": ["Pearson Correlation", "Factor Cosine", "Trajectory Proximity"]
        },
        "recommendation_specifications": {
            "methods": ["Method A: Prediction Only", "Method B: Similarity Only", "Method C: Combined 50/50 Rank Fusion"],
            "k_peer_stocks": 5,
            "benchmark": "Equal-weighted eligible cross-sectional universe return"
        },
        "installed_dependencies": get_installed_packages()
    }
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    manifest_path = os.path.join(METADATA_DIR, "reproducibility_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Reproducibility manifest generated: {manifest_path}")
    return manifest

if __name__ == "__main__":
    generate_reproducibility_manifest()
