"""
Temporal Split Module.
Implements strictly chronological partitioning with explicit purge gaps.
Derives exact trading dates from the synchronized trading calendar.
Zero random splitting. Zero look-ahead leakage.
"""

import json
import os
import pandas as pd
import numpy as np

METADATA_DIR = os.path.abspath("metadata")

def build_temporal_splits(calendar_dates: list, H=5):
    """
    Given a sorted list of unique synchronized trading dates (1,759 dates),
    determines the exact partition indices and date boundaries.
    
    Target boundaries:
    Train end: ~2024-03-28
    Validation end: ~2025-06-27
    Test end: 2026-09-25
    Purge gap: H trading days
    """
    dates = pd.Series(pd.to_datetime(calendar_dates)).sort_values().drop_duplicates().reset_index(drop=True)
    
    # Locate exact target date anchors
    # 1. Train period: starts at dates[0], ends at 2024-03-28 (or closest <= 2024-03-28)
    train_end_target = pd.Timestamp("2024-03-28")
    train_mask = dates <= train_end_target
    train_end_idx = train_mask[train_mask].index[-1]
    train_dates = dates.iloc[0:train_end_idx + 1]
    
    # 2. First Purge Gap: H trading days after train_end_idx
    # Because target at t is forward H days, records at train_end_idx predict up to train_end_idx + H
    purge_1_start_idx = train_end_idx + 1
    purge_1_end_idx = train_end_idx + H
    purge_1_dates = dates.iloc[purge_1_start_idx:purge_1_end_idx + 1]
    
    # 3. Validation period: starts immediately after purge 1
    val_start_idx = purge_1_end_idx + 1
    val_end_target = pd.Timestamp("2025-06-27")
    val_mask = dates <= val_end_target
    val_end_idx = val_mask[val_mask].index[-1]
    val_dates = dates.iloc[val_start_idx:val_end_idx + 1]
    
    # 4. Second Purge Gap: H trading days after val_end_idx
    purge_2_start_idx = val_end_idx + 1
    purge_2_end_idx = val_end_idx + H
    purge_2_dates = dates.iloc[purge_2_start_idx:purge_2_end_idx + 1]
    
    # 5. Out-of-time Test period: starts after purge 2 through the final synchronized date (2026-09-25)
    test_start_idx = purge_2_end_idx + 1
    # Test targets predict forward H days, so the last H days in test cannot have realized H-day forward targets
    test_end_idx = len(dates) - 1
    test_dates = dates.iloc[test_start_idx:test_end_idx + 1]
    
    splits_spec = {
        "H": H,
        "total_calendar_days": len(dates),
        "overall_start": str(dates.iloc[0].date()),
        "overall_end": str(dates.iloc[-1].date()),
        "train": {
            "start_date": str(train_dates.iloc[0].date()),
            "end_date": str(train_dates.iloc[-1].date()),
            "num_days": len(train_dates),
            "start_idx": int(0),
            "end_idx": int(train_end_idx)
        },
        "purge_1": {
            "start_date": str(purge_1_dates.iloc[0].date()),
            "end_date": str(purge_1_dates.iloc[-1].date()),
            "num_days": len(purge_1_dates),
            "start_idx": int(purge_1_start_idx),
            "end_idx": int(purge_1_end_idx)
        },
        "validation": {
            "start_date": str(val_dates.iloc[0].date()),
            "end_date": str(val_dates.iloc[-1].date()),
            "num_days": len(val_dates),
            "start_idx": int(val_start_idx),
            "end_idx": int(val_end_idx)
        },
        "purge_2": {
            "start_date": str(purge_2_dates.iloc[0].date()),
            "end_date": str(purge_2_dates.iloc[-1].date()),
            "num_days": len(purge_2_dates),
            "start_idx": int(purge_2_start_idx),
            "end_idx": int(purge_2_end_idx)
        },
        "test": {
            "start_date": str(test_dates.iloc[0].date()),
            "end_date": str(test_dates.iloc[-1].date()),
            "num_days": len(test_dates),
            "start_idx": int(test_start_idx),
            "end_idx": int(test_end_idx)
        }
    }
    
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(os.path.join(METADATA_DIR, f"temporal_splits_h{H}.json"), "w") as f:
        json.dump(splits_spec, f, indent=2)
        
    return splits_spec, {
        "train_dates": [str(d.date()) for d in train_dates],
        "purge_1_dates": [str(d.date()) for d in purge_1_dates],
        "val_dates": [str(d.date()) for d in val_dates],
        "purge_2_dates": [str(d.date()) for d in purge_2_dates],
        "test_dates": [str(d.date()) for d in test_dates]
    }
