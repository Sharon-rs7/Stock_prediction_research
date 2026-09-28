"""
Universe Construction and Audit Script.
Applies the 5 filtering gates for Universe B (Liquid Core):
Gate 1: instrument_type_heuristic == COMMON_CANDIDATE
Gate 2: Data Quality (negative_price == 0, invalid_ohlc == 0, min_close > 0)
Gate 3: Trading Activity (zero_volume_pct <= 1%)
Synchronization: rows == 1,759 (2019-09-26 to 2026-09-25 strictly balanced calendar)
Liquidity: median_volume >= 100,000 shares

Expected final universe: 2,435 tickers.
"""

import os
import glob
import json
import re
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

RAW_DAILY_DIR = os.path.abspath("data/raw/data/daily")
METADATA_DIR = os.path.abspath("metadata")
LOGS_DIR = os.path.abspath("logs")

def is_common_candidate(ticker: str) -> bool:
    """
    Gate 1: Common stock heuristic.
    Excludes preferred shares, warrants, rights, units, test symbols.
    Common candidates are alphabetic tickers (typically 1-5 letters)
    without hyphens, dots, or warrant/unit/preferred suffixes.
    """
    if not ticker.isalpha():
        return False
    # Check length
    if len(ticker) < 1 or len(ticker) > 5:
        return False
    # Typical 5-letter suffixes for non-common equities
    if len(ticker) == 5:
        # 'W' for warrants, 'R' for rights, 'U' for units, 'P' for preferred
        if ticker.endswith(('W', 'R', 'U', 'P', 'Z')):
            return False
    return True

def audit_single_file(file_path: str):
    """
    Audits a single parquet file against the research gates.
    """
    ticker = os.path.basename(file_path).replace(".parquet", "")
    res = {
        "ticker": ticker,
        "file_path": file_path,
        "is_common": is_common_candidate(ticker),
        "row_count": 0,
        "min_date": None,
        "max_date": None,
        "negative_price_count": 0,
        "invalid_ohlc_count": 0,
        "min_close": 0.0,
        "zero_volume_pct": 1.0,
        "median_volume": 0.0,
        "gate1_pass": False,
        "gate2_pass": False,
        "gate3_pass": False,
        "sync_pass": False,
        "liquidity_pass": False,
        "universe_b_pass": False,
        "error": None
    }
    
    res["gate1_pass"] = res["is_common"]
    
    try:
        df = pd.read_parquet(file_path)
        res["row_count"] = len(df)
        if len(df) == 0:
            return res
            
        res["min_date"] = str(df["date"].min())[:10]
        res["max_date"] = str(df["date"].max())[:10]
        
        # Gate 2 checks
        neg_prices = (
            (df["open"] < 0) | (df["high"] < 0) | (df["low"] < 0) | 
            (df["close"] < 0) | (df["adj_close"] < 0)
        ).sum()
        res["negative_price_count"] = int(neg_prices)
        
        # invalid OHLC: high < low, open < low, close < low, open > high, close > high
        invalid_ohlc = (
            (df["high"] < df["low"]) | (df["open"] < df["low"]) | (df["close"] < df["low"]) |
            (df["open"] > df["high"]) | (df["close"] > df["high"])
        ).sum()
        res["invalid_ohlc_count"] = int(invalid_ohlc)
        
        min_close = float(df["close"].min())
        res["min_close"] = min_close
        
        res["gate2_pass"] = (neg_prices == 0) and (invalid_ohlc == 0) and (min_close > 0)
        
        # Gate 3 checks: zero volume percentage <= 1%
        zero_vol_pct = float((df["volume"] == 0).mean())
        res["zero_volume_pct"] = zero_vol_pct
        res["gate3_pass"] = (zero_vol_pct <= 0.01)
        
        # Synchronization check: rows == 1759, 2019-09-26 to 2026-09-25
        res["sync_pass"] = (len(df) == 1759) and (res["min_date"] == "2019-09-26") and (res["max_date"] == "2026-09-25")
        
        # Liquidity check: median_volume >= 100,000 shares
        med_vol = float(df["volume"].median())
        res["median_volume"] = med_vol
        res["liquidity_pass"] = (med_vol >= 100000.0)
        
        # Final Universe B membership
        res["universe_b_pass"] = (
            res["gate1_pass"] and 
            res["gate2_pass"] and 
            res["gate3_pass"] and 
            res["sync_pass"] and 
            res["liquidity_pass"]
        )
        
    except Exception as e:
        res["error"] = str(e)
        
    return res

def run_universe_audit(max_workers=8):
    files = sorted(glob.glob(os.path.join(RAW_DAILY_DIR, "*.parquet")))
    print(f"[{datetime.now().isoformat()}] Auditing {len(files)} raw daily parquet files...")
    
    results = []
    # Use ProcessPoolExecutor for CPU-bound parquet inspection
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(audit_single_file, f): f for f in files}
        for i, future in enumerate(as_completed(futures)):
            results.append(future.result())
            if (i + 1) % 500 == 0 or (i + 1) == len(files):
                print(f"Audited {i + 1}/{len(files)} files...")
                
    audit_df = pd.DataFrame(results)
    
    # Save full audit records
    audit_path = os.path.join(METADATA_DIR, "universe_b_audit.parquet")
    audit_df.to_parquet(audit_path, index=False)
    audit_df.to_csv(os.path.join(METADATA_DIR, "universe_b_audit.csv"), index=False)
    
    # Filtering Funnel Breakdown
    total_raw = len(audit_df)
    gate1_candidates = audit_df["gate1_pass"].sum()
    gate2_clean = (audit_df["gate1_pass"] & audit_df["gate2_pass"]).sum()
    gate3_active = (audit_df["gate1_pass"] & audit_df["gate2_pass"] & audit_df["gate3_pass"]).sum()
    sync_passed = (audit_df["gate1_pass"] & audit_df["gate2_pass"] & audit_df["gate3_pass"] & audit_df["sync_pass"]).sum()
    universe_b_final = audit_df["universe_b_pass"].sum()
    
    funnel = {
        "total_raw_files": int(total_raw),
        "gate1_common_candidate": int(gate1_candidates),
        "gate2_data_quality": int(gate2_clean),
        "gate3_zero_vol_le_1pct": int(gate3_active),
        "synchronization_1759_rows": int(sync_passed),
        "liquidity_median_vol_ge_100k": int(universe_b_final),
        "universe_b_count": int(universe_b_final),
        "target_expected_count": 2435,
        "is_exact_match": int(universe_b_final) == 2435
    }
    
    funnel_path = os.path.join(METADATA_DIR, "universe_b_funnel.json")
    with open(funnel_path, "w") as f:
        json.dump(funnel, f, indent=2)
        
    # Save final Universe B tickers
    universe_b_tickers = sorted(audit_df[audit_df["universe_b_pass"]]["ticker"].tolist())
    tickers_path = os.path.join(METADATA_DIR, "universe_b_tickers.json")
    with open(tickers_path, "w") as f:
        json.dump(universe_b_tickers, f, indent=2)
        
    print(f"\n============================================================")
    print(f"UNIVERSE B AUDIT COMPLETE")
    print(f"============================================================")
    print(f"Total Raw Files:         {total_raw}")
    print(f"Gate 1 (Common):         {gate1_candidates}")
    print(f"Gate 2 (Clean OHLCV):    {gate2_clean}")
    print(f"Gate 3 (Volume > 0):     {gate3_active}")
    print(f"Sync (1,759 rows):       {sync_passed}")
    print(f"Liquidity (Vol >= 100k): {universe_b_final}")
    print(f"Target Universe B Count: 2435")
    print(f"Exact Match:             {int(universe_b_final) == 2435}")
    print(f"============================================================\n")
    
    return funnel, universe_b_tickers

if __name__ == "__main__":
    run_universe_audit()
