"""
High-Performance Resilient Dataset Downloader.
Downloads all 6,708 raw daily parquet files from Hugging Face:
Dataset: AmirTrader/YahooFinance
Commit: c3c01ff2fc62e02c338d2e03bdfd71016da09701
Direct CDN streaming via aiohttp (bypassing HF API rate limits)
with concurrency control, auto-resuming, and pyarrow integrity verification.
"""

import os
import sys
import json
import time
import asyncio
import aiohttp
import pandas as pd
import pyarrow.parquet as pq
from datetime import datetime

DATASET_ID = "AmirTrader/YahooFinance"
COMMIT_SHA = "c3c01ff2fc62e02c338d2e03bdfd71016da09701"
BASE_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{COMMIT_SHA}"

TARGET_DAILY_DIR = os.path.abspath("data/raw/data/daily")
TARGET_STATE_DIR = os.path.abspath("data/raw/data/state")
METADATA_DIR = os.path.abspath("metadata")
LOGS_DIR = os.path.abspath("logs")

os.makedirs(TARGET_DAILY_DIR, exist_ok=True)
os.makedirs(TARGET_STATE_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

def is_valid_parquet(file_path: str) -> bool:
    """Verifies that the parquet file is complete and not truncated/corrupt."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) < 100:
        return False
    try:
        meta = pq.read_metadata(file_path)
        return meta.num_rows > 0
    except Exception:
        return False

async def download_ticker(session: aiohttp.ClientSession, semaphore: asyncio.Semaphore, ticker: str, max_retries=5):
    target_path = os.path.join(TARGET_DAILY_DIR, f"{ticker}.parquet")
    if is_valid_parquet(target_path):
        return True
        
    url = f"{BASE_URL}/data/daily/{ticker}.parquet"
    temp_path = target_path + ".tmp"
    
    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                        with open(temp_path, "wb") as f:
                            f.write(content)
                        if is_valid_parquet(temp_path):
                            os.replace(temp_path, target_path)
                            return True
                    elif resp.status == 429 or resp.status >= 500:
                        await asyncio.sleep(1.0 + attempt * 1.5)
        except Exception:
            await asyncio.sleep(1.0)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
    return False

async def main_async():
    start_time = time.time()
    state_file = os.path.join(METADATA_DIR, "ingestion_state.parquet")
    if not os.path.exists(state_file):
        print(f"Error: {state_file} not found!")
        sys.exit(1)
        
    state_df = pd.read_parquet(state_file)
    valid_tickers = state_df[state_df['last_date'].notna()]['ticker'].tolist()
    print(f"[{datetime.now().isoformat()}] Found {len(valid_tickers)} target tickers in ingestion_state.")
    
    already_valid = sum(1 for t in valid_tickers if is_valid_parquet(os.path.join(TARGET_DAILY_DIR, f"{t}.parquet")))
    print(f"Already downloaded and valid: {already_valid}/{len(valid_tickers)}")
    
    remaining_tickers = [t for t in valid_tickers if not is_valid_parquet(os.path.join(TARGET_DAILY_DIR, f"{t}.parquet"))]
    print(f"Remaining tickers to download: {len(remaining_tickers)}")
    
    conn = aiohttp.TCPConnector(limit=50, limit_per_host=30)
    semaphore = asyncio.Semaphore(30)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AutonomousStockResearch/1.0"}
    
    async with aiohttp.ClientSession(connector=conn, headers=headers) as session:
        tasks = [download_ticker(session, semaphore, t) for t in remaining_tickers]
        batch_size = 500
        total_remaining = len(tasks)
        completed_since_start = 0
        
        for i in range(0, total_remaining, batch_size):
            batch = tasks[i:i + batch_size]
            results = await asyncio.gather(*batch)
            completed_since_start += len(batch)
            elapsed = time.time() - start_time
            rate = completed_since_start / (elapsed + 1e-8)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch complete. Progress: {already_valid + completed_since_start}/{len(valid_tickers)} ({rate:.1f} files/s)")
            
    # Final full integrity pass
    print(f"\n[{datetime.now().isoformat()}] Running full pyarrow integrity check on all {len(valid_tickers)} files...")
    verified_valid = 0
    missing_or_corrupt = []
    
    for t in valid_tickers:
        p = os.path.join(TARGET_DAILY_DIR, f"{t}.parquet")
        if is_valid_parquet(p):
            verified_valid += 1
        else:
            missing_or_corrupt.append(t)
            
    total_time = time.time() - start_time
    print(f"\n============================================================")
    print(f"DATASET DOWNLOAD AND VERIFICATION REPORT")
    print(f"============================================================")
    print(f"Dataset ID:                {DATASET_ID}")
    print(f"Commit SHA:                {COMMIT_SHA}")
    print(f"Expected Raw Daily Files:  6,708")
    print(f"Verified Valid Files:      {verified_valid}")
    print(f"Missing or Corrupted:      {len(missing_or_corrupt)}")
    print(f"Total Duration:            {total_time:.2f} seconds")
    print(f"Status:                    {'PASS (100% Complete)' if verified_valid == 6708 else 'INCOMPLETE'}")
    print(f"============================================================\n")
    
    manifest = {
        "dataset_id": DATASET_ID,
        "commit_sha": COMMIT_SHA,
        "download_timestamp": datetime.now().isoformat(),
        "verified_raw_count_expected": 6708,
        "verified_raw_count_actual": verified_valid,
        "exact_match": verified_valid == 6708,
        "duration_seconds": total_time,
        "missing_tickers": missing_or_corrupt
    }
    
    with open(os.path.join(METADATA_DIR, "raw_dataset_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    if verified_valid != 6708:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main_async())
