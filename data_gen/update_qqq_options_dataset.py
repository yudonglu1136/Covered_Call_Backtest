import os
import time
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime, timedelta, date
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.option_utils import (
    fetch_and_calculate_iv_delta
)
TICKER = "QQQ"

# === Robust pathing: resolve data/ based on this script's location ===
HERE = Path(__file__).resolve().parent          # .../data_gen
DATA_DIR = (HERE.parent / "data").resolve()     # .../data
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = (DATA_DIR / "options_with_iv_delta.csv").resolve()

print(f"[PATH] Dataset path: {OUTPUT_PATH}")

# === Step 1: Load existing dataset if present ===
file_exists = OUTPUT_PATH.exists()
try:
    if file_exists:
        df_all = pd.read_csv(OUTPUT_PATH, parse_dates=["date", "expiration"])
        existing_dates = set(df_all["date"].dt.date)
        print(f"[INFO] Existing rows: {len(df_all)} | Unique dates: {len(existing_dates)}")
    else:
        raise FileNotFoundError
except FileNotFoundError:
    df_all = pd.DataFrame()
    existing_dates = set()
    print("[WARN] No historical file found. Will bootstrap from scratch.")

# === Step 2: Build the last 4 years of NYSE trading days ===
today = datetime.today().date()
# today = date(2025, 7, 14)  # uncomment to pin a reference date
start_date = today - timedelta(days=365 * 4)

nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(start_date=start_date, end_date=today)
trading_days = [d.date() for d in schedule.index]

# === Bootstrap: if the file does not exist, create it with the earliest available trading day ===
if not file_exists:
    print("[BOOTSTRAP] Initializing dataset (first-time CSV creation)...")
    bootstrap_success = False
    for boot_d in trading_days:  # iterate from the earliest trading day forward
        try:
            print(f"  -> Trying bootstrap date: {boot_d}")
            boot_df = fetch_and_calculate_iv_delta(TICKER, boot_d)
            if boot_df is not None and not boot_df.empty:
                boot_df.to_csv(OUTPUT_PATH, index=False)
                df_all = boot_df.copy()
                existing_dates = {boot_d}
                file_exists = True
                bootstrap_success = True
                print(f"[OK] Bootstrap success: wrote {len(boot_df)} rows -> {OUTPUT_PATH}")
                break
            else:
                print(f"  [SKIP] No valid data for {boot_d}; trying the next trading day...")
        except Exception as e:
            print(f"  [ERROR] Bootstrap failed for {boot_d}: {e}")
        time.sleep(1.0)

    if not bootstrap_success:
        raise RuntimeError(
            "Bootstrap failed: unable to obtain valid options data for multiple consecutive trading days. "
            "Please check data source and network connectivity."
        )

# === Step 3: Determine missing dates (process in reverse chronological order) ===
missing_dates = sorted([d for d in trading_days if d not in existing_dates], reverse=True)
print(f"[INFO] Pending trading days to backfill: {len(missing_dates)}")

# === Step 4: Fetch and append day-by-day ===
for i, d in enumerate(missing_dates):
    print(f"\n[RUN] {i+1}/{len(missing_dates)} | Date: {d}")
    try:
        new_df = fetch_and_calculate_iv_delta(TICKER, d)
        if new_df is not None and not new_df.empty:
            df_all = pd.concat([df_all, new_df], ignore_index=True)
            df_all.to_csv(OUTPUT_PATH, index=False)
            print(f"[OK] Saved {len(df_all)} cumulative rows -> {OUTPUT_PATH}")
        else:
            print(f"[SKIP] No valid data for {d}")
    except Exception as e:
        print(f"[FAIL] Fetch failed for {d}: {e}")

    time.sleep(1.5)

print("\n[DONE] Backfill completed successfully.")


