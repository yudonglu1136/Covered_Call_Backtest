# File: Fetch/fetch_ohlcv_polygon.py
# Purpose: Fetch 10 years of daily OHLCV (+VWAP) for QQQ, VOO, TQQQ from Polygon,
#          multi-threaded, and save to ./data folder as CSV.

import os
import time
import json
import datetime as dt
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from dotenv import load_dotenv
load_dotenv()  # will read .env into process env

API_KEY = os.getenv("POLYGON_API_KEY") 
TICKERS = ["QQQ", "TQQQ"]
DAYS_BACK = 365 * 10 + 3  # ~10 years
MAX_WORKERS = 3


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

BASE_URL = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{date_from}/{date_to}"
SESSION = requests.Session()
HEADERS = {"User-Agent": "ohlcv-downloader/1.0 (contact: example@example.com)"}

def _iso_date(d: dt.date) -> str:
    return d.strftime("%Y-%m-%d")

def _request_with_retries(url: str, params: dict, max_retries: int = 6, timeout: int = 30) -> requests.Response:
    """GET with retries for rate limits (429) and server errors (503)."""
    for i in range(max_retries):
        r = SESSION.get(url, params=params, headers=HEADERS, timeout=timeout)
        if r.status_code == 200:
            return r
        if r.status_code in (429, 503):
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else 1.5 * (i + 1)
            print(f"[warn] {r.status_code} for {url}, retrying in {sleep_s:.1f}s ...")
            time.sleep(sleep_s)
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed after retries: {url}  params={params}")

def fetch_ohlcv_daily(ticker: str, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Fetch daily OHLCV + VWAP for given date range."""
    url = BASE_URL.format(ticker=ticker, date_from=_iso_date(start_date), date_to=_iso_date(end_date))
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": API_KEY,
    }
    r = _request_with_retries(url, params=params)
    j = r.json()
    results = j.get("results") or []
    if not results:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "vwap", "dollar_value"])
    rows = []
    for x in results:
        d = dt.date.fromtimestamp(x["t"] / 1000.0)
        rows.append({
            "date": d.isoformat(),
            "open": x.get("o"),
            "high": x.get("h"),
            "low":  x.get("l"),
            "close": x.get("c"),
            "volume": x.get("v"),
            "vwap": x.get("vw"),
            "transactions": x.get("n"),
            "dollar_value": (x.get("v") or 0) * (x.get("c") or 0)
        })
    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

def save_csv_for_ticker(ticker: str, df: pd.DataFrame):
    out_path = os.path.join(DATA_DIR, f"{ticker}_ohlcv_1d.csv")
    df.to_csv(out_path, index=False)
    print(f"[saved] {ticker}: {len(df)} rows -> {out_path}")

def task(ticker: str, start_date: dt.date, end_date: dt.date):
    try:
        df = fetch_ohlcv_daily(ticker, start_date, end_date)
        save_csv_for_ticker(ticker, df)
        return (ticker, True)
    except Exception as e:
        return (ticker, False, str(e))

if __name__ == "__main__":
    end = dt.date.today()
    start = end - dt.timedelta(days=DAYS_BACK)
    print(f"[info] fetching daily OHLCV for {TICKERS} from {start} to {end} ...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(task, ticker, start, end) for ticker in TICKERS]
        for f in futures:
            result = f.result()
            print("[done]", result)
