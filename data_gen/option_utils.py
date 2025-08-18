# File: option_utils.py
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import brentq
import requests
from datetime import timedelta
import time
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()  # will read .env into process env

API_KEY = os.getenv("POLYGON_API_KEY")          # was hardcoded
BASE_URL = "https://api.polygon.io"
FRED_API_KEY = os.getenv('FRED_API_KEY')        # was hardcoded

if not API_KEY:
    raise RuntimeError("POLYGON_API_KEY is not set. Put it in .env or export it in your shell.")
if not FRED_API_KEY:
    raise RuntimeError("FRED_API_KEY is not set. Put it in .env or export it in your shell.")

# === Underlying average price (via yfinance) ===
def get_avg_price_yf(ticker, date):
    """
    Fetch the underlying's daily average price (mean of Open and Close)
    for the specified date using yfinance.
    """
    try:
        date_obj = pd.to_datetime(date)
        next_day = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
        df = yf.download(ticker, start=date, end=next_day, interval="1d", progress=False)
        if not df.empty:
            open_price = df["Open"].iloc[0]
            close_price = df["Close"].iloc[0]
            avg_price = (open_price + close_price) / 2
            return float(avg_price)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {ticker} price: {e}")
    return None


# === Risk-free rate (from FRED) ===

import requests

# Cache the last valid risk-free rate to backfill missing dates
latest_valid_rate = None

def get_risk_free_rate(date: str, term: str = "DGS1"):
    """
    Retrieve the risk-free rate for the given date from FRED.
    Falls back to the most recent valid rate if the specific date is unavailable.
    If no prior value exists, uses a default of 4.07%.
    """
    global latest_valid_rate
    url = 'https://api.stlouisfed.org/fred/series/observations'
    params = {
        'series_id': term,
        'api_key': FRED_API_KEY,
        'file_type': 'json',
        'observation_start': date,
        'observation_end': date,
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        rate_str = data['observations'][0]['value']
        rate = float(rate_str) / 100
        latest_valid_rate = rate
        return rate
    except Exception as e:
        if latest_valid_rate is not None:
            print(f"[WARN] No FRED rate on {date}; using last valid rate {latest_valid_rate:.4%}")
            return latest_valid_rate
        else:
            print(f"[WARN] No FRED rate on {date} and no prior value; using default 4.07%")
            latest_valid_rate = 0.0407
            return latest_valid_rate

# === Implied Volatility (IV) and Delta calculation (Black–Scholes) ===
def calculate_with_iv_delta(df):
    """
    For each option row in df, compute implied volatility and call delta using Black–Scholes.
    Assumes df contains: 'underlying_price', 'strike', 'expiration', 'date', 'risk_free_rate', and 'vw' (price).
    """
    df = df.copy()
    ivs, deltas = [], []

    for _, row in df.iterrows():
        S = row['underlying_price']
        K = row['strike']
        T = (row['expiration'] - row['date']).days / 365
        r = row['risk_free_rate']
        price = row['vw']

        if pd.isna(S) or pd.isna(price) or pd.isna(K) or T <= 0 or price <= 0:
            ivs.append(np.nan)
            deltas.append(np.nan)
            continue

        def bs_call_price(S, K, T, r, sigma):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        def implied_volatility():
            try:
                return brentq(lambda sigma: bs_call_price(S, K, T, r, sigma) - price, 1e-6, 5.0)
            except:
                return np.nan

        def bs_delta(sigma):
            if np.isnan(sigma):
                return np.nan
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            return norm.cdf(d1)

        iv = implied_volatility()
        delta = bs_delta(iv)

        ivs.append(iv)
        deltas.append(delta)

    df['iv'] = ivs
    df['delta'] = deltas
    return df

def get_monthly_option_contracts(ticker, date):
    """
    Query Polygon reference endpoint for option contracts as of the given date.
    Paginates through results using 'next_url' if present.
    """
    url = f"{BASE_URL}/v3/reference/options/contracts"
    date_obj = pd.to_datetime(date)
    date_str = date_obj.strftime('%Y-%m-%d')
    
    one_month_later = (date_obj + timedelta(days=32)).strftime('%Y-%m-%d')
    params = {
        "underlying_ticker": ticker,
        'as_of' : date,
        'expired' : False,
        "limit": 1000,
        "apiKey": API_KEY
    }
    contracts = []
    page = 1

    while True:
        print(f"[INFO] Fetching page {page}")
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print(f"[ERROR] HTTP {response.status_code}: {response.text}")
            break
        data = response.json()
        page_results = data.get("results", [])
        contracts += page_results
        print(f"[INFO] Page {page}: fetched {len(page_results)} rows; cumulative {len(contracts)}")

        if "next_url" in data:
            # Note: 'next_url' may not include apiKey; append manually if missing.
            url = data["next_url"]
            if "apiKey=" not in url:
                join_char = "&" if "?" in url else "?"
                url = f"{url}{join_char}apiKey={API_KEY}"
            params = {}
            page += 1
        else:
            break
    return contracts

def get_option_market_data(option_ticker, date):
    """
    Fetch daily aggregate bars for a specific option contract and date from Polygon.
    Returns the 'results' array or an empty list.
    """
    url = f"{BASE_URL}/v2/aggs/ticker/{option_ticker}/range/1/day/{date}/{date}"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 5000,
        "apiKey": API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data.get('results', [])

from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_single_day(ticker, date, max_workers=8):
    """
    Multithreaded daily fetch: retrieves option market data for all contracts returned
    by Polygon's contract list for the given date. Each contract is queried for its
    aggregate bar on that date. Results are concatenated into a single DataFrame.
    """
    try:
        contracts = get_monthly_option_contracts(ticker, date)
    except Exception as e:
        print(f"[ERROR] Failed to fetch contract list: {e}")
        return pd.DataFrame()

    all_data = []

    def fetch_option_data(c):
        opt_ticker = c.get("ticker")
        strike = c.get("strike_price")
        opt_type = c.get("contract_type")
        exp_date = c.get("expiration_date")

        if not all([opt_ticker, strike, opt_type, exp_date]):
            return []

        try:
            md_list = get_option_market_data(opt_ticker, date)
            
            rows = []
            for md in md_list:
                if md.get("vw") is None:
                    continue

                row = {
                    "option_ticker": opt_ticker,
                    "symbol": ticker,
                    "date": pd.to_datetime(md['t'], unit='ms'),
                    "expiration": pd.to_datetime(exp_date),
                    "strike": strike,
                    "type": opt_type,
                    "o": md.get("o"),
                    "c": md.get("c"),
                    "h": md.get("h"),
                    "l": md.get("l"),
                    "vw": md.get("vw"),
                    "v": md.get("v"),
                    "t": md.get("t"),
                    "n": md.get("n")
                }
                rows.append(row)
            return rows
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_option_data, c) for c in contracts]

        for future in as_completed(futures):
            all_data.extend(future.result())

    return pd.DataFrame(all_data)

def fetch_and_calculate_iv_delta(ticker, date):
    """
    Orchestrates daily fetch + IV/Delta computation:
    1) Pulls all option contracts' market data for the date.
    2) Fetches underlying average price and risk-free rate.
    3) Computes implied volatility and delta per contract.
    4) Returns a cleaned DataFrame (rows with NaN IV removed).
    """
    df = fetch_single_day(ticker, date)
    if df.empty:
        print(f"[WARN] No valid option data on {date}")
        return df

    # Underlying price (S) and risk-free rate (r)
    avg_price = get_avg_price_yf(ticker, date)
    risk_free_rate = get_risk_free_rate(date)

    df['underlying_price'] = avg_price
    df['risk_free_rate'] = risk_free_rate

    # Compute IV & Delta
    df = calculate_with_iv_delta(df)
    df = df[df['iv'].notna()].reset_index(drop=True)
    return df


