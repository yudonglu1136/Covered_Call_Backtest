import requests
import pandas as pd
import yfinance as yf
from datetime import date
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()  # will read .env into process env

# ---------------- Path handling (robust) ----------------
# Resolve ../data relative to this script's location, regardless of current working directory.
HERE = Path(__file__).resolve().parent          # .../data_gen (typically)
DATA_DIR = (HERE.parent / "data").resolve()     # .../data
DATA_DIR.mkdir(parents=True, exist_ok=True)

file_path = DATA_DIR / "Fear_and_greed.csv"
print(f"[PATH] Using Fear & Greed CSV at: {file_path}")

# Initialize the CSV if it does not exist (index: Date, column: Value).
if not file_path.exists():
    print("[BOOTSTRAP] Fear_and_greed.csv not found. Creating an empty file with columns [Date, Value].")
    init_df = pd.DataFrame(columns=["Value"])
    init_df.index.name = "Date"
    init_df.to_csv(file_path)


api_key = os.getenv("RAPIDAPI_KEY")
if not api_key:
    raise RuntimeError("RAPIDAPI_KEY is not set. Put it in .env or export it in your shell.")

# ---------------- Load existing index values ----------------
index_values = pd.read_csv(file_path, index_col='Date', parse_dates=True)

# Get today's date and a start date
today = date.today()
start_day = '2021-12-15'  # or: str(index_values.tail(1).index[0])[:10]
print("the start_day is : {}".format(start_day))

def get_value_from_dictornary(data):
    key = list(data.keys())[0]
    return data[key]

def fetch_historical_trading_dates(ticker, start_date, end_date):
    # Fetch historical daily bars to infer trading dates
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return data.index  # DatetimeIndex

def get_historical_fear_and_greed_index(api_key, date="2021-06-03"):
    url = 'https://fear-and-greed-index2.p.rapidapi.com/historical'
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": 'fear-and-greed-index2.p.rapidapi.com',
    }
    params = {'date': date}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error fetching data: {response.status_code}"

# Replace with your actual RapidAPI key
api_key = 'e39756d1cfmshc93a6fa99bc4832p1d49e0jsnaa0cf52aaa0f'

# Pull trading dates for QQQ
dates = fetch_historical_trading_dates("QQQ", start_day, today)

# Backfill missing dates in Fear & Greed CSV
for dt in dates:
    if dt not in index_values.index:
        value = get_historical_fear_and_greed_index(api_key, str(dt)[:10])
        if isinstance(value, dict):
            date_value = get_value_from_dictornary(value)
            print(dt, date_value)
            index_values.loc[dt] = date_value

# Persist sorted result
index_values = index_values.sort_index()
index_values.to_csv(file_path)
print("[DONE] Fear & Greed index updated.")
