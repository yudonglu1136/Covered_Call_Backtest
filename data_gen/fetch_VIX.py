# -*- coding: utf-8 -*-
import yfinance as yf
import pandas as pd
from pandas_datareader import data as pdr
import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


start = "2015-01-01"

end   = datetime.date.today().strftime("%Y-%m-%d")

# ---------- VIX ----------
vix = yf.download("^VIX", start=start, end=end)
vix = vix[['Close']].rename(columns={'Close': 'close'})
vix.reset_index(inplace=True)
vix.to_csv(DATA / "VIX.csv", index=False)
print(f"[OK] VIX -> data/VIX.csv  shape={vix.shape}  range=({vix['Date'].min()} → {vix['Date'].max()})")

