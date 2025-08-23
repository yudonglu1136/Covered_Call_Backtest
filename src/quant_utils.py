# src/quant_utils.py
from __future__ import annotations
import math
import numpy as np
import pandas as pd

# -------- 日期/时间 --------
def parse_date_or_none(x):
    if x is None:
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return pd.to_datetime(x).normalize()

def normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

# -------- 业务小工具 --------
def open_hedge_contracts(active_hedges) -> int:
    return int(sum(int(h.get("contracts", 1)) for h in active_hedges if not h.get("settled", False)))

def iv_to_delta(iv: float, steepness: float = 10, mid: float = 0.27,
                min_delta: float = 0.018, max_delta: float = 0.15) -> float:
    norm = 1.0 / (1.0 + math.exp(-steepness * (mid - float(iv))))
    delta_val = min_delta + (max_delta - min_delta) * norm
    return round(float(delta_val), 4)

def price_on_or_before(idx_price: pd.DataFrame, ts: pd.Timestamp, current_fallback: float) -> float:
    if ts in idx_price.index:
        return float(idx_price.loc[ts, "Open"])
    earlier = idx_price.index[idx_price.index <= ts]
    if len(earlier) > 0:
        # 修复原来的拼写错误: earier -> earlier
        return float(idx_price.loc[earlier[-1], "Open"])
    return float(current_fallback)

def shares_affordable_for_put(cash_free: float, strike: float) -> int:
    if strike <= 0:
        return 0
    return int(cash_free // (strike * 100))

def shares_affordable(cash: float, price: float) -> int:
    lots = int(cash // price) // 100
    return int(lots * 100)

def sharpe_ratio(equity_curve: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return 0.0
    rf_per_period = rf_annual / periods_per_year
    excess = rets - rf_per_period
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))

def prep_price_df(raw: pd.DataFrame) -> pd.DataFrame:
    if "date" not in raw.columns:
        raise KeyError(f"'date' column not found: {list(raw.columns)}")
    if "Open" not in raw.columns:
        if "open" in raw.columns:
            raw = raw.rename(columns={"open": "Open"})
        else:
            for alt in ["Close","close","Adj Close","adj_close","adjClose"]:
                if alt in raw.columns:
                    raw = raw.rename(columns={alt: "Open"})
                    break
    if "Open" not in raw.columns:
        raise KeyError(f"No 'Open' or 'open'-like column found: {list(raw.columns)}")
    raw["date"] = normalize_date_series(raw["date"])
    raw["Open"] = pd.to_numeric(raw["Open"], errors="coerce")
    return raw[["date","Open"]].dropna()

def compute_mdd(curve: pd.Series):
    if curve.empty:
        return 0.0, None, None, None
    roll_max = curve.cummax()
    dd = curve / roll_max - 1.0
    trough_date = dd.idxmin()
    mdd = float(dd.loc[trough_date])
    peak_date = curve.loc[:trough_date].idxmax()
    prior_peak_val = float(curve.loc[peak_date])
    rec = curve.loc[trough_date:]
    rec_idx = rec[rec >= prior_peak_val]
    recovery_date = rec_idx.index[0] if not rec_idx.empty else None
    return mdd, peak_date, trough_date, recovery_date

def fmt_currency(x):
    return f"${x:,.2f}"

def _pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns: return c
    return None

def load_fear_greed_csv(path: Path) -> pd.Series:
    """
    Accepts files like Fear_and_greed.csv with arbitrary date & value column names.
    Returns a business-day-forward-filled Series named 'FGI' indexed by date.
    """
    df = pd.read_csv(path)
    # normalize columns to lower for search
    lower_map = {c: c.lower().strip() for c in df.columns}
    rev_map = {v: k for k, v in lower_map.items()}
    df = df.rename(columns=lower_map)

    date_col = _pick_col(df, ["date", "timestamp", "time"])
    if date_col is None:
        raise KeyError("Fear_and_greed.csv must contain a date-like column (date/timestamp/time).")

    val_col = _pick_col(df, ["value", "fgi", "index", "feargreed", "fg_value"])
    if val_col is None:
        # fallback: last column
        val_col = df.columns[-1]

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, val_col]].rename(columns={date_col: "date", val_col: "FGI"})
    df["FGI"] = pd.to_numeric(df["FGI"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df = df.set_index("date").asfreq("B").ffill()  # business-day freq, forward fill
    return df["FGI"]

def load_tqqq_open_csv(path: Path) -> pd.Series:
    """
    Accepts TQQQ_ohlcv_1d.csv with flexible column names.
    Returns a Series 'Open' (float) indexed by date (normalized).
    """
    df = pd.read_csv(path)
    # lower-case normalize and keep original for renaming
    lower_map = {c: c.lower().strip() for c in df.columns}
    df.columns = [lower_map[c] for c in df.columns]

    date_col = _pick_col(df, ["date", "timestamp", "time"])
    if date_col is None:
        # some files store date as index
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
            date_col = "date"
        else:
            raise KeyError("TQQQ_ohlcv_1d.csv must contain a date-like column (date/timestamp/time).")

    open_col = _pick_col(df, ["open", "adj close", "adj_close", "open_price"])
    if open_col is None:
        # fallback to 'close' if open missing
        open_col = _pick_col(df, ["close", "price"])
        if open_col is None:
            raise KeyError("TQQQ_ohlcv_1d.csv must contain an 'Open' (or 'Adj Close/Close')-like column.")

    df[date_col] = pd.to_datetime(df[date_col])
    s = df[[date_col, open_col]].rename(columns={date_col: "date", open_col: "Open"}).dropna()
    s["Open"] = pd.to_numeric(s["Open"], errors="coerce")
    s = s.dropna(subset=["Open"]).sort_values("date").set_index("date")
    # no resample here; we will align by date in main loop
    return s["Open"]

def deploy_cash_into_shares(cash: float, shares: int, price: float):
    lots_shares = shares_affordable(cash, price)
    if lots_shares > 0:
        cost = lots_shares * price
        return cash - cost, shares + lots_shares, lots_shares
    return cash, shares, 0
