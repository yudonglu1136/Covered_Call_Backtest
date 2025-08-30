# -*- coding: utf-8 -*-
"""
Build multiple market datasets (single CLI):
- Yield curve: FRED DGS10 & DGS2 -> slope -> data/yield_curve.csv
- IG/HY: FRED yields & OAS -> data/ig_hy_fred.csv (+ plots)
- VIX: yfinance ^VIX -> data/VIX.csv
- US10Y: FRED DGS10 -> data/US10Y.csv
- DXY: FRED DTWEXBGS -> data/dxy.csv

Usage (from project root):
  python data_gen/build_all_market_data.py
  python data_gen/build_all_market_data.py --what ig_hy,vix --start 2015-01-01 --end 2025-08-20
  python data_gen/build_all_market_data.py --years 10
"""

from __future__ import annotations
import os
import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

# third-party deps
from dotenv import load_dotenv
from fredapi import Fred
import matplotlib.pyplot as plt
import yfinance as yf
from pandas_datareader import data as pdr


# -------------------- Paths & env --------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]            # project root
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DATA.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# load .env from root (fallback: CWD)
ENV_PATH = ROOT / ".env"
if not ENV_PATH.exists():
    alt = Path.cwd() / ".env"
    if alt.exists():
        ENV_PATH = alt
load_dotenv(dotenv_path=ENV_PATH)

def make_fred() -> Fred:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "FRED_API_KEY not found. 请在项目根目录 .env 中设置：\n"
            "FRED_API_KEY=xxxxxxxxxxxxxxxx"
        )
    return Fred(api_key=api_key)


# -------------------- Helpers --------------------
def clamp_by_years(df: pd.DataFrame, years: int) -> pd.DataFrame:
    """Keep last N years if explicit start not provided."""
    if df.empty:
        return df
    last = pd.to_datetime(df.index.max())
    start = last - pd.DateOffset(years=years)
    return df[df.index >= start]

def clip_to_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start:
        df = df.loc[pd.Timestamp(start):]
    if end:
        df = df.loc[:pd.Timestamp(end)]
    return df


# -------------------- 1) Yield curve (DGS10, DGS2) --------------------
def build_yield_curve(start: Optional[str], end: Optional[str], years: int) -> pd.DataFrame:
    fred = make_fred()
    dgs10 = fred.get_series("DGS10", observation_start=start, observation_end=end)
    dgs2  = fred.get_series("DGS2",  observation_start=start, observation_end=end)
    df = pd.concat([dgs10.rename("DGS10"), dgs2.rename("DGS2")], axis=1)
    df.index.name = "date"
    df = df.sort_index()
    df["DGS10"] = pd.to_numeric(df["DGS10"], errors="coerce")
    df["DGS2"]  = pd.to_numeric(df["DGS2"],  errors="coerce")
    df["slope"] = df["DGS10"] - df["DGS2"]
    df = df.dropna(subset=["DGS10","DGS2","slope"])
    if not start:  # only clamp when no explicit start
        df = clamp_by_years(df, years)
    out = DATA / "yield_curve.csv"
    df.to_csv(out)
    print(f"[OK] yield_curve -> {out}  shape={df.shape}")
    return df


# -------------------- 2) IG/HY yields & OAS --------------------
IG_YIELD_CODE = "BAMLC0A0CMEY"     # IG Effective Yield
HY_YIELD_CODE = "BAMLH0A0HYM2EY"   # HY Effective Yield
IG_OAS_CODE   = "BAMLC0A0CM"       # IG OAS (bps)
HY_OAS_CODE   = "BAMLH0A0HYM2"     # HY OAS (bps)

def fetch_ig_hy(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    fred = make_fred()
    series_map = {
        "IG Yield":  IG_YIELD_CODE,
        "HY Yield":  HY_YIELD_CODE,
        "IG Spread": IG_OAS_CODE,
        "HY Spread": HY_OAS_CODE,
    }
    frames = []
    for name, code in series_map.items():
        print(f"[fetch] {name} ({code}) start={start} end={end}")
        s = fred.get_series(code, observation_start=start, observation_end=end)
        s = s.rename(name)
        frames.append(s)
    df = pd.concat(frames, axis=1).sort_index()
    if {"HY Spread", "IG Spread"}.issubset(df.columns):
        df["HY-IG Spread (bps)"] = df["HY Spread"] - df["IG Spread"]

    out_csv = DATA / "ig_hy_fred.csv"
    df.to_csv(out_csv)
    print(f"[OK] IG/HY -> {out_csv}  shape={df.shape}")

    # plots
    plt.figure(figsize=(10,6))
    df[["IG Yield","HY Yield"]].plot(figsize=(10,6), title="US IG vs HY Yield (ICE BofA via FRED)")
    plt.ylabel("Yield (%)"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(REPORTS / "ig_hy_yield.png", dpi=150); plt.close()
    print(f"[OK] Chart -> {REPORTS / 'ig_hy_yield.png'}")

    plt.figure(figsize=(10,6))
    df[["IG Spread","HY Spread"]].plot(figsize=(10,6), title="US IG vs HY OAS (bps)")
    plt.ylabel("Spread vs Treasury (bps)"); plt.xlabel("")
    plt.tight_layout(); plt.savefig(REPORTS / "ig_hy_spread.png", dpi=150); plt.close()
    print(f"[OK] Chart -> {REPORTS / 'ig_hy_spread.png'}")

    if "HY-IG Spread (bps)" in df.columns:
        plt.figure(figsize=(10,5))
        df[["HY-IG Spread (bps)"]].plot(title="HY-IG Spread Differential (bps)")
        plt.ylabel("bps"); plt.xlabel("")
        plt.tight_layout(); plt.savefig(REPORTS / "hy_minus_ig_spread.png", dpi=150); plt.close()
        print(f"[OK] Chart -> {REPORTS / 'hy_minus_ig_spread.png'}")

    return df

# -------------------- 4) US10Y via FRED (DGS10) --------------------
def fetch_us10y(start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    fred = make_fred()
    s = fred.get_series("DGS10", observation_start=start, observation_end=end)
    if s is None or len(s) == 0:
        # fallback: FRED via pandas_datareader
        df = pdr.DataReader("DGS10", "fred", start, end)
        df = df.rename(columns={"DGS10": "close"})
        df = df.rename_axis("date").reset_index()
    else:
        df = s.to_frame(name="close").rename_axis("date").reset_index()
    out = DATA / "US10Y.csv"
    df.to_csv(out, index=False)
    print(f"[OK] US10Y -> {out}  shape={df.shape}")
    return df



def _to_two_cols(df: pd.DataFrame, ticker_hint: str) -> pd.DataFrame:
    """
    统一清洗：取 Adj Close（无则 Close），得到 (date,dxy) 两列，工作日频率并前向填充。
    """
    if df is None or df.empty:
        raise ValueError("empty dataframe")

    # 处理多层列名（少数情形 yfinance 会返回 MultiIndex）
    if isinstance(df.columns, pd.MultiIndex):
        # 优先 'Adj Close' 这一层
        col_layer0 = [lev for lev in df.columns.levels[0]]
        price = None
        if "Adj Close" in col_layer0:
            sub = df["Adj Close"]
            # 如果还有第二层 ticker，则优先匹配
            if isinstance(sub, pd.DataFrame):
                if ticker_hint in sub.columns:
                    price = sub[ticker_hint]
                else:
                    price = sub.iloc[:, 0]
            else:
                price = sub
        else:
            # 退化取 'Close'
            sub = df["Close"] if "Close" in col_layer0 else df.xs(df.columns[0], axis=1, level=0)
            price = sub[ticker_hint] if (isinstance(sub, pd.DataFrame) and ticker_hint in sub.columns) else sub.iloc[:, 0]
        s = price.rename("dxy")
    else:
        # 常见情形：单层列
        col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[-1])
        s = df[col].rename("dxy")

    out = s.to_frame()
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    out = out.sort_index().asfreq("B").ffill()   # 对齐到工作日并前向填充
    return out[["dxy"]]

def fetch_dxy(*_args, **_kwargs) -> pd.DataFrame:
    """
    Yahoo-only DXY, last 10y, 1d bars -> data/dxy.csv
    Clean to two columns: date,dxy
    Ignores any passed arguments for backward compatibility.
    """
    from pathlib import Path
    import pandas as pd
    import yfinance as yf

    TICKERS = ["^DXY", "DX-Y.NYB", "DX=F"]
    PERIOD   = "10y"
    INTERVAL = "1d"
    DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CSV  = DATA_DIR / "dxy.csv"

    def _to_two_cols(df: pd.DataFrame, tk: str) -> pd.DataFrame:
        if df is None or df.empty:
            raise ValueError("empty dataframe")
        if isinstance(df.columns, pd.MultiIndex):
            # 优先 Adj Close，其次 Close
            if "Adj Close" in df.columns.levels[0]:
                sub = df["Adj Close"]
            elif "Close" in df.columns.levels[0]:
                sub = df["Close"]
            else:
                sub = df.xs(df.columns.levels[0][0], axis=1, level=0)
            ser = sub[tk] if isinstance(sub, pd.DataFrame) and tk in sub.columns else sub.iloc[:, 0]
            s = ser.rename("dxy")
        else:
            col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else df.columns[-1])
            s = df[col].rename("dxy")
        out = s.to_frame()
        out.index = pd.to_datetime(out.index); out.index.name = "date"
        out = out.sort_index().asfreq("B").ffill()
        return out[["dxy"]]

    last_ok = None
    for tk in TICKERS:
        try:
            df = yf.download(tk, period=PERIOD, interval=INTERVAL,
                             auto_adjust=False, progress=False,
                             group_by="column", threads=False)
            if df is not None and not df.empty:
                clean = _to_two_cols(df, tk)
                last_ok = (tk, clean)
                if tk == "^DXY":  # 命中指数就直接用
                    break
        except Exception as e:
            print(f"[WARN] Yahoo {tk} 失败：{e}；尝试下一个…")

    if last_ok is None:
        raise RuntimeError("无法从 Yahoo 获取 DXY（^DXY / DX-Y.NYB / DX=F 都失败）")

    tk, out = last_ok
    out.to_csv(OUT_CSV)
    print(f"[OK] DXY (Yahoo {tk}) -> {OUT_CSV}  shape={out.shape}  last={out.index.max().date()}")
    return out

# -------------------- CLI --------------------
CHOICES = ["yield_curve", "ig_hy", "vix", "us10y", "dxy"]

def parse_args():
    # 默认 end=今天
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")
    ap = argparse.ArgumentParser(description="Build multiple market datasets (FRED, yfinance).")
    ap.add_argument("--what", type=str, default="all",
                    help="comma-separated in {yield_curve,ig_hy,vix,us10y,dxy} or 'all'")
    ap.add_argument("--start", type=str, default=None, help="YYYY-MM-DD (if set, overrides --years)")
    ap.add_argument("--end", type=str, default=today, help=f"YYYY-MM-DD (default: today={today})")
    ap.add_argument("--years", type=int, default=10, help="Keep last N years when --start not provided")
    return ap.parse_args()

def main():
    args = parse_args()
    sel = [w.strip().lower() for w in (args.what.split(",") if args.what != "all" else CHOICES)]
    sel = [w for w in sel if w in CHOICES]
    if not sel:
        sel = CHOICES

    print(f"[ARGS] what={sel}, start={args.start}, end={args.end}, years={args.years}")

    if "yield_curve" in sel:
        try:
            build_yield_curve(args.start, args.end, args.years)
        except Exception as e:
            print(f"[ERR] yield_curve 失败: {e}")

    if "ig_hy" in sel:
        try:
            fetch_ig_hy(args.start, args.end)
        except Exception as e:
            print(f"[ERR] ig_hy 失败: {e}")

    if "us10y" in sel:
        try:
            fetch_us10y(args.start, args.end)
        except Exception as e:
            print(f"[ERR] us10y 失败: {e}")

    if "dxy" in sel:
        try:
            fetch_dxy(args.start, args.end, args.years)
            print(args.end)
        except Exception as e:
            print(f"[ERR] dxy 失败: {e}")


    print("[DONE] All requested datasets have been built ✅")

if __name__ == "__main__":
    main()
