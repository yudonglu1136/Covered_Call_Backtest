# -*- coding: utf-8 -*-
"""
live_picker.py — Covered Call（日内：reference/contracts -> 聚合价 -> 反解IV/Δ）
- 不依赖 option_utils / src.quant_utils
输出：
  data/options_today_polygon.csv
  data/today_recommendations.csv
  data/debug_contracts_raw.csv
  data/debug_contracts_candidates.csv
  data/debug_pricing_input.csv
  data/debug_pricing_with_iv_delta.csv
"""

import os, sys, time, math, requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from dateutil.tz import gettz
from math import log, sqrt, exp
from scipy.stats import norm  
import os
from dotenv import load_dotenv
load_dotenv()  # will read .env into process env

# ========= 用户配置 =========
UNDERLYING = "QQQ"
EXCHANGE_TZ = "America/New_York"
DATA_DIR = Path("data"); DATA_DIR.mkdir(parents=True, exist_ok=True)

# DTE/行权/Δ规则
CALL_MIN_DTE = 28     
CALL_MAX_DTE = 30      
CALL_STRIKE_FLOOR_PCT = 1.06   
DELTA_BAND = 0.01               

RF_DEFAULT = 0.045

API_KEY = os.getenv("POLYGON_API_KEY")
if not API_KEY:
    print("请先设置 POLYGON_API_KEY", file=sys.stderr)
    sys.exit(1)

# ========= 时间与现价 =========
def now_et_date():
    return datetime.now(gettz(EXCHANGE_TZ)).date()

def get_spot_price(ticker: str) -> float:
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
    r = requests.get(url, params={"apiKey": API_KEY}, timeout=15)
    r.raise_for_status()
    j = r.json() or {}
    last = j.get("ticker", {}) or {}
    price = (last.get("lastTrade", {}) or {}).get("p")
    if price is None:
        q = last.get("lastQuote", {}) or {}
        bp, ap = q.get("bp"), q.get("ap")
        if bp is not None and ap is not None and ap > 0:
            price = (bp + ap) / 2.0
    if price is None:
        raise RuntimeError("无法取得现价")
    return float(price)

# ========= Polygon 数据层（不使用 options snapshot）=========
def list_call_contracts(spot: float) -> pd.DataFrame:
    """Reference 合约清单（只取 CALL + 目标 DTE 窗口）"""
    today = now_et_date()
    def iso_plus(d): return (pd.to_datetime(today) + pd.Timedelta(days=d)).date().isoformat()
    params = {
        "underlying_ticker": UNDERLYING,
        "contract_type": "call",
        "expired": "false",
        "expiration_date.gte": iso_plus(CALL_MIN_DTE),
        "expiration_date.lte": iso_plus(CALL_MAX_DTE),
        "strike_price.gte": 0.8 * spot,   # 先宽一些，后面再筛 ≥6%
        "strike_price.lte": 1.5 * spot,
        "order": "asc",
        "sort": "expiration_date",
        "limit": 1000,
        "apiKey": API_KEY,
    }
    url = "https://api.polygon.io/v3/reference/options/contracts"
    out, seen = [], set()
    while True:
        rr = requests.get(url, params=params, timeout=20)
        rr.raise_for_status()
        jj = rr.json() or {}
        res = jj.get("results", []) or []
        out.extend(res)
        next_url = jj.get("next_url")
        if not next_url or next_url in seen:
            break
        seen.add(next_url)
        url, params = next_url, {}
        time.sleep(0.03)

    rows = []
    for r in out:
        tkr = r.get("ticker") or r.get("options_symbol")
        K   = r.get("strike_price")
        exp = r.get("expiration_date")
        if not (tkr and K and exp): 
            continue
        rows.append({
            "ticker": tkr,
            "expiration": pd.to_datetime(exp, utc=True).tz_convert(None).normalize(),
            "strike": float(K),
        })
    df = pd.DataFrame(rows)
    if df.empty: return df
    today_ts = pd.to_datetime(today)
    df["dte"] = (df["expiration"] - today_ts).dt.days
    return df.sort_values(["expiration","strike"]).reset_index(drop=True)

def get_option_market_data(option_ticker: str, date_iso: str) -> dict | None:
    """取当天日级聚合（有 vw / c / h / l）；取不到返回 None"""
    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/day/{date_iso}/{date_iso}"
    r = requests.get(url, params={"apiKey": API_KEY, "limit": 1}, timeout=15)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    j = r.json() or {}
    res = j.get("results") or []
    if not res: return None
    return res[-1]

# ========= 风险利率 & Black-Scholes =========
def get_risk_free_rate(date_iso: str) -> float:
    # 简化：固定默认值（你可以替换成自家的 T-Bill/FED Data）
    return RF_DEFAULT

def year_fraction(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    return max(1e-8, (t1 - t0).days / 365.0)

def bs_price_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:  # 极端兜底
        return max(0.0, S - K)
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)

def bs_delta_call(S, K, T, r, sigma):
    if sigma <= 0 or T <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*sqrt(T))
    return norm.cdf(d1)

def implied_vol_call(S, K, T, r, price_target, tol=1e-6, max_iter=100):
    # 二分法
    low, high = 1e-6, 5.0
    p_low = bs_price_call(S, K, T, r, low)
    p_high = bs_price_call(S, K, T, r, high)
    # 如目标价超界，尽量贴边
    if price_target <= p_low: return low
    if price_target >= p_high: return high
    for _ in range(max_iter):
        mid = 0.5*(low+high)
        p_mid = bs_price_call(S, K, T, r, mid)
        if abs(p_mid - price_target) < tol:
            return mid
        if p_mid > price_target:
            high = mid
        else:
            low = mid
    return 0.5*(low+high)

def calculate_with_iv_delta(df: pd.DataFrame) -> pd.DataFrame:
    """输入列：underlying_price, strike, expiration, date, risk_free_rate, vw
       输出：增加 iv, delta 列
    """
    out = df.copy()
    ivs, deltas = [], []
    for _, row in out.iterrows():
        S = float(row["underlying_price"])
        K = float(row["strike"])
        r = float(row["risk_free_rate"])
        vw = float(row["vw"]) if pd.notna(row["vw"]) else np.nan
        if not np.isfinite(vw) or vw <= 0:
            ivs.append(np.nan); deltas.append(np.nan); continue
        T = year_fraction(pd.to_datetime(row["date"]), pd.to_datetime(row["expiration"]))
        sigma = implied_vol_call(S, K, T, r, vw)
        delta = bs_delta_call(S, K, T, r, sigma)
        ivs.append(sigma); deltas.append(delta)
    out["iv"] = ivs; out["delta"] = deltas
    return out

# ========= 目标Δ映射 & 选约逻辑 =========
def iv_to_delta(iv: float, steepness: float = 10, mid: float = 0.27,
                min_delta: float = 0.018, max_delta: float = 0.15) -> float:
    iv = float(iv) if np.isfinite(iv) else 0.27
    norm_v = 1.0 / (1.0 + math.exp(-steepness*(mid - iv)))
    return round(min_delta + (max_delta - min_delta)*norm_v, 4)
def pick_covered_call(spot: float, df: pd.DataFrame):
    """
    先尝试在 strike >= floor 的集合里选（优先 Δ 带宽内；否则 Δ 最近）。
    只有当 ≥floor 集合为空时，才退回到全体中选。
    """
    if df.empty:
        return None

    # 目标Δ
    iv_today = float(df["iv"].astype(float).dropna().mean()) if df["iv"].notna().any() else 0.25
    target_delta = iv_to_delta(iv_today)
    strike_floor = round(CALL_STRIKE_FLOOR_PCT * spot, 2)

    def _choose(pool: pd.DataFrame) -> pd.Series:
        p = pool.copy()
        # 缺失值处理，避免排序出 NaN
        p["mid"] = pd.to_numeric(p["mid"], errors="coerce").fillna(0.0)
        p["delta_abs"] = p["delta"].abs()
        p["delta_gap"] = (p["delta_abs"] - target_delta).abs()
        p["floor_gap"] = (p["strike"] - strike_floor).abs()

        band = p["delta_gap"] <= DELTA_BAND
        if band.any():
            row = p[band].sort_values(
                ["delta_gap", "floor_gap", "expiration", "mid"],
                ascending=[True, True, True, False],
            ).iloc[0]
            reason = "Δ在带宽内（首选集合）"
        else:
            row = p.sort_values(
                ["delta_gap", "floor_gap", "expiration", "mid"],
                ascending=[True, True, True, False],
            ).iloc[0]
            reason = "Δ最接近（首选集合）"

        row = row.copy()
        row["target_delta"] = target_delta
        row["iv_today"] = iv_today
        row["strike_floor"] = strike_floor
        row["meets_floor"] = bool(row["strike"] >= strike_floor)
        row["meets_band"]  = bool(abs(abs(row["delta"]) - target_delta) <= DELTA_BAND)
        row["reason"] = reason
        return row

    pool_floor = df[df["strike"] >= strike_floor]
    if not pool_floor.empty:
        return _choose(pool_floor)        # ★ 只要有 ≥floor，就一定从这堆里挑
    else:
        return _choose(df)                # 没有 ≥floor 再放宽



# ========= 主流程 =========
def main():
    print("拉取 QQQ 现价与参考合约（contracts + aggs + 反解IV/Δ）...")
    spot = get_spot_price(UNDERLYING)
    print(f"现价 QQQ ≈ ${spot:.2f}")

    # 1) contracts
    df_raw = list_call_contracts(spot)
    df_raw.to_csv(DATA_DIR / "debug_contracts_raw.csv", index=False)
    print(f"[DEBUG] reference/contracts 条数: {len(df_raw)}")
    if df_raw.empty:
        print("contracts 为空，退出。"); return

    # 2) 近 floor 的候选
    floor = CALL_STRIKE_FLOOR_PCT * spot
    today_ts = pd.to_datetime(now_et_date())
    df_raw["floor_gap"] = (df_raw["strike"] - floor).abs()
    df_cand = df_raw[(df_raw["dte"] >= CALL_MIN_DTE) & (df_raw["dte"] <= CALL_MAX_DTE)] \
                .sort_values(["expiration","floor_gap"]).head(80).reset_index(drop=True)
    df_cand.to_csv(DATA_DIR / "debug_contracts_candidates.csv", index=False)

    # 3) 拉聚合价，构造定价输入
    pricing_rows = []
    date_str = today_ts.date().isoformat()
    r = float(get_risk_free_rate(date_str))
    for _, rec in df_cand.iterrows():
        tkr = rec["ticker"]; K = float(rec["strike"]); exp = pd.to_datetime(rec["expiration"])
        md = get_option_market_data(tkr, date_str) or {}
        vw = md.get("vw")
        if vw is None:
            c = md.get("c"); h = md.get("h"); l = md.get("l")
            if c is not None: vw = float(c)
            elif h is not None and l is not None: vw = (float(h)+float(l))/2.0
        pricing_rows.append({
            "option_ticker": tkr,
            "underlying_price": float(spot),
            "strike": K,
            "expiration": exp,
            "date": today_ts,
            "risk_free_rate": r,
            "vw": float(vw) if vw is not None else np.nan,
        })
        time.sleep(0.01)

    df_price = pd.DataFrame(pricing_rows)
    df_price.to_csv(DATA_DIR / "debug_pricing_input.csv", index=False)
    if df_price.empty:
        print("没有可定价候选，退出。"); return

    # 4) 反解 IV/Δ
    df_iv = calculate_with_iv_delta(df_price.copy())
    df_iv.to_csv(DATA_DIR / "debug_pricing_with_iv_delta.csv", index=False)

    # 5) 合并，写候选链
    chain = df_cand.merge(df_iv[["option_ticker","iv","delta","vw"]],
                          left_on="ticker", right_on="option_ticker", how="left")
    chain = chain.drop(columns=["option_ticker"])
    chain["mid"] = chain["vw"].astype(float)
    out = chain.rename(columns={"ticker":"option_ticker"}).copy()
    out["dte"] = (pd.to_datetime(out["expiration"]) - today_ts).dt.days
    out = out[["option_ticker","expiration","strike","dte","iv","delta","mid"]].sort_values(
        ["expiration","strike"]).reset_index(drop=True)
    out_path = DATA_DIR / "options_today_polygon.csv"
    out.to_csv(out_path, index=False)
    print(f"候选 CALL 条目：{len(out)} | 已保存 {out_path}")

    if out.empty:
        print("候选链为空（聚合价都拿不到），退出。"); return

    # 6) 选约并输出
    pick = pick_covered_call(spot, out)
    if pick is None:
        print(f"[覆盖式 CALL] 未找到满足 {CALL_MIN_DTE}-{CALL_MAX_DTE}DTE 的合约。")
        return

    print("\n[覆盖式 CALL 建议]")
    print(f"  代码:       {pick['option_ticker']}")
    print(f"  行权价:     {pick['strike']:.2f}   (floor={pick['strike_floor']:.2f}  达标? {bool(pick['meets_floor'])})")
    print(f"  到期:       {pd.to_datetime(pick['expiration']).date().isoformat()}  (DTE={int(pick['dte'])})")
    print(f"  Δ:          {float(pick['delta']):.4f}  (目标≈{float(pick['target_delta']):.4f}  在带宽? {bool(pick['meets_band'])})")
    print(f"  IV:         {float(pick['iv']):.4f}")
    print(f"  中间价估计: ${float(pick['mid']):.2f}")
    print(f"  选择依据:   {pick.get('reason','')}")

    rec_csv = DATA_DIR / "today_recommendations.csv"
    pd.DataFrame([{
        "side":"SELL_CALL",
        "ticker": pick["option_ticker"],
        "strike": float(pick["strike"]),
        "expiration": pd.to_datetime(pick["expiration"]).date(),
        "dte": int(pick["dte"]),
        "delta": float(pick.get("delta", np.nan)),
        "iv": float(pick.get("iv", np.nan)),
        "mid": float(pick.get("mid", np.nan)),
        "spot": float(spot),
        "strike_floor": float(CALL_STRIKE_FLOOR_PCT*spot),
    }]).to_csv(rec_csv, index=False)
    print(f"\n推荐已保存: {rec_csv}")

if __name__ == "__main__":
    main()

