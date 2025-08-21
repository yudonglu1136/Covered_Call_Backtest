# -*- coding: utf-8 -*-
"""
Standalone inference for SINGLE XGB model:
- Load saved model: models/single_xgb_fw12d_thr-3pct.joblib
- Rebuild features (must match training FE)
- Predict probabilities -> apply saved threshold -> signal
- Save (date,signal) for 2021-07-26..today to data/put_singals.csv

Run:
  python scripts/make_put_signals_single.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

# =================== 固定路径 ===================
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
DATA.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS / "single_xgb_fw12d_thr-3pct.joblib"
OUT_CSV    = DATA / "put_signals.csv"   # 按你的要求命名（注意：singals）
# ===============================================

# ----------------- Robust loaders（与训练一致） -----------------
def _pick_col(df, cands):
    for c in cands:
        if c in df.columns:
            return c
    return None

def _load_csv_indexed(path: Path):
    df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        cl = c.lower()
        if "date" in cl or cl == "timestamp":
            date_col = c
            break
    if date_col is None:
        date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def load_qqq_close(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    for cand in ["Adj Close","adj_close","Adj_Close","Close","close","Open","open"]:
        if cand in df.columns:
            out = df[[cand]].rename(columns={cand:"close"})
            return out.asfreq("B").ffill()
    raise ValueError("QQQ csv lacks price column")

def load_ig_hy(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(path, index_col=0); d.index = pd.to_datetime(d.index)
    except Exception:
        d = _load_csv_indexed(path)
    d = d.sort_index()
    cols = {c.strip(): c.strip() for c in d.columns}
    d = d.rename(columns=cols)
    if "HY Spread" not in d.columns:
        cand = [c for c in d.columns if "HY" in c and "Spread" in c]
        d = d.rename(columns={cand[0]:"HY Spread"})
    if "IG Spread" not in d.columns:
        cand = [c for c in d.columns if "IG" in c and "Spread" in c]
        if cand:
            d = d.rename(columns={cand[0]:"IG Spread"})
        else:
            d["IG Spread"] = np.nan
    if "HY-IG Spread (bps)" not in d.columns:
        d["HY-IG Spread (bps)"] = (d["HY Spread"] - d["IG Spread"]) * 100.0
    return d.asfreq("B").ffill()[["HY Spread","IG Spread","HY-IG Spread (bps)"]]

def load_fgi(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    val_col = _pick_col(df, ["Value","value","FGI","Index","FearGreed"]) or df.columns[-1]
    out = df[[val_col]].rename(columns={val_col:"FGI"})
    out["FGI"] = pd.to_numeric(out["FGI"], errors="coerce")
    return out.asfreq("B").ffill()

def load_yield_curve(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    if "slope" in df.columns:
        out = df[["slope"]].copy()
    elif {"DGS10","DGS2"}.issubset(df.columns):
        out = pd.DataFrame(index=df.index)
        out["slope"] = pd.to_numeric(df["DGS10"], errors="coerce") - pd.to_numeric(df["DGS2"], errors="coerce")
    else:
        raise ValueError("yield_curve.csv needs slope or (DGS10,DGS2)")
    return out.asfreq("B").ffill()

def load_dxy(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    price_col = _pick_col(df, ["DXY_FRED","DXY","Adj Close","adj_close","Close","close","PX_LAST","Price","price"])
    out = df[[price_col]].rename(columns={price_col:"dxy"})
    out["dxy"] = pd.to_numeric(out["dxy"], errors="coerce")
    return out.asfreq("B").ffill()

def load_vix(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    vix_col = _pick_col(df, ["VIX","Adj Close","adj_close","Close","close","PX_LAST","Price","price","vix"])
    out = df[[vix_col]].rename(columns={vix_col:"vix"})
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    return out.asfreq("B").ffill()

def load_us10y(path: Path) -> pd.DataFrame:
    df = _load_csv_indexed(path)
    r_col = _pick_col(df, ["DGS10","US10Y","Close","close","PX_LAST","Rate","rate"])
    out = df[[r_col]].rename(columns={r_col:"us10y"})
    out["us10y"] = pd.to_numeric(out["us10y"], errors="coerce")
    return out.asfreq("B").ffill()

# ----------------- Feature Engineering（与训练一致） -----------------
def build_features():
    qqq = load_qqq_close(DATA/"QQQ_ohlcv_1d.csv")
    hy  = load_ig_hy(DATA/"ig_hy_fred.csv")
    fgi = load_fgi(DATA/"Fear_and_greed.csv")
    yc  = load_yield_curve(DATA/"yield_curve.csv")
    dxy = load_dxy(DATA/"dxy.csv")
    vix = load_vix(DATA/"VIX.csv")
    us10y = load_us10y(DATA/"US10Y.csv")

    df = qqq.join([hy,fgi,yc,dxy,vix,us10y], how="inner").dropna(
        subset=["close","HY Spread","slope","dxy","vix","us10y"]
    ).copy()

    # 基础
    df["ret1d"]   = df["close"].pct_change(1)
    df["vol20d"]  = df["ret1d"].rolling(20).std()

    # 信用
    df["hyig_diff_bps"]  = df["HY-IG Spread (bps)"]
    df["ig_level_pct"]   = df["IG Spread"]
    df["hy_vol_20d_bps"] = (df["HY Spread"].diff().rolling(20).std() * 100.0)

    # 曲线
    df["yc_slope"]     = df["slope"]
    df["yc_slope_sq"]  = df["yc_slope"]**2

    # FGI
    df["FGI"] = df["FGI"].ffill()

    # DXY
    df["dxy_ma20"]     = df["dxy"].rolling(20).mean()
    df["dxy_ma_gap20"] = df["dxy"]/df["dxy_ma20"] - 1.0

    # VIX
    df["vix_level"]    = df["vix"]
    df["vix_ma20"]     = df["vix"].rolling(20).mean()
    df["vix_ma_gap20"] = df["vix"]/df["vix_ma20"] - 1.0

    # 利率 regime
    roll3y = df["us10y"].rolling(756, min_periods=252)
    p75 = roll3y.quantile(0.75)
    df["regime_high_rate"] = (df["us10y"] >= p75).astype(int)
    df["us10y_chg_6m_bp"]  = (df["us10y"] - df["us10y"].shift(126)) * 100.0
    df["regime_easing"]    = (df["us10y_chg_6m_bp"] <= -50.0).astype(int)

    # 交互项
    df["fgi_hyig_inter"] = (100.0 - df["FGI"]) * df["hyig_diff_bps"]
    df["vix_yc_inter"]   = df["vix_level"] * df["yc_slope"]
    df["dxy_vix_inter"]  = df["dxy_ma_gap20"] * df["vix_level"]

    return df.dropna().copy()

# =============================== MAIN ===============================
if __name__ == "__main__":
    # 1) 加载模型与 meta（单模型保存格式：{"model":..., "meta":...}）
    bundle = load(MODEL_PATH)  # {"model": estimator, "meta": {...}}
    model = bundle["model"]
    meta  = bundle["meta"]
    features = meta["features"]
    thr = float(meta["threshold"])

    # 2) 重建特征
    df = build_features()
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing features in data: {missing}")

    # 3) 取 2021-07-26 到今天的子集
    start = pd.Timestamp("2021-07-26")
    end   = pd.Timestamp.today().normalize()
    df = df.loc[(df.index >= start) & (df.index <= end)].copy()

    # 4) 推理概率 -> 阈值出信号
    X = df[features].values
    p = model.predict_proba(X)[:, 1]
    signal = (p >= thr).astype(int)

    # 5) 保存 CSV（date, signal）
    out = pd.DataFrame({
        "date": df.index.strftime("%Y-%m-%d"),
        "signal": signal
    })
    out.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved signals -> {OUT_CSV} (rows={len(out)}, positives={int(signal.sum())})")
