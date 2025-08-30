# -*- coding: utf-8 -*-
"""
Generate two CSVs and a plot:
1) put_signals_real.csv  (date, signal)          —— 当日“真实信号”
2) put_signals_T1.csv    (source_date, date, signal) —— 上述信号统一顺延 1 个交易日（T+1）
3) put_signals_on_qqq.(png/pdf) —— 在 QQQ 上标记“真实信号”，标题显示最近一次触发日期

阈值模式与上一版一致：
- "replay_test"：按训练 TEST 逻辑复现阈值
- "exact_thr"  ：使用给定阈值
- "rolling"    ：滚动分位
- "global"     ：整体分位
- "fixed"      ：读 meta['threshold'] 或 trainval_q95，否则退化 global
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pandas.tseries.offsets import BDay
from joblib import load

# ================ User options ================
MODEL_NAME        = "xgb"        # "xgb" or "lgbm"
START_DATE        = "2021-07-26"
END_DATE          = None         # None -> today

THRESH_MODE       = "replay_test"   # "replay_test" | "exact_thr" | "rolling" | "global" | "fixed"
GIVEN_THRESHOLD   = 0.962986        # 仅 exact_thr 使用

ROLLING_WINDOW    = 252             # 约 1 年交易日
TARGET_COVERAGE   = 0.05            # rolling / global 覆盖

# 可选：强制覆盖率（只对 rolling/global 生效）
OVERRIDE_COVERAGE = False
OVERRIDE_COV_VAL  = 0.05

DEBUG_PRINTS      = True
# ============================================

# -------- 与训练保持一致的常量 --------
FORWARD_WINDOW = 12
CRASH_THRESH   = -0.035
TEST_RATIO     = 0.30
VAL_RATIO      = 0.15

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
MODELS = ROOT / "models"
OUTPUT = ROOT / "output"
DATA.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "xgb":  MODELS / f"xgb_short_fw{FORWARD_WINDOW}.joblib",
    "lgbm": MODELS / f"lgbm_short_fw{FORWARD_WINDOW}.joblib",
}
CSV_REAL = DATA / "put_signals_real.csv"
CSV_T1   = DATA / "put_signals_T1.csv"
PLOT_PNG = OUTPUT / "put_signals_on_qqq.png"
PLOT_PDF = OUTPUT / "put_signals_on_qqq.pdf"

# ---------------- Robust loaders（与训练一致） ----------------
def _pick_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    return None

def _load_csv_indexed(path: Path):
    df = pd.read_csv(path)
    date_col = None
    for c in df.columns:
        if "date" in c.lower() or c.lower()=="timestamp":
            date_col = c; break
    if date_col is None: date_col = df.columns[0]
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
        if cand: d = d.rename(columns={cand[0]:"IG Spread"})
        else: d["IG Spread"] = np.nan
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
    candidates = ["dxy","DXY","DXY_FRED","Adj Close","adj_close","Close","close","PX_LAST","Price","price"]
    price_col = next((c for c in candidates if c in df.columns), None)
    if price_col is None and len(df.columns)==1:
        price_col = df.columns[0]
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

# ---------------- 训练一致的 11 个特征 ----------------
BASE_FEATURES = [
    "vol20d","hyig_diff_bps","FGI","ig_level_pct",
    "dxy_ma_gap20","vix_ma_gap20",
    "hy_vol_20d_bps","fgi_hyig_inter","yc_slope_sq",
    "vix_yc_inter","dxy_vix_inter"
]

def future_max_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = len(prices); out = np.full(n, np.nan)
    for i in range(n):
        j = min(n, i + horizon + 1); w = prices[i:j]
        if len(w) < 2: continue
        p0, pmin = w[0], w.min()
        out[i] = (pmin - p0) / p0
    return out

def build_dataset():
    """训练同款：返回包含特征 + target 的完整表。"""
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

    # base features
    df["ret1d"]   = df["close"].pct_change(1)
    df["vol20d"]  = df["ret1d"].rolling(20).std()

    df["hyig_diff_bps"]  = df["HY-IG Spread (bps)"]
    df["ig_level_pct"]   = df["IG Spread"]
    df["hy_vol_20d_bps"] = (df["HY Spread"].diff().rolling(20).std() * 100.0)

    df["yc_slope"]     = df["slope"]
    df["yc_slope_sq"]  = df["yc_slope"]**2

    df["FGI"] = df["FGI"].ffill()

    df["dxy_ma20"]     = df["dxy"].rolling(20).mean()
    df["dxy_ma_gap20"] = df["dxy"]/df["dxy_ma20"] - 1.0

    df["vix_level"]    = df["vix"]
    df["vix_ma20"]     = df["vix"].rolling(20).mean()
    df["vix_ma_gap20"] = df["vix"]/df["vix_ma20"] - 1.0

    df["fgi_hyig_inter"] = (100.0 - df["FGI"]) * df["hyig_diff_bps"]
    df["vix_yc_inter"]   = df["vix_level"] * df["yc_slope"]
    df["dxy_vix_inter"]  = df["dxy_ma_gap20"] * df["vix_level"]

    # target
    df["fwd_dd"] = future_max_drawdown(df["close"].values, FORWARD_WINDOW)
    df["target"] = (df["fwd_dd"] <= CRASH_THRESH).astype(int)

    df = df.dropna().copy()
    return df

def build_features_only():
    """仅特征，用于任意日期范围推理。"""
    return build_dataset().drop(columns=["fwd_dd","target"])

def chrono_split_idx(n, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    n_test = int(n * test_ratio)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n)
    return tr, va, te

# ---------------- 阈值工具 ----------------
def threshold_by_coverage(p: np.ndarray, coverage: float) -> float:
    c = float(np.clip(coverage, 0.0, 1.0))
    return float(np.quantile(p, 1.0 - c))

def _rolling_quantile(s: pd.Series, cov: float, win: int) -> pd.Series:
    q = 1.0 - float(np.clip(cov, 0.0, 1.0))
    r = s.rolling(win, min_periods=max(20, int(win*0.3))).quantile(q)
    e = s.expanding(min_periods=20).quantile(q)
    return r.combine_first(e)

# =============================== MAIN ===============================
if __name__ == "__main__":
    # 1) 加载模型
    model_path = MODEL_PATHS.get(MODEL_NAME.lower())
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    bundle = load(model_path)
    model = bundle["model"]
    meta  = bundle.get("meta", {})

    # 2) 构造 FE 并裁剪输出区间
    df_fe = build_features_only()
    features = meta.get("features", BASE_FEATURES)
    for f in features:
        if f not in df_fe.columns: df_fe[f] = np.nan
    df_fe = df_fe.dropna(subset=features)

    start = pd.Timestamp(START_DATE) if START_DATE else df_fe.index.min()
    end_cap = pd.to_datetime(END_DATE).normalize() if END_DATE else pd.Timestamp.today().normalize()
    df_out = df_fe.loc[(df_fe.index >= start) & (df_fe.index <= end_cap)].copy()
    if df_out.empty:
        raise RuntimeError("No rows in the requested date range after feature building.")

    X_out = df_out[features].values
    p_out = model.predict_proba(X_out)[:, 1]
    p_out_series = pd.Series(p_out, index=df_out.index)

    thr_used = None
    thr_desc = ""
    signal_series = None  # 当日真实信号

    # 3) 选择阈值并生成“当天的模型信号”
    if THRESH_MODE == "replay_test":
        df_full = build_dataset()
        X_all = df_full[features].values
        y_all = df_full["target"].values
        n = len(df_full)
        tr, va, te = chrono_split_idx(n)
        Xte = X_all[te]; yte = y_all[te]

        p_test = model.predict_proba(Xte)[:, 1]

        use_fixed = bool(meta.get("use_fixed_coverage", True))
        if use_fixed:
            cov = float(meta.get("fixed_coverage", TARGET_COVERAGE))
            thr_used = threshold_by_coverage(p_test, cov)
            thr_desc = f"replay_test fixed @{cov:.0%} (thr={thr_used:.6f})"
        else:
            grid = meta.get("coverages_grid", [0.05, 0.10, 0.15])
            from sklearn.metrics import precision_score
            best = None
            for c in grid:
                thr = threshold_by_coverage(p_test, c)
                pred = (p_test >= thr).astype(int)
                prec = precision_score(yte, pred, zero_division=0)
                cand = (prec, -c, thr)  # 先比 P，再偏好更小覆盖
                if (best is None) or (cand > best): best = cand
            thr_used = best[2]
            thr_desc = f"replay_test grid={grid} -> thr={thr_used:.6f}"
        signal_series = (p_out_series >= thr_used).astype(int)

    elif THRESH_MODE == "exact_thr":
        thr_used = float(GIVEN_THRESHOLD)
        thr_desc = f"exact_thr={thr_used:.6f}"
        signal_series = (p_out_series >= thr_used).astype(int)

    elif THRESH_MODE == "rolling":
        cov = OVERRIDE_COV_VAL if OVERRIDE_COVERAGE else TARGET_COVERAGE
        thr_series = _rolling_quantile(p_out_series, cov, ROLLING_WINDOW)
        signal_series = (p_out_series >= thr_series).astype(int)
        thr_desc = f"rolling-{ROLLING_WINDOW}d @{cov:.0%}"

    elif THRESH_MODE == "global":
        cov = OVERRIDE_COV_VAL if OVERRIDE_COVERAGE else TARGET_COVERAGE
        thr_used = threshold_by_coverage(p_out, cov)
        signal_series = (p_out_series >= thr_used).astype(int)
        thr_desc = f"global @{cov:.0%} (thr={thr_used:.6f})"

    elif THRESH_MODE == "fixed":
        if "threshold" in meta:
            thr_used = float(meta["threshold"])
            thr_desc = f"fixed meta.threshold={thr_used:.6f}"
        elif "threshold_from_trainval_q95" in meta:
            thr_used = float(meta["threshold_from_trainval_q95"])
            thr_desc = f"fixed meta.trainval_q95={thr_used:.6f}"
        else:
            cov = OVERRIDE_COV_VAL if OVERRIDE_COVERAGE else TARGET_COVERAGE
            thr_used = threshold_by_coverage(p_out, cov)
            thr_desc = f"fixed fallback global @{cov:.0%} thr={thr_used:.6f}"
        signal_series = (p_out_series >= thr_used).astype(int)

    else:
        raise ValueError(f"Unknown THRESH_MODE={THRESH_MODE}")

    # 4) 保存两份 CSV
    # 4.1 当日真实信号
    real_df = pd.DataFrame({
        "date": signal_series.index.strftime("%Y-%m-%d"),
        "signal": signal_series.values.astype(int)
    })
    real_df.to_csv(CSV_REAL, index=False, encoding="utf-8")

    # 4.2 T+1 信号（顺延一个交易日；不超过 end_cap）
    t_dates  = signal_series.index
    t1_dates = t_dates + BDay(1)
    valid    = t1_dates <= end_cap
    t1_df = pd.DataFrame({
        "source_date": t_dates[valid].strftime("%Y-%m-%d"),
        "date":        t1_dates[valid].strftime("%Y-%m-%d"),
        "signal":      signal_series.values[valid].astype(int)
    })
    t1_df.to_csv(CSV_T1, index=False, encoding="utf-8")

    # 5) 画图：QQQ + 当日真实信号
    qqq = load_qqq_close(DATA/"QQQ_ohlcv_1d.csv").loc[signal_series.index.min():signal_series.index.max()]
    # 与信号日期对齐
    qqq_plot = qqq.reindex(signal_series.index).ffill()
    sig_idx = np.where(signal_series.values == 1)[0]
    last_sig_date = signal_series.index[sig_idx[-1]].date().isoformat() if len(sig_idx) else "N/A"

    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.plot(qqq_plot.index, qqq_plot["close"].values, lw=1.2, label="QQQ Close")
    if len(sig_idx) > 0:
        ax.scatter(signal_series.index[sig_idx],
                   qqq_plot["close"].values[sig_idx],
                   marker="v", s=46, color="tab:orange",
                   edgecolor="k", linewidths=0.4, zorder=3,
                   label="PUT Signal (real)")
    ax.set_title(f"QQQ with Real PUT Signals  |  Last trigger: {last_sig_date}")
    ax.set_xlabel("Date"); ax.set_ylabel("Price (USD)")
    ax.grid(True, alpha=0.3); ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(PLOT_PNG, dpi=150, bbox_inches="tight")
    plt.savefig(PLOT_PDF, bbox_inches="tight")
    plt.close()

    # 6) Debug prints
    if DEBUG_PRINTS:
        qs = np.quantile(p_out, [0.5,0.9,0.95,0.99])
        print("[DEBUG] p_out quantiles: med={:.6f} 90%={:.6f} 95%={:.6f} 99%={:.6f}".format(*qs))
        print(f"[META ] use_fixed_coverage={meta.get('use_fixed_coverage', True)}  "
              f"fixed_coverage={meta.get('fixed_coverage', 0.05)}  "
              f"grid={meta.get('coverages_grid', [0.05,0.10,0.15])}")
        if not t1_df.empty:
            diffs = (pd.to_datetime(t1_df["date"]) - pd.to_datetime(t1_df["source_date"])).dt.days.unique().tolist()
            print(f"[CHECK] T+1 diff unique = {diffs}  (应为 [1]；节假日可能>1)")
        print(real_df.tail(3).to_string(index=False))
        print(t1_df.tail(3).to_string(index=False))

    n_real = len(real_df); pos_real = int(real_df["signal"].sum())
    n_t1   = len(t1_df);   pos_t1   = int(t1_df["signal"].sum())

    print(f"[MODEL] {MODEL_NAME.upper()}  path={model_path.name}")
    print(f"[THR  ] mode={THRESH_MODE} -> {thr_desc}")
    print(f"[SAVE ] real  -> {CSV_REAL}   ({pos_real}/{n_real} = {pos_real/n_real:.2%} ones)")
    print(f"[SAVE ] T+1   -> {CSV_T1}     ({pos_t1}/{n_t1} = {pos_t1/n_t1:.2%} ones)")
    print(f"[PLOT ] {PLOT_PNG}  |  {PLOT_PDF}  (Last real-signal date: {last_sig_date})")




