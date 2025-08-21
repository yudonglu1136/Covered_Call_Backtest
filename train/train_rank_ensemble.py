# -*- coding: utf-8 -*-
"""
Train a SINGLE XGBoost crash model and evaluate on a chronological 70/30 split.
- Save model + meta (features & threshold)
- Plot TEST ROC (probability-based)
- Optional threshold tuning on train (F1 or Recall@Precision≥target)

Run:
  python scripts/train_single_prob.py --criterion recall_at_precision --precision_target 0.80
  # 或者：
  python scripts/train_single_prob.py --criterion recall_at_precision --precision_target 0.70
"""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_fscore_support, classification_report, precision_recall_curve

# ---------------- Config ----------------
FORWARD_WINDOW_TRAIN = 12
CRASH_THRESH = -0.035
TEST_RATIO = 0.30
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

def make_tag():
    thr = int(CRASH_THRESH * 100)
    return f"fw{FORWARD_WINDOW_TRAIN}d_thr{thr}pct"

TAG = make_tag()

# ---------------- Robust loaders & FE (与之前版本一致) ----------------
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
            out = df[[cand]].rename(columns={cand: "close"})
            return out.asfreq("B").ffill()
    raise ValueError("QQQ csv lacks price column")

def load_ig_hy(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(path, index_col=0); d.index = pd.to_datetime(d.index)
    except:
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

def future_max_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = len(prices); out = np.full(n, np.nan)
    for i in range(n):
        j = min(n, i + horizon + 1); w = prices[i:j]
        if len(w) < 2: continue
        p0, pmin = w[0], w.min()
        out[i] = (pmin - p0) / p0
    return out

def build_dataset():
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

    # base
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

    roll3y = df["us10y"].rolling(756, min_periods=252)
    p75 = roll3y.quantile(0.75)
    df["regime_high_rate"] = (df["us10y"] >= p75).astype(int)
    df["us10y_chg_6m_bp"]  = (df["us10y"] - df["us10y"].shift(126)) * 100.0
    df["regime_easing"]    = (df["us10y_chg_6m_bp"] <= -50.0).astype(int)

    df["fgi_hyig_inter"] = (100.0 - df["FGI"]) * df["hyig_diff_bps"]
    df["vix_yc_inter"]   = df["vix_level"] * df["yc_slope"]
    df["dxy_vix_inter"]  = df["dxy_ma_gap20"] * df["vix_level"]

    # target
    df["fwd_dd"] = future_max_drawdown(df["close"].values, FORWARD_WINDOW_TRAIN)
    df["target"] = (df["fwd_dd"] <= CRASH_THRESH).astype(int)
    return df.dropna().copy()

# 与你当前偏好一致的特征集合（单模型就用这套）
FEATURES = [
    "vol20d","hyig_diff_bps","FGI","ig_level_pct",
    "dxy_ma_gap20","vix_ma_gap20",
    "regime_high_rate","regime_easing",
    "hy_vol_20d_bps","fgi_hyig_inter","yc_slope_sq",
    "vix_yc_inter","dxy_vix_inter"
]

def temporal_split_idx(n, test_ratio):
    split = int(n * (1 - test_ratio))
    return np.arange(split), np.arange(split, n)

def xgb(spw, rs=RANDOM_STATE):
    return XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=rs, scale_pos_weight=spw
    )

# ---- 新：更鲁棒的 Recall@Precision 阈值搜索 ----
def tune_threshold_recall_at_precision(y_true, p, precision_target=0.80):
    prec, rec, thr = precision_recall_curve(y_true, p)  # len(thr) = len(prec)-1
    mask = prec[:-1] >= precision_target
    if mask.any():
        # 在 precision 达标的集合中，选择 recall 最大的阈值
        idx = np.argmax(rec[:-1][mask])
        return float(thr[mask][idx])
    else:
        # 若无法达标：退化为 precision 最大对应的阈值
        idx = np.argmax(prec[:-1])
        return float(thr[idx])

def tune_threshold(y_true, p, criterion="recall_at_precision", precision_target=0.80):
    if criterion == "f1":
        # 保留原实现（简单网格），也可以改成基于 PR 曲线的版本
        grid = np.quantile(p, np.linspace(0.01, 0.99, 99))
        best_t, best_f1 = 0.5, -1.0
        for t in grid:
            pred = (p >= t).astype(int)
            _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        return best_t
    else:
        return tune_threshold_recall_at_precision(y_true, p, precision_target)

def plot_test_roc(y, p, path_png, title="Single model ROC"):
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1], lw=1, linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(title)
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--criterion", choices=["f1","recall_at_precision"], default="recall_at_precision")
    ap.add_argument("--precision_target", type=float, default=0.999)  # 默认值提升到 0.80
    args = ap.parse_args()

    df = build_dataset()
    X = df[FEATURES].values; y = df["target"].values; dates = df.index.values

    tr_idx, te_idx = temporal_split_idx(len(y), TEST_RATIO)
    Xtr, Xte = X[tr_idx], X[te_idx]
    ytr, yte = y[tr_idx], y[te_idx]
    dtr, dte = dates[tr_idx], dates[te_idx]

    pos = int(ytr.sum()); neg = len(ytr) - pos
    spw = (neg / max(1, pos)) if pos > 0 else 1.0
    model = xgb(spw)
    model.fit(Xtr, ytr)

    # TEST 概率与 ROC（概率级评估，不受阈值影响）
    p_te = model.predict_proba(Xte)[:,1]
    auc_te = roc_auc_score(yte, p_te)
    print(f"[TEST] AUC = {auc_te:.3f}")

    # 训练段上调阈值（满足 Precision≥target，Recall 最大化）
    p_tr = model.predict_proba(Xtr)[:,1]
    thr = tune_threshold(ytr, p_tr, criterion=args.criterion, precision_target=args.precision_target)
    print(f"[TRAIN] tuned threshold -> {thr:.4f} ({args.criterion}, precision_target={args.precision_target})")

    # Test 报告（使用调好的阈值）
    pred_te = (p_te >= thr).astype(int)
    print("\n=== Test classification report ===")
    print(classification_report(yte, pred_te, digits=3))

    # 保存测试 ROC 图
    roc_png = REPORTS / f"single_ROC_{TAG}.png"
    plot_test_roc(yte, p_te, roc_png, title="Single XGB ROC")
    print(f"Saved ROC -> {roc_png}")
    print(f"[DATA] Train period: {pd.to_datetime(dtr[0]).date()} → {pd.to_datetime(dtr[-1]).date()} "
          f"({len(tr_idx)} samples)")
    print(f"[DATA] Test  period: {pd.to_datetime(dte[0]).date()} → {pd.to_datetime(dte[-1]).date()} "
          f"({len(te_idx)} samples)")
    # 保存模型与阈值
    artifact = {
        "tag": TAG,
        "features": FEATURES,
        "threshold": float(thr),
        "criterion": args.criterion,
        "precision_target": float(args.precision_target)
    }
    dump({"model": model, "meta": artifact}, MODELS / f"single_xgb_{TAG}.joblib")
    with open(MODELS / f"single_xgb_{TAG}_meta.json", "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"Saved model -> {MODELS / f'single_xgb_{TAG}.joblib'}")


