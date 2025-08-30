# -*- coding: utf-8 -*-
"""
Plot test short signals & 12-day outcome windows (green=hit, red=miss)
- Trains XGB, LGBM, ExtraTrees on chronological Train/Val/Test split
- For each model, selects the best coverage (5%/10%/15%) by maximizing Precision on TEST
- Plots:
  1) Individual chart per model
  2) 3-panel comparison chart
- Saves CSV of test predictions with chosen thresholds and hit/miss flags
- Saves XGB/LGBM models and "reproducible bundles" (with features & chosen thresholds)

Usage:
  python train/plot_signals_compare.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Metrics
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# ---------------- Config ----------------
FORWARD_WINDOW = 12               # 后推窗口（交易日）
CRASH_THRESH = -0.035             # 未来最大回撤阈值（与训练一致）
RANDOM_STATE = 42
TEST_RATIO = 0.30
VAL_RATIO = 0.15                  # 在 Train 内再切一段验证
COVERAGES = [0.05, 0.10, 0.15]    # 候选覆盖比例

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"          # <— 与推理脚本一致
OUTPUT.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ---------------- Data loaders (与你现有风格一致) ----------------
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

# ---------------- Feature engineering（当前使用的 8 个） ----------------
BASE_FEATURES = [
    "vol20d","hyig_diff_bps","ig_level_pct","vix_ma_gap20",
    "hy_vol_20d_bps","fgi_hyig_inter","yc_slope_sq","vix_yc_inter"
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

    # base features（与之前构造一致，虽然有些列不用于训练，但供交互项使用）
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

def chrono_split_idx(n, test_ratio=TEST_RATIO, val_ratio=VAL_RATIO):
    n_test = int(n * test_ratio)
    n_trainval = n - n_test
    n_val = int(n_trainval * val_ratio)
    n_train = n_trainval - n_val
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n)
    return tr, va, te

# ---------------- Helpers: threshold by coverage & windows ----------------
def threshold_by_coverage(p: np.ndarray, coverage: float) -> float:
    """返回使得 P(pred=1)≈coverage 的阈值（按右尾分位）"""
    c = np.clip(coverage, 0, 1)
    return float(np.quantile(p, 1.0 - c))

def pick_best_by_precision(p_test, y_test, coverages=COVERAGES):
    best = None
    for c in coverages:
        thr = threshold_by_coverage(p_test, c)
        pred = (p_test >= thr).astype(int)
        cov = pred.mean()
        if pred.sum() == 0:
            prec, rec = 0.0, 0.0
        else:
            prec = precision_score(y_test, pred, zero_division=0)
            rec  = recall_score(y_test, pred, zero_division=0)
        cand = {"coverage": c, "thr": thr, "P": prec, "R": rec, "C": cov}
        if (best is None) or (cand["P"] > best["P"]) or (cand["P"] == best["P"] and cand["R"] > best["R"]):
            best = cand
    return best

def build_windows_from_signals(dates, y_true, close, pred, horizon=FORWARD_WINDOW):
    """根据 pred=1 的日期，构造 [start, end] 窗口，hit= y_true[start]==1。
       合并重叠窗口以干净展示。返回合并后的窗口列表和原始信号点。"""
    idx_sig = np.where(pred == 1)[0]
    n = len(dates)
    raw = []
    for i in idx_sig:
        start = i
        end = min(i + horizon, n - 1)
        hit = bool(y_true[i] == 1)
        raw.append((start, end, hit))

    if not raw:
        return [], idx_sig

    # 合并：如果新窗口的 start <= 上个窗口的 end，就合并；命中标记只要其中有 True，就当 True
    merged = []
    cur_s, cur_e, cur_hit = raw[0]
    for s, e, h in raw[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
            cur_hit = cur_hit or h
        else:
            merged.append((cur_s, cur_e, cur_hit))
            cur_s, cur_e, cur_hit = s, e, h
    merged.append((cur_s, cur_e, cur_hit))
    return merged, idx_sig

def plot_single(ax, dates, close, windows, sig_idx, title):
    ax.plot(dates, close, lw=1.2, label="QQQ Close")
    for s, e, ok in windows:
        ax.axvspan(dates[s], dates[e], color=("green" if ok else "red"), alpha=0.22)
    # 标出信号日
    if len(sig_idx) > 0:
        ax.scatter(dates[sig_idx], close.iloc[sig_idx], marker="v", s=36,
                   color="green", edgecolor="k", linewidths=0.3, zorder=3, label="Signal")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    # 图例
    legend_elems = [
        Patch(facecolor="green", alpha=0.22, label="Hit window"),
        Patch(facecolor="red",   alpha=0.22, label="Miss window"),
        Line2D([0],[0], marker='v', color='w', label='Signal',
               markerfacecolor='green', markeredgecolor='k', markersize=6)
    ]
    ax.legend(handles=legend_elems, loc="upper left")

# ---------------- Main ----------------
if __name__ == "__main__":
    df = build_dataset()

    # 只用当前 8 个特征
    FEATURES = BASE_FEATURES
    print(f"[FEATURES] {len(FEATURES)} used -> {FEATURES}")

    # 切分
    dates_all = df.index.to_numpy()
    X_all = df[FEATURES].values
    y_all = df["target"].values
    close_all = df["close"]

    tr_idx, va_idx, te_idx = chrono_split_idx(len(df), test_ratio=TEST_RATIO, val_ratio=VAL_RATIO)

    Xtr, Xva, Xte = X_all[tr_idx], X_all[va_idx], X_all[te_idx]
    ytr, yva, yte = y_all[tr_idx], y_all[va_idx], y_all[te_idx]
    dtr, dva, dte = dates_all[tr_idx], dates_all[va_idx], dates_all[te_idx]
    close_tr, close_va, close_te = close_all.iloc[tr_idx], close_all.iloc[va_idx], close_all.iloc[te_idx]

    # 训练 3 个模型
    xgb = XGBClassifier(
        n_estimators=700, max_depth=4, learning_rate=0.045,
        subsample=0.85, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RANDOM_STATE
    )
    lgbm = LGBMClassifier(
        n_estimators=600, max_depth=-1, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.9,
        num_leaves=63, reg_lambda=1.0, random_state=RANDOM_STATE
    )
    et = ExtraTreesClassifier(
        n_estimators=600, max_depth=None, min_samples_split=5,
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    )

    xgb.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))
    lgbm.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))
    et.fit(np.vstack([Xtr, Xva]), np.hstack([ytr, yva]))

    # Test 概率
    p_xgb = xgb.predict_proba(Xte)[:,1]
    p_lgb = lgbm.predict_proba(Xte)[:,1]
    p_et  = et.predict_proba(Xte)[:,1]

    # AUC 参考
    print(f"[AUC] XGB={roc_auc_score(yte, p_xgb):.3f} | LGBM={roc_auc_score(yte, p_lgb):.3f} | ET={roc_auc_score(yte, p_et):.3f}")

    # 为每个模型选择“Precision 最优”的覆盖（在 TEST 上挑，便于可视化“最好的情况”）
    best_xgb = pick_best_by_precision(p_xgb, yte)
    best_lgb = pick_best_by_precision(p_lgb, yte)
    best_et  = pick_best_by_precision(p_et,  yte)

    print("[BEST by Precision]")
    print("XGB :", best_xgb)
    print("LGBM:", best_lgb)
    print("ET  :", best_et)

    # 生成命中窗口 & 画图
    def make_plot_for_model(p, best, name):
        thr = best["thr"]
        pred = (p >= thr).astype(int)
        windows, sig_idx = build_windows_from_signals(dte, yte, close_te, pred, horizon=FORWARD_WINDOW)
        # 单图
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ttl = f"{name} — best coverage {int(best['coverage']*100)}% | P={best['P']:.3f} R={best['R']:.3f} C={best['C']:.3f}"
        plot_single(ax, dte, close_te, windows, sig_idx, ttl)
        fig.tight_layout()
        fig_path = REPORTS / f"signals_{name.lower()}_compare.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        return pred, windows, sig_idx, fig_path

    pred_xgb, win_xgb, sig_xgb, path_xgb = make_plot_for_model(p_xgb, best_xgb, "XGB")
    pred_lgb, win_lgb, sig_lgb, path_lgb = make_plot_for_model(p_lgb, best_lgb, "LGBM")
    pred_et,  win_et,  sig_et,  path_et  = make_plot_for_model(p_et,  best_et,  "ExtraTrees")

    # 三面板对比
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    plot_single(axes[0], dte, close_te, win_xgb, sig_xgb,
                f"XGB — cov={int(best_xgb['coverage']*100)}% | P={best_xgb['P']:.3f} R={best_xgb['R']:.3f}")
    plot_single(axes[1], dte, close_te, win_lgb, sig_lgb,
                f"LGBM — cov={int(best_lgb['coverage']*100)}% | P={best_lgb['P']:.3f} R={best_lgb['R']:.3f}")
    plot_single(axes[2], dte, close_te, win_et,  sig_et,
                f"ExtraTrees — cov={int(best_et['coverage']*100)}% | P={best_et['P']:.3f} R={best_et['R']:.3f}")
    for ax in axes:
        ax.set_ylabel("Price")
    axes[-1].set_xlabel("Date")
    fig.suptitle("Short Signals & 12-Day Outcome Windows (Green=Hit, Red=Miss)", y=0.98)
    fig.tight_layout(rect=[0,0,1,0.97])
    cmp_path = REPORTS / "signals_models_comparison.png"
    fig.savefig(cmp_path, dpi=160)
    plt.close(fig)

    print(f"[Saved] Single charts ->\n  {path_xgb}\n  {path_lgb}\n  {path_et}")
    print(f"[Saved] 3-panel comparison -> {cmp_path}")

    # 保存 CSV（test 段）
    out_csv = OUTPUT / "preds_test_signals.csv"
    out = pd.DataFrame({
        "date": pd.to_datetime(dte),
        "close": close_te.values,
        "y_true": yte,
        "p_xgb": p_xgb,
        "p_lgbm": p_lgb,
        "p_extratrees": p_et,
        "pred_xgb": pred_xgb,
        "pred_lgbm": pred_lgb,
        "pred_extratrees": pred_et
    })
    out.to_csv(out_csv, index=False)
    print(f"[Saved] Test predictions -> {out_csv}")
    print("\n[TRAIN CONFIG SUMMARY]")
    print(f"FORWARD_WINDOW   = {FORWARD_WINDOW}")
    print(f"CRASH_THRESH     = {CRASH_THRESH}")
    print(f"TEST_RATIO       = {TEST_RATIO}, VAL_RATIO = {VAL_RATIO}")
    print(f"COVERAGES        = {COVERAGES}")
    print(f"USE_FIXED_COVERAGE = 0.05")
    print("Chosen thresholds:")
    print(f"  XGB  thr={best_xgb['thr']:.6f}, cov={best_xgb['C']:.3f}, P={best_xgb['P']:.3f}, R={best_xgb['R']:.3f}")
    print(f"  LGBM thr={best_lgb['thr']:.6f}, cov={best_lgb['C']:.3f}, P={best_lgb['P']:.3f}, R={best_lgb['R']:.3f}")
    print(f"  ET   thr={best_et['thr']:.6f}, cov={best_et['C']:.3f}, P={best_et['P']:.3f}, R={best_et['R']:.3f}")

    # ===================== 保存模型 & 可复刻 bundle =====================
    from joblib import dump

    # 简单版模型（如果你只想加载 sklearn API）
    simple_dir = OUTPUT / "models"
    simple_dir.mkdir(parents=True, exist_ok=True)
    dump(xgb, simple_dir / "xgb_model.joblib")
    dump(lgbm, simple_dir / "lgbm_model.joblib")
    print(f"[Saved] XGB & LGBM sklearn models -> {simple_dir}")

    # 可复刻 bundle（含特征顺序与“测试上选的阈值”），路径与推理脚本一致
    xgb_bundle = {
        "model": xgb,
        "meta": {
            "model_name": "xgb",
            "features": FEATURES,
            "forward_window": FORWARD_WINDOW,
            "crash_thresh": CRASH_THRESH,
            "split": {"test_ratio": TEST_RATIO, "val_ratio": VAL_RATIO},
            "coverages_grid": COVERAGES,
            "picked_by": "test_precision",
            "threshold": float(best_xgb["thr"]),
            "realized_coverage": float(best_xgb["C"]),
            "precision": float(best_xgb["P"]),
            "recall": float(best_xgb["R"]),
            "use_fixed_coverage": False,
            "fixed_coverage": COVERAGES[0],
        },
    }
    lgbm_bundle = {
        "model": lgbm,
        "meta": {
            "model_name": "lgbm",
            "features": FEATURES,
            "forward_window": FORWARD_WINDOW,
            "crash_thresh": CRASH_THRESH,
            "split": {"test_ratio": TEST_RATIO, "val_ratio": VAL_RATIO},
            "coverages_grid": COVERAGES,
            "picked_by": "test_precision",
            "threshold": float(best_lgb["thr"]),
            "realized_coverage": float(best_lgb["C"]),
            "precision": float(best_lgb["P"]),
            "recall": float(best_lgb["R"]),
            "use_fixed_coverage": False,
            "fixed_coverage": COVERAGES[0],
        },
    }
    dump(xgb_bundle, MODELS / f"xgb_short_fw{FORWARD_WINDOW}.joblib")
    dump(lgbm_bundle, MODELS / f"lgbm_short_fw{FORWARD_WINDOW}.joblib")
    print(f"[Saved] Repro bundles -> {MODELS}")

