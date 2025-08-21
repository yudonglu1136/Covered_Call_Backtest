# -*- coding: utf-8 -*-
"""
Automated feature selection for macro+credit crash model (QQQ) + Visualizations
- 按你的要求更新了特征集与数据源
- 生成 permutation 与 XGB gain 的重要性图
- 生成候选 ΔAUC 排名图（若有候选）

Inputs (under data/):
  QQQ_ohlcv_1d.csv
  ig_hy_fred.csv
  Fear_and_greed.csv
  yield_curve.csv
  dxy.csv
  VIX.csv
  US10Y.csv

Outputs (under reports/):
  interaction_search_{TAG}.csv
  final_features_{TAG}.csv
  perm_importance_base_{TAG}.png
  xgb_gain_base_{TAG}.png
  candidate_delta_auc_{TAG}.png (可选, 若有候选)
  perm_importance_final_{TAG}.png
  xgb_gain_final_{TAG}.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.inspection import permutation_importance

# ---------------- User config ----------------
FORWARD_WINDOW_TRAIN = 12        # 未来N个交易日
CRASH_THRESH = -0.035            # 定义“未来N日最大回撤<=阈值”为1类
TEST_RATIO = 0.30                # 后30%为测试集
RANDOM_STATE = 42
N_SELECT = 3                     # Top-N 候选并入最终集合（若 candidates 为空则无效）

# Regime 配置（可调）
REGIME_HIGH_RATE_THRESH = 4.0    # us10y > 4% 视为高利率环境
REGIME_EASING_WINDOW_D = 60      # 近60个交易日
REGIME_EASING_DROP_BP  = 50.0    # 下行阈值（基点）

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

QQQ_PATH = DATA / "QQQ_ohlcv_1d.csv"
HY_PATH  = DATA / "ig_hy_fred.csv"
FG_PATH  = DATA / "Fear_and_greed.csv"
YC_PATH  = DATA / "yield_curve.csv"
DXY_PATH = DATA / "dxy.csv"
VIX_PATH = DATA / "VIX.csv"
US10Y_PATH = DATA / "US10Y.csv"
OPTIONS_PATH= DATA / "options_with_iv_delta.csv"

def make_tag():
    thr = int(CRASH_THRESH * 100)  # negative
    return f"fw{FORWARD_WINDOW_TRAIN}d_thr{thr}pct"
TAG = make_tag()

# ---------------- Utils ----------------
def future_max_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = len(prices)
    out = np.full(n, np.nan)
    for i in range(n):
        j = min(n, i + horizon + 1)
        w = prices[i:j]
        if len(w) < 2:
            continue
        p0, pmin = w[0], w.min()
        out[i] = (pmin - p0) / p0
    return out

def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def zscore(s: pd.Series, win: int) -> pd.Series:
    r = s.rolling(win, min_periods=win)
    return (s - r.mean()) / (r.std() + 1e-12)

# ---------------- Robust loaders ----------------
def _read_indexed_csv(path: Path):
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower() or c.lower()=="timestamp"), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def load_qqq_close(path: Path) -> pd.DataFrame:
    df = _read_indexed_csv(path)
    for cand in ["Adj Close","adj_close","Adj_Close","Close","close","Open","open"]:
        if cand in df.columns:
            out = df[[cand]].rename(columns={cand: "close"})
            return out.asfreq("B").ffill()
    raise ValueError(f"No price column in {path}. Columns={list(df.columns)}")

def load_ig_hy(path: Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(path, index_col=0); d.index = pd.to_datetime(d.index)
    except Exception:
        d = _read_indexed_csv(path)
    d = d.sort_index()
    cols = {c.strip(): c.strip() for c in d.columns}
    d = d.rename(columns=cols)
    if "HY Spread" not in d.columns:
        cand = [c for c in d.columns if "HY" in c and "Spread" in c]
        if not cand: raise ValueError(f"'HY Spread' not found. Columns: {list(d.columns)}")
        d = d.rename(columns={cand[0]:"HY Spread"})
    if "IG Spread" not in d.columns:
        cand = [c for c in d.columns if "IG" in c and "Spread" in c]
        if cand: d = d.rename(columns={cand[0]:"IG Spread"})
        else: d["IG Spread"] = np.nan
    if "HY-IG Spread (bps)" not in d.columns:
        d["HY-IG Spread (bps)"] = (d["HY Spread"] - d["IG Spread"]) * 100.0
    return d.asfreq("B").ffill()[["HY Spread","IG Spread","HY-IG Spread (bps)"]]

def load_fgi(path: Path) -> pd.DataFrame:
    df = _read_indexed_csv(path)
    val_col = next((c for c in ["Value","value","FGI","Index","FearGreed"] if c in df.columns), df.columns[-1])
    out = df[[val_col]].rename(columns={val_col:"FGI"})
    out["FGI"] = pd.to_numeric(out["FGI"], errors="coerce")
    return out.asfreq("B").ffill()

def load_yield_curve(path: Path) -> pd.DataFrame:
    df = _read_indexed_csv(path)
    if "slope" in df.columns:
        out = df[["slope"]].copy()
    elif {"DGS10","DGS2"}.issubset(df.columns):
        out = pd.DataFrame(index=df.index)
        out["slope"] = pd.to_numeric(df["DGS10"], errors="coerce") - pd.to_numeric(df["DGS2"], errors="coerce")
    else:
        raise ValueError(f"yield_curve.csv needs 'slope' or ('DGS10','DGS2'). Columns={list(df.columns)}")
    return out.asfreq("B").ffill()

def load_dxy(path: Path) -> pd.DataFrame:
    df = _read_indexed_csv(path)
    for cand in ["DXY_FRED","DXY","Adj Close","adj_close","Close","close","PX_LAST","Price","price","value"]:
        if cand in df.columns:
            price_col = cand; break
    else:
        raise ValueError(f"No DXY price column. Columns={list(df.columns)}")
    out = df[[price_col]].rename(columns={price_col:"dxy"})
    out["dxy"] = pd.to_numeric(out["dxy"], errors="coerce")
    return out.asfreq("B").ffill()

def load_vix(path: Path) -> pd.DataFrame:
    df = _read_indexed_csv(path)
    for cand in ["VIX","Close","PX_LAST","Adj Close","close","value","Price"]:
        if cand in df.columns:
            col = cand; break
    else:
        raise ValueError(f"No VIX price column. Columns={list(df.columns)}")
    out = df[[col]].rename(columns={col:"vix"})
    out["vix"] = pd.to_numeric(out["vix"], errors="coerce")
    return out.asfreq("B").ffill()

def load_us10y(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next((c for c in df.columns if "date" in c.lower()), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # 支持 "close" 列
    for cand in ["DGS10","US10Y","Close","close","Rate","value","PX_LAST"]:
        if cand in df.columns:
            col = cand
            break
    else:
        raise ValueError(f"No US10Y column. Columns={list(df.columns)}")

    out = df[[col]].rename(columns={col:"us10y"})
    out["us10y"] = pd.to_numeric(out["us10y"], errors="coerce")
    return out.asfreq("B").ffill()
def load_options_features(path: Path) -> pd.DataFrame:
    """
    从期权链构造聚合特征:
      - iv_mean, iv_vw, iv_5p, iv_95p
      - delta_mean
      - call_put_ratio (成交量看涨/看跌)
      - opt_volume_total
    """
    df = pd.read_csv(path)
    # 日期处理
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    if date_col is None:
        raise ValueError("options file needs a 'date' column")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col:"date"})
    
    # 只取主要列
    needed = {"date","type","iv","delta","n"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"options file missing {missing}")
    
    # 按日聚合
    feats = []
    for d, g in df.groupby("date"):
        row = {}
        row["date"] = d
        row["iv_mean"] = np.nanmean(g["iv"])
        row["iv_vw"]   = np.nansum(g["iv"]*g["n"]) / max(1, np.nansum(g["n"]))
        row["iv_5p"]   = np.nanpercentile(g["iv"], 5)
        row["iv_95p"]  = np.nanpercentile(g["iv"], 95)
        row["delta_mean"] = np.nanmean(g["delta"])
        row["opt_volume_total"] = np.nansum(g["n"])
        
        calls = g.loc[g["type"].str.lower()=="call","n"].sum()
        puts  = g.loc[g["type"].str.lower()=="put","n"].sum()
        row["call_put_ratio"] = calls / max(1, puts)
        feats.append(row)
    
    out = pd.DataFrame(feats).set_index("date").sort_index()
    return out.asfreq("B").ffill()
# ---------------- Build dataset ----------------
def build_dataset():
    qqq = load_qqq_close(QQQ_PATH)
    hy  = load_ig_hy(HY_PATH)
    fgi = load_fgi(FG_PATH)
    yc  = load_yield_curve(YC_PATH)
    dxy = load_dxy(DXY_PATH)
    vix = load_vix(VIX_PATH)
    us10y = load_us10y(US10Y_PATH)
    
    # === NEW: options features ===
    OPT_PATH = DATA / "options_with_iv_delta.csv"
    opt_feats = load_options_features(OPT_PATH)

    # join
    df = qqq.join([hy, fgi, yc, dxy, vix, us10y, opt_feats], how="inner").copy()

    # ... 下面保持和你原来一样 ...
    # Price/vol
    df["ret1d"]  = df["close"].pct_change(1)
    df["ret5d"]  = df["close"].pct_change(5)
    df["vol20d"] = df["ret1d"].rolling(20).std()

    # Credit
    df["hy_level_pct"]   = df["HY Spread"]
    df["ig_level_pct"]   = df["IG Spread"]
    df["hyig_diff_bps"]  = df["HY-IG Spread (bps)"]
    df["hy_chg_5d_bps"]  = df["HY Spread"].diff(5) * 100.0
    df["hy_vol_20d_bps"] = (df["HY Spread"].diff().rolling(20).std() * 100.0)

    # FGI
    df["FGI"] = df["FGI"].ffill()
    df["FGI_chg5d"] = df["FGI"].diff(5)

    # DXY
    df["dxy_ret5d"]    = df["dxy"].pct_change(5)
    df["dxy_ma20"]     = df["dxy"].rolling(20).mean()
    df["dxy_ma_gap20"] = df["dxy"]/df["dxy_ma20"] - 1.0

    # VIX
    df["vix_ma20"]     = df["vix"].rolling(20).mean()
    df["vix_ma_gap20"] = df["vix"]/df["vix_ma20"] - 1.0

    # Regime
    df["regime_high_rate"] = (df["us10y"] > REGIME_HIGH_RATE_THRESH).astype(int)
    df["regime_easing"] = ((df["us10y"] - df["us10y"].shift(REGIME_EASING_WINDOW_D)) * 100.0 <= -REGIME_EASING_DROP_BP).astype(int)

    # Interactions
    df["fgi_hyig_inter"] = (100.0 - df["FGI"]) * df["hyig_diff_bps"]
    df["yc_slope_sq"]    = df["slope"]**2
    df["vix_yc_inter"]   = df["vix_ma_gap20"] * df["slope"]
    df["dxy_vix_inter"]  = df["dxy_ma_gap20"] * df["vix_ma_gap20"]

    # Target
    df["fwd_dd"] = future_max_drawdown(df["close"].values, FORWARD_WINDOW_TRAIN)
    df["target"] = (df["fwd_dd"] <= CRASH_THRESH).astype(int)

    # 只丢掉必须的缺失
    return df.dropna(subset=["close","target"]).copy()

# ---------------- Model helpers ----------------
def make_xgb(scale_pos_weight):
    return XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    )

def train_eval_auc(df, features):
    X = df[features].values
    y = df["target"].values
    dates = df.index

    split = int(len(X) * (1 - TEST_RATIO))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    dates_train, dates_test = dates[:split], dates[split:]

    spw = (y_train.tolist().count(0) / max(1, y_train.tolist().count(1)))
    model = make_xgb(spw)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)

    return auc, model, (X_train, X_test, y_train, y_test, dates_train, dates_test)

def perm_importance_on_holdout(model, X_test, y_test, feature_names):
    perm = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc"
    )
    return dict(zip(feature_names, perm.importances_mean))

# ---------------- Visualization helpers ----------------
def _plot_series_barh(series: pd.Series, title: str, outfile: Path):
    s = series.copy().astype(float).sort_values(ascending=True)
    plt.figure(figsize=(9, max(3, 0.45 * len(s))))
    s.plot(kind="barh")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(outfile, dpi=220)
    plt.close()

def xgb_gain_importance_dict(model: XGBClassifier, feature_names: list[str]) -> dict:
    booster = model.get_booster()
    raw = booster.get_score(importance_type="gain")  # {'f0': val, ...}
    idx2name = {f"f{i}": name for i, name in enumerate(feature_names)}
    mapped = {idx2name.get(k, k): v for k, v in raw.items()}
    out = pd.Series(mapped, index=feature_names).fillna(0.0)
    return out.to_dict()

# ---------------- Main pipeline ----------------
if __name__ == "__main__":
    df = build_dataset()

    # === 你的“指定特征集”（基线） ===
    base_features = [
        "vol20d", "hyig_diff_bps", "FGI", "ig_level_pct", "ret5d",
        "dxy_ma_gap20", "vix_ma_gap20", "regime_easing",
        "hy_vol_20d_bps", "yc_slope_sq", "dxy_vix_inter",
    ]
    # 只取数据中存在的列，避免列名兼容性问题
    base_features = [f for f in base_features if f in df.columns]

    # === 候选特征（如不需要可留空） ===
    candidates = ["iv_mean", "iv_vw", "iv_5p", "iv_95p",
"delta_mean", "call_put_ratio", "opt_volume_total"]  # 例如可以添加: ["yc_vol_inter","dxy_vol_inter","hy_mom_mix"]

    print(f"[BASE] features={base_features}")
    auc_base, model_base, pack_base = train_eval_auc(df, base_features)
    X_train_b, X_test_b, y_train_b, y_test_b, dtr, dte = pack_base
    perm_base = perm_importance_on_holdout(model_base, X_test_b, y_test_b, base_features)
    print(f"[BASE] AUC(holdout) = {auc_base:.3f}\n")

    # --- 可视化：基线重要性 ---
    _plot_series_barh(
        pd.Series(perm_base),
        title=f"Permutation Importance (Base) — {TAG}",
        outfile=REPORTS / f"perm_importance_base_{TAG}.png",
    )
    _plot_series_barh(
        pd.Series(xgb_gain_importance_dict(model_base, base_features)),
        title=f"XGB Gain Importance (Base) — {TAG}",
        outfile=REPORTS / f"xgb_gain_base_{TAG}.png",
    )

    # --- 逐个候选评估（若 candidates 为空则跳过） ---
    rows = []
    for cand in candidates:
        feats = base_features + [cand]
        auc_c, model_c, pack_c = train_eval_auc(df, feats)
        X_train_c, X_test_c, y_train_c, y_test_c, _, _ = pack_c
        perm_c = perm_importance_on_holdout(model_c, X_test_c, y_test_c, feats)
        delta = auc_c - auc_base
        imp_delta = perm_c.get(cand, np.nan)

        print(f" - {cand:<16} | AUC={auc_c:.3f}  ΔAUC={delta:+.3f}  perm_imp={imp_delta:+.5f}  "
              f"(train={len(X_train_c)}, test={len(X_test_c)})")

        rows.append({
            "candidate": cand,
            "samples_train": len(X_train_c),
            "samples_test": len(X_test_c),
            "auc_base": auc_base,
            "auc_with": auc_c,
            "delta_auc": delta,
            "perm_importance_mean": imp_delta
        })

    out_df = pd.DataFrame(rows).sort_values(["delta_auc","perm_importance_mean"], ascending=False) if rows else pd.DataFrame(columns=[
        "candidate","samples_train","samples_test","auc_base","auc_with","delta_auc","perm_importance_mean"
    ])
    out_csv = REPORTS / f"interaction_search_{TAG}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv}")

    if not out_df.empty:
        _plot_series_barh(
            out_df.set_index("candidate")["delta_auc"],
            title=f"ΔAUC by Candidate — {TAG}",
            outfile=REPORTS / f"candidate_delta_auc_{TAG}.png",
        )

    # --- 选择 Top-N 候选（若 candidates 为空则 final = base） ---
    top = out_df.head(N_SELECT)["candidate"].tolist() if not out_df.empty else []
    final_feats = base_features + top
    final_csv = REPORTS / f"final_features_{TAG}.csv"
    pd.DataFrame({"final_features": final_feats}).to_csv(final_csv, index=False)
    print("Top candidates:")
    print(out_df.head(N_SELECT).to_string(index=False))
    print(f"\n[FINAL FEATURE SET] -> {final_feats}")
    print(f"Saved -> {final_csv}")

    # --- 训练最终模型并报告 ---
    auc_final, model_final, pack_final = train_eval_auc(df, final_feats)
    X_train_f, X_test_f, y_train_f, y_test_f, dates_train, dates_test = pack_final
    proba_f = model_final.predict_proba(X_test_f)[:,1]
    pred_f  = (proba_f >= 0.5).astype(int)
    print(f"\nFinal AUC(holdout) with top-{max(0,len(top))}: {auc_final:.3f}")
    print("\n=== Final holdout classification report (p=0.5) ===")
    print(classification_report(y_test_f, pred_f, digits=3))

    # --- 最终模型重要性可视化 ---
    perm_final = perm_importance_on_holdout(model_final, X_test_f, y_test_f, final_feats)
    _plot_series_barh(
        pd.Series(perm_final),
        title=f"Permutation Importance (Final) — {TAG}",
        outfile=REPORTS / f"perm_importance_final_{TAG}.png",
    )
    _plot_series_barh(
        pd.Series(xgb_gain_importance_dict(model_final, final_feats)),
        title=f"XGB Gain Importance (Final) — {TAG}",
        outfile=REPORTS / f"xgb_gain_final_{TAG}.png",
    )


