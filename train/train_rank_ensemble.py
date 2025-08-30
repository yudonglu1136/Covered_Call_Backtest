# -*- coding: utf-8 -*-
"""
Feature importance comparison with YOUR original 11 features.
- Data window: 2021-07-26 ~ 2025-08-21
- Split the time range into 4 equal-time partitions
- Train XGB on each partition, plot side-by-side importances
- Train on ALL data once, save global importances (CSV + bar chart)

Run:
  python train/feature_importance_compare.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# ---------------- Config ----------------
RANDOM_STATE = 42
FORWARD_WINDOW = 12
CRASH_THRESH = -0.035

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

# ---------------- Feature set (与原脚本一致的11个) ----------------
BASE_FEATURES = [
    "vol20d","hyig_diff_bps","FGI","ig_level_pct",
    "dxy_ma_gap20","vix_ma_gap20",
    "hy_vol_20d_bps","fgi_hyig_inter","yc_slope_sq",
    "vix_yc_inter","dxy_vix_inter"
]

# ---------------- Data loaders（与你原脚本一致的容错写法） ----------------
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

# ---------------- Feature engineering（与原脚本完全一致） ----------------
def future_max_drawdown(prices: np.ndarray, horizon: int) -> np.ndarray:
    n = len(prices); out = np.full(n, np.nan)
    for i in range(n):
        j = min(n, i + horizon + 1); w = prices[i:j]
        if len(w) < 2: continue
        p0, pmin = w[0], w.min()
        out[i] = (pmin - p0) / p0
    return out

def build_dataset() -> pd.DataFrame:
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

    # --- base features (11个) ---
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

    # Target（与原训练一致）
    df["fwd_dd"] = future_max_drawdown(df["close"].values, FORWARD_WINDOW)
    df["target"] = (df["fwd_dd"] <= CRASH_THRESH).astype(int)

    df = df.dropna().copy()
    return df

# ---------------- Helpers ----------------
def time_quartile_splits(df: pd.DataFrame, start: str, end: str):
    """按“时间等分”而非“行数等分”切四份，保证每份时间跨度尽量一致。"""
    sub = df.loc[start:end].copy()
    if sub.empty:
        raise ValueError("No data in the requested date range.")
    dates = sub.index
    t0, t1 = dates[0], dates[-1]
    # 四等份边界
    qs = [0.25, 0.50, 0.75]
    cuts = [t0] + [t0 + (t1 - t0) * q for q in qs] + [t1]
    # 生成四段（闭区间左、闭区间右都纳入，避免缝隙）
    parts = []
    for i in range(4):
        s = cuts[i]
        e = cuts[i+1]
        part = sub.loc[s:e].copy()
        # 去掉重叠首行（除第一段外）
        if i > 0 and len(part) > 0:
            part = part.iloc[1:].copy()
        parts.append(part)
    return parts

def train_xgb_get_importance(X, y):
    if len(np.unique(y)) < 2:
        return None, np.zeros(X.shape[1])
    model = XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9,
        reg_lambda=1.0, random_state=RANDOM_STATE
    )
    model.fit(X, y)
    return model, model.feature_importances_

"""---------------- Main ----------------
if __name__ == "__main__":
    # 1) 构建数据 + 限制日期
    df = build_dataset()
    df = df.loc["2021-07-26":"2025-08-21"].copy()

    FEATURES = BASE_FEATURES
    print(f"[FEATURES] {len(FEATURES)} -> {FEATURES}")

    # 2) 四等分（按时间）
    splits = time_quartile_splits(df, "2021-07-26", "2025-08-21")

    # 3) 各期训练 & 收集重要性
    importances = []
    labels = []
    for i, chunk in enumerate(splits, 1):
        if chunk.empty:
            print(f"[WARN] Split {i} is empty.")
            imp = np.zeros(len(FEATURES))
        else:
            X = chunk[FEATURES].values
            y = chunk["target"].values
            _, imp = train_xgb_get_importance(X, y)
        importances.append(imp)
        # 期名：用日期范围
        if not chunk.empty:
            labels.append(f"{chunk.index[0].date()} ~ {chunk.index[-1].date()}")
        else:
            labels.append(f"Split{i}")

    importances = np.vstack(importances)

    # 4) 画“分期对比”图（并排柱状）
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(FEATURES))
    width = 0.18
    for i in range(4):
        ax.bar(x + i*width, importances[i], width=width, alpha=0.9, label=f"Q{i+1}\n{labels[i]}")

    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels(FEATURES, rotation=30, ha="right")
    ax.set_ylabel("Feature Importance")
    ax.set_title("XGB Feature Importance — 4 Time Splits (same 11 features)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_cmp = REPORTS / "feature_importance_splits.png"
    fig.savefig(out_cmp, dpi=160)
    plt.close(fig)
    print(f"[Saved] Split comparison -> {out_cmp}")

    # 5) 全数据训练 + 保存 CSV + 单图
    X_all = df[FEATURES].values
    y_all = df["target"].values
    model_all, imp_all = train_xgb_get_importance(X_all, y_all)

    full_csv = REPORTS / "feature_importance_all.csv"
    pd.DataFrame({"feature": FEATURES, "importance": imp_all}).to_csv(full_csv, index=False)
    print(f"[Saved] Full-data feature importance CSV -> {full_csv}")

    # 单独柱状图
    fig, ax = plt.subplots(figsize=(10, 4.8))
    order = np.argsort(imp_all)[::-1]
    ax.bar(np.array(FEATURES)[order], imp_all[order], alpha=0.9)
    ax.set_ylabel("Feature Importance")
    ax.set_title("XGB Feature Importance — Full Data (2021-07-26 ~ 2025-08-21)")
    ax.set_xticklabels(np.array(FEATURES)[order], rotation=30, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out_full = REPORTS / "feature_importance_all.png"
    fig.savefig(out_full, dpi=160)
    plt.close(fig)
    print(f"[Saved] Full-data bar plot -> {out_full}")
    """

    # ---------------- 因子分析扩展 ----------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
import shap
import statsmodels.api as sm

# =============== 1) 单因子检验 IC / RankIC ===============
def single_factor_ic(df, features, target_col="target"):
    results = []
    for f in features:
        x = df[f].values
        y = df[target_col].values
        # Pearson IC
        ic, _ = pearsonr(x, y)
        # RankIC
        ric, _ = spearmanr(x, y)
        results.append({"feature": f, "IC": ic, "RankIC": ric})
    return pd.DataFrame(results)

# =============== 2) 滚动窗口稳定性 ===============
def rolling_ic(df, features, target_col="target", window=250):
    roll_res = {}
    for f in features:
        vals = []
        for i in range(window, len(df)):
            x = df[f].iloc[i-window:i].values
            y = df[target_col].iloc[i-window:i].values
            if len(np.unique(y)) < 2:
                vals.append(np.nan)
                continue
            ic, _ = spearmanr(x, y)
            vals.append(ic)
        roll_res[f] = pd.Series(vals, index=df.index[window:])
    return pd.DataFrame(roll_res)

# =============== 3) 正交化 / 去共线性 (VIF + PCA) ===============
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

def vif_scores(df, features):
    X = df[features].dropna().values
    vif_data = []
    for i in range(X.shape[1]):
        vif = variance_inflation_factor(X, i)
        vif_data.append({"feature": features[i], "VIF": vif})
    return pd.DataFrame(vif_data)

def pca_explained_variance(df, features, n=5):
    X = df[features].dropna().values
    pca = PCA(n_components=n)
    pca.fit(X)
    return pca.explained_variance_ratio_

# =============== 4) 模型对比 (Logistic + SHAP) ===============
def logistic_vs_xgb(df, features, target_col="target"):
    X = df[features].values
    y = df[target_col].values
    logit = LogisticRegression(max_iter=1000).fit(X, y)
    auc_logit = roc_auc_score(y, logit.predict_proba(X)[:,1])

    # SHAP on XGB (用你之前的全量模型)
    explainer = shap.TreeExplainer(model_all)
    shap_vals = explainer.shap_values(X)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    return auc_logit, pd.DataFrame({"feature": features, "mean_abs_SHAP": mean_abs_shap})

# =============== 5) 统计稳健性 (均值/标准差 + bootstrap) ===============
def bootstrap_ic(df, feature, target_col="target", n_boot=500):
    ic_vals = []
    y = df[target_col].values
    x = df[feature].values
    n = len(y)
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        ic, _ = spearmanr(x[idx], y[idx])
        ic_vals.append(ic)
    return np.nanmean(ic_vals), np.nanstd(ic_vals)

# ---------------- 运行分析 ----------------
if __name__ == "__main__":
    # 1) 构建数据 + 限制日期
    df = build_dataset()
    df = df.loc["2021-07-26":"2025-08-21"].copy()

    FEATURES = BASE_FEATURES
    print(f"[FEATURES] {len(FEATURES)} -> {FEATURES}")

    # 2) 全数据训练 (给因子分析用)
    X_all = df[FEATURES].values
    y_all = df["target"].values
    model_all, imp_all = train_xgb_get_importance(X_all, y_all)

    # 3) 保存全量重要性 (可选)
    full_csv = REPORTS / "feature_importance_all.csv"
    pd.DataFrame({"feature": FEATURES, "importance": imp_all}).to_csv(full_csv, index=False)
    print(f"[Saved] Full-data feature importance CSV -> {full_csv}")

    # 4) ---------------- 因子分析扩展 ----------------
    ic_df = single_factor_ic(df, FEATURES)
    ic_df.to_csv(REPORTS/"ic_single_factors.csv", index=False)
    print("[Saved] 单因子IC/RankIC -> ic_single_factors.csv")

    ric_df = rolling_ic(df, FEATURES, window=250)
    ric_df.to_csv(REPORTS/"rolling_ic.csv")
    print("[Saved] Rolling RankIC -> rolling_ic.csv")

    vif_df = vif_scores(df, FEATURES)
    vif_df.to_csv(REPORTS/"vif_scores.csv", index=False)
    print("[Saved] VIF -> vif_scores.csv")

    pca_var = pca_explained_variance(df, FEATURES)
    print("[PCA] 前5主成分方差解释率:", pca_var)

    auc_logit, shap_df = logistic_vs_xgb(df, FEATURES)
    print(f"[Logit] AUC={auc_logit:.3f}")
    shap_df.to_csv(REPORTS/"shap_importance.csv", index=False)
    print("[Saved] SHAP importance -> shap_importance.csv")

    mean_ic, std_ic = bootstrap_ic(df, "FGI", n_boot=300)
    print(f"[Bootstrap FGI] meanIC={mean_ic:.3f}, std={std_ic:.3f}")
    # ===== 在 ric_df 生成之后追加 =====
# ric_df: index=日期, columns=FEATURES, 值=滚动 RankIC（-1..1）

    # 1) 原始热力图
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(ric_df.T.values, aspect="auto", interpolation="nearest",
                   extent=[0, ric_df.shape[0], 0, ric_df.shape[1]])
    # 纵轴特征名
    ax.set_yticks(np.arange(len(FEATURES)) + 0.5)
    ax.set_yticklabels(FEATURES)
    # 横轴日期（稀疏取刻度）
    xticks = np.linspace(0, ric_df.shape[0]-1, 8, dtype=int)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([ric_df.index[i].strftime("%Y-%m-%d") for i in xticks], rotation=30, ha="right")
    ax.set_title("Rolling RankIC (window=250 trading days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RankIC")
    fig.tight_layout()
    out_heat = REPORTS / "rolling_rankic_heatmap.png"
    fig.savefig(out_heat, dpi=160)
    plt.close(fig)
    print(f"[Saved] Rolling RankIC heatmap -> {out_heat}")

    # 2) 平滑版（对每列再做20日滚动均值）
    ric_smooth = ric_df.rolling(20, min_periods=5).mean()
    fig, ax = plt.subplots(figsize=(14, 6))
    im2 = ax.imshow(ric_smooth.T.values, aspect="auto", interpolation="nearest",
                    extent=[0, ric_smooth.shape[0], 0, ric_smooth.shape[1]])
    ax.set_yticks(np.arange(len(FEATURES)) + 0.5)
    ax.set_yticklabels(FEATURES)
    xticks = np.linspace(0, ric_smooth.shape[0]-1, 8, dtype=int)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels([ric_smooth.index[i].strftime("%Y-%m-%d") for i in xticks], rotation=30, ha="right")
    ax.set_title("Rolling RankIC (250-day window) — 20-day smoothed")
    ax.set_xlabel("Date")
    ax.set_ylabel("Feature")
    cbar2 = plt.colorbar(im2, ax=ax)
    cbar2.set_label("RankIC (smoothed)")
    fig.tight_layout()
    out_heat_s = REPORTS / "rolling_rankic_heatmap_smoothed.png"
    fig.savefig(out_heat_s, dpi=160)
    plt.close(fig)
    print(f"[Saved] Smoothed Rolling RankIC heatmap -> {out_heat_s}")


