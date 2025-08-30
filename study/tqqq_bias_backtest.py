# -*- coding: utf-8 -*-
"""
QQQ Bias Backtest — Day0全买QQQ；后续入金买TQQQ；DCA基准与策略对比（三连图）
----------------------------------------------------------------
数据：
  data/QQQ_ohlcv_1d.csv  (需含 date + close/adj_close)
  data/TQQQ_ohlcv_1d.csv (需含 date + close/adj_close)
输出：
  output/plot_compare_3panel_equity_TQQQ.png
  output/plot_compare_3panel_buy30_hist_TQQQ.png
（可选）RSI两张图：
  output/plot_rsi_equity_TQQQ.png
  output/plot_rsi_buy30_hist_TQQQ.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================= 参数（可改） =================
DATA_QQQ   = Path("data/QQQ_ohlcv_1d.csv")
DATA_TQQQ  = Path("data/TQQQ_ohlcv_1d.csv")
OUT_DIR    = Path("output")

MA_WINDOW  = 25
THRESHOLDS = [-0.03, -0.07, -0.10]   # 三个乖离阈值（按顺序对应三连图）
INITIAL_CASH = 300_000
DEPOSIT_EVERY_TRADING_DAYS = 63
DEPOSIT_SIZE = 15_000
BUY_RET_DAYS = 30

# RSI（可选）
RSI_WINDOW = 14
RSI_OVERSOLD = 30          # RSI<=30 视为超卖
USE_ADJ_CLOSE_IF_AVAILABLE = True
# =================================================

def _pick_close(df):
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next((c for c in ["date","timestamp","time"] if c in df.columns), None)
    if date_col is None: raise ValueError("缺少 date/timestamp 列")
    if USE_ADJ_CLOSE_IF_AVAILABLE:
        close_col = next((c for c in ["adj_close","adjclose","adjusted_close","adj close"] if c in df.columns), None)
    else:
        close_col = None
    if close_col is None:
        close_col = next((c for c in ["close","close_price"] if c in df.columns), None)
    if close_col is None: raise ValueError("缺少 close/adj_close 列")
    df = df[[date_col, close_col]].dropna().copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col:"date", close_col:"close"}).sort_values("date").drop_duplicates("date")
    return df[["date","close"]].reset_index(drop=True)

def load_prices():
    qqq  = _pick_close(pd.read_csv(DATA_QQQ)).rename(columns={"close":"close_qqq"})
    tqqq = _pick_close(pd.read_csv(DATA_TQQQ)).rename(columns={"close":"close_tqqq"})
    # 对齐日期（交集）
    px = pd.merge(qqq, tqqq, on="date", how="inner").sort_values("date").reset_index(drop=True)
    # 固定位置索引，用于T+30
    px["pos"] = np.arange(len(px))
    return px

def compute_bias_cols(px, window=MA_WINDOW):
    out = px.copy()
    out["ma25"]  = out["close_qqq"].rolling(window, min_periods=window).mean()
    out["bias"]  = (out["close_qqq"] - out["ma25"]) / out["ma25"]
    out["prev_bias"] = out["bias"].shift(1)
    return out

def first_cross_mask_bias(px_bias, thr: float):
    return (px_bias["bias"] <= thr) & (px_bias["prev_bias"] > thr)

def compute_rsi(px, window=RSI_WINDOW):
    out = px.copy()
    delta = out["close_qqq"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100/(1+rs))
    out["prev_rsi"] = out["rsi"].shift(1)
    return out

def first_cross_mask_rsi(px_rsi, thr=RSI_OVERSOLD):
    return (px_rsi["rsi"] <= thr) & (px_rsi["prev_rsi"] > thr)

# ========== 回测（Day0全买QQQ；之后入金与买入均针对TQQQ） ==========
def backtest_dca_TQQQ(px):
    """DCA基准：第0天全买QQQ；之后每63天入金并立即把现金买成TQQQ。"""
    cash = INITIAL_CASH
    sh_qqq = 0
    sh_tqqq = 0
    rows = []
    for i, row in px.iterrows():
        p_q = row["close_qqq"]; p_t = row["close_tqqq"]
        if i == 0:
            qty = int(cash // p_q)
            if qty > 0:
                cash -= qty * p_q
                sh_qqq += qty
        elif i % DEPOSIT_EVERY_TRADING_DAYS == 0:
            cash += DEPOSIT_SIZE
            qty = int(cash // p_t)
            if qty > 0:
                cash -= qty * p_t
                sh_tqqq += qty
        nav = cash + sh_qqq*p_q + sh_tqqq*p_t
        rows.append((row["date"], cash, sh_qqq, sh_tqqq, nav))
    return pd.DataFrame(rows, columns=["date","cash","sh_qqq","sh_tqqq","nav"])

def backtest_signal_TQQQ(px, sig_mask):
    """
    策略：第0天全买QQQ；之后仅在“信号日”把全部现金买成TQQQ（整股）。
    返回：equity曲线 & 每次买入记录（含TQQQ的T+30收益）。不把初始买QQQ计入买入分布。
    """
    cash = INITIAL_CASH
    sh_qqq = 0
    sh_tqqq = 0
    rows = []
    buys = []
    n = len(px)
    for i, row in px.iterrows():
        p_q = row["close_qqq"]; p_t = row["close_tqqq"]
        if i == 0:
            qty = int(cash // p_q)
            if qty > 0:
                cash -= qty * p_q
                sh_qqq += qty
        elif i % DEPOSIT_EVERY_TRADING_DAYS == 0:
            cash += DEPOSIT_SIZE

        # 仅在信号日买TQQQ（同日先入金再买）
        if i > 0 and sig_mask.iloc[i]:
            qty = int(cash // p_t)
            if qty > 0:
                cash -= qty * p_t
                sh_tqqq += qty
                j = i + BUY_RET_DAYS
                if j < n:
                    ret30 = px.iloc[j]["close_tqqq"] / p_t - 1.0
                    buys.append((row["date"], p_t, qty, ret30))

        nav = cash + sh_qqq*p_q + sh_tqqq*p_t
        rows.append((row["date"], cash, sh_qqq, sh_tqqq, nav))
    eq = pd.DataFrame(rows, columns=["date","cash","sh_qqq","sh_tqqq","nav"])
    buys_df = pd.DataFrame(buys, columns=["date","buy_price","qty","ret_30d"])
    return eq, buys_df

# ================= 画图 =================
def plot_compare_3panel_equity_TQQQ(dca_eq, bias_eq_map, thresholds, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, thr in zip(axes, thresholds):
        ax.plot(dca_eq["date"], dca_eq["nav"], label="DCA (deposits -> TQQQ)")
        ax.plot(bias_eq_map[thr]["date"], bias_eq_map[thr]["nav"], label=f"Signal Strategy (Bias ≤ {thr:.0%})")
        ax.set_title(f"Equity Curve — Deposits buy TQQQ | Bias ≤ {thr:.0%}")
        ax.set_ylabel("NAV (USD)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_compare_3panel_buyhist_TQQQ(buys_map, thresholds, out_path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, thr in zip(axes, thresholds):
        bdf = buys_map[thr]
        ax.hist(bdf["ret_30d"].dropna(), bins=30)
        ax.set_title(f"TQQQ 30-Day Return after Each Buy — Bias ≤ {thr:.0%} (N={len(bdf)})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylabel("Count")
    axes[-1].set_xlabel("Return over next 30 trading days")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ========（可选）RSI 两张图：同样Day0买QQQ，后续买TQQQ ========
def plot_rsi_equity_TQQQ(dca_eq, rsi_eq, out_path):
    plt.figure(figsize=(12,6))
    plt.plot(dca_eq["date"], dca_eq["nav"], label="DCA (deposits -> TQQQ)")
    plt.plot(rsi_eq["date"], rsi_eq["nav"], label=f"RSI≤{RSI_OVERSOLD} Strategy (deposits -> TQQQ)")
    plt.title(f"Equity Curve — Deposits buy TQQQ | RSI Oversold (RSI≤{RSI_OVERSOLD})")
    plt.xlabel("Date"); plt.ylabel("NAV (USD)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_rsi_buyhist_TQQQ(rsi_buys, out_path):
    plt.figure(figsize=(12,6))
    plt.hist(rsi_buys["ret_30d"].dropna(), bins=30)
    plt.title(f"TQQQ 30-Day Return after Each Buy — RSI≤{RSI_OVERSOLD} (N={len(rsi_buys)})")
    plt.xlabel("Return over next 30 trading days"); plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# ================= 主流程 =================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    px0     = load_prices()               # date, close_qqq, close_tqqq, pos
    px_bias = compute_bias_cols(px0)      # + ma25, bias, prev_bias
    px_rsi  = compute_rsi(px0)            # + rsi, prev_rsi

    # DCA（Day0买QQQ，后续买TQQQ）
    dca_tqqq = backtest_dca_TQQQ(px0)

    # Bias 三阈值（信号来自QQQ；买入标的=TQQQ）
    bias_eq_map, buys_map = {}, {}
    for thr in THRESHOLDS:
        sig_mask = first_cross_mask_bias(px_bias, thr)
        eq, buys = backtest_signal_TQQQ(px0, sig_mask)
        bias_eq_map[thr] = eq
        buys_map[thr]    = buys

    # 三连图（净值 & 买入T+30分布）
    plot_compare_3panel_equity_TQQQ(dca_tqqq, bias_eq_map, THRESHOLDS,
                                    OUT_DIR / "plot_compare_3panel_equity_TQQQ.png")
    plot_compare_3panel_buyhist_TQQQ(buys_map, THRESHOLDS,
                                     OUT_DIR / "plot_compare_3panel_buy30_hist_TQQQ.png")

    # （可选）RSI 同逻辑
    rsi_sig = first_cross_mask_rsi(px_rsi, RSI_OVERSOLD)
    rsi_eq, rsi_buys = backtest_signal_TQQQ(px0, rsi_sig)
    plot_rsi_equity_TQQQ(dca_tqqq, rsi_eq, OUT_DIR / "plot_rsi_equity_TQQQ.png")
    plot_rsi_buyhist_TQQQ(rsi_buys, OUT_DIR / "plot_rsi_buy30_hist_TQQQ.png")

if __name__ == "__main__":
    main()

