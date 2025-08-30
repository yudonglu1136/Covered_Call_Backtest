# -*- coding: utf-8 -*-
"""
QQQ Bias Backtest — Initial 300k always fully invested; later deposits follow rules
----------------------------------------------------------------------------------
数据：data/QQQ_ohlcv_1d.csv（需包含 date + close）

规则：
- 初始资金 $300,000 在第 0 天“无脑整股买入”（DCA 与 Bias 策略相同）。
- 每 63 个交易日入金 $15,000，现金累计：
  * DCA 基准：入金当日立即整股买入。
  * Bias 策略：只在“信号日”（bias <= 阈值且首次跌破）才把所有现金整股买入。
- 仅输出两张三连图：
  1) plot_compare_3panel_equity.png：每个面板画一条 Bias 策略净值曲线 + 一条 DCA 基准净值曲线。
  2) plot_compare_3panel_buy30_hist.png：每个面板画 Bias 策略“每次买入后 30 交易日收益”的直方图。

参数：修改脚本顶部变量即可。
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
DATA_PATH   = Path("data/QQQ_ohlcv_1d.csv")
OUT_DIR     = Path("output")
MA_WINDOW   = 25
THRESHOLDS  = [-0.03, -0.07, -0.12]  # 三个阈值（按顺序用于三连图）
INITIAL_CASH = 300_000
DEPOSIT_EVERY_TRADING_DAYS = 63
DEPOSIT_SIZE = 15_000
BUY_RET_DAYS = 30  # 每次买入后观察 30 交易日收益

# ---------------- Helpers ----------------
def load_price(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = next((c for c in ["date","timestamp","time"] if c in df.columns), None)
    if date_col is None:
        raise ValueError("缺少 date/timestamp 列")
    close_col = next((c for c in ["close","adj_close","adjclose","adjusted_close","adj close","close_price"] if c in df.columns), None)
    if close_col is None:
        raise ValueError("缺少 close/adj_close 列")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    return df.rename(columns={date_col:"date", close_col:"close"})[["date","close"]]

def compute_bias(px: pd.DataFrame, window: int) -> pd.DataFrame:
    out = px.copy()
    out["ma25"] = out["close"].rolling(window, min_periods=window).mean()
    out["bias"] = (out["close"] - out["ma25"]) / out["ma25"]
    out["prev_bias"] = out["bias"].shift(1)
    return out

def first_cross_mask(px: pd.DataFrame, thr: float) -> pd.Series:
    return (px["bias"] <= thr) & (px["prev_bias"] > thr)

# ---------------- Backtests ----------------
def backtest_dca(px: pd.DataFrame) -> pd.DataFrame:
    """初始300k第0天全额整股买；之后每63交易日入金立刻整股买。"""
    cash = INITIAL_CASH
    shares = 0
    rows = []
    for i, row in px.iterrows():
        price = row["close"]
        # deposits
        if i == 0:
            # initial buy
            buy_qty = int(cash // price)
            if buy_qty > 0:
                cost = buy_qty * price
                cash -= cost
                shares += buy_qty
        elif i % DEPOSIT_EVERY_TRADING_DAYS == 0:
            cash += DEPOSIT_SIZE
            # immediately buy on deposit day
            buy_qty = int(cash // price)
            if buy_qty > 0:
                cost = buy_qty * price
                cash -= cost
                shares += buy_qty
        nav = cash + shares * price
        rows.append((row["date"], cash, shares, price, nav))
    return pd.DataFrame(rows, columns=["date","cash","shares","close","nav"])

def backtest_bias(px: pd.DataFrame, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """初始300k第0天全额整股买；之后只在“首次跌破”信号日整股买。
       返回：
         - equity曲线 DataFrame
         - 每次买入记录(含后30日收益) DataFrame
    """
    sig_mask = first_cross_mask(px, thr)
    cash = INITIAL_CASH
    shares = 0
    rows = []
    buys = []  # records of (date, price, qty, ret30)

    for i, row in px.iterrows():
        price = row["close"]
        # deposits (accumulate cash)
        if i == 0:
            # initial full buy
            buy_qty = int(cash // price)
            if buy_qty > 0:
                cost = buy_qty * price
                cash -= cost
                shares += buy_qty
                # record initial buy for 30D return if possible
                j = i + BUY_RET_DAYS
                if j < len(px):
                    ret30 = px.loc[j, "close"] / price - 1.0
                    buys.append((row["date"], price, buy_qty, ret30))
        elif i % DEPOSIT_EVERY_TRADING_DAYS == 0:
            cash += DEPOSIT_SIZE

        # only buy on signal days (after deposit if same day)
        if i > 0 and sig_mask.iloc[i]:
            buy_qty = int(cash // price)
            if buy_qty > 0:
                cost = buy_qty * price
                cash -= cost
                shares += buy_qty
                # record buy and its 30D forward return (if available)
                j = i + BUY_RET_DAYS
                if j < len(px):
                    ret30 = px.loc[j, "close"] / price - 1.0
                    buys.append((row["date"], price, buy_qty, ret30))

        nav = cash + shares * price
        rows.append((row["date"], cash, shares, price, nav))

    eq = pd.DataFrame(rows, columns=["date","cash","shares","close","nav"])
    buys_df = pd.DataFrame(buys, columns=["date","buy_price","qty","ret_30d"])
    return eq, buys_df

# ---------------- Plots ----------------
def plot_compare_3panel_equity(px: pd.DataFrame, dca: pd.DataFrame, bias_equities: dict, thresholds: list[float], out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, thr in zip(axes, thresholds):
        ax.plot(dca["date"], dca["nav"], label="DCA Benchmark")
        ax.plot(bias_equities[thr]["date"], bias_equities[thr]["nav"], label=f"Bias ≤ {thr:.0%}")
        ax.set_title(f"Equity Curve — Bias ≤ {thr:.0%}")
        ax.set_ylabel("NAV (USD)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_compare_3panel_buyhist(buys_map: dict, thresholds: list[float], out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for ax, thr in zip(axes, thresholds):
        bdf = buys_map[thr]
        ax.hist(bdf["ret_30d"].dropna(), bins=30)
        ax.set_title(f"30-Day Return after Each Buy — Bias ≤ {thr:.0%} (N={len(bdf)})")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.set_ylabel("Count")
    axes[-1].set_xlabel("Return over next 30 trading days")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ---------------- Main ----------------


# ================= RSI Oversold Strategy (add-on) =================
RSI_WINDOW = 14
RSI_OVERSOLD = 20  # RSI <= 30 considered oversold
def compute_rsi(px: pd.DataFrame, window: int = RSI_WINDOW) -> pd.DataFrame:
    out = px.copy()
    # Compute simple RSI
    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi"] = 100 - (100 / (1 + rs))
    out["prev_rsi"] = out["rsi"].shift(1)
    return out

def rsi_oversold_first_cross_mask(rsi_df: pd.DataFrame, thr: float = RSI_OVERSOLD) -> pd.Series:
    return (rsi_df["rsi"] <= thr) & (rsi_df["prev_rsi"] > thr)

def backtest_signal_strategy(px: pd.DataFrame, sig_mask: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generic: initial 300k buy fully; deposits add cash; ONLY buy on sig_mask days."""
    cash = INITIAL_CASH
    shares = 0
    rows, buys = [], []
    for i, row in px.iterrows():
        price = row["close"]
        # initial buy at day 0
        if i == 0:
            qty = int(cash // price)
            if qty > 0:
                cash -= qty * price
                shares += qty
                j = i + BUY_RET_DAYS
                if j < len(px):
                    ret30 = px.iloc[j]["close"] / price - 1.0
                    buys.append((row["date"], price, qty, ret30))
        elif i % DEPOSIT_EVERY_TRADING_DAYS == 0:
            cash += DEPOSIT_SIZE

        if i > 0 and sig_mask.iloc[i]:
            qty = int(cash // price)
            if qty > 0:
                cash -= qty * price
                shares += qty
                j = i + BUY_RET_DAYS
                if j < len(px):
                    ret30 = px.iloc[j]["close"] / price - 1.0
                    buys.append((row["date"], price, qty, ret30))

        nav = cash + shares * price
        rows.append((row["date"], cash, shares, price, nav))
    eq = pd.DataFrame(rows, columns=["date","cash","shares","close","nav"])
    buys_df = pd.DataFrame(buys, columns=["date","buy_price","qty","ret_30d"])
    return eq, buys_df

def plot_rsi_equity(dca: pd.DataFrame, rsi_eq: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(12,6))
    plt.plot(dca["date"], dca["nav"], label="DCA Benchmark")
    plt.plot(rsi_eq["date"], rsi_eq["nav"], label=f"RSI≤{RSI_OVERSOLD} Strategy")
    plt.title(f"Equity Curve — RSI Oversold (RSI≤{RSI_OVERSOLD})")
    plt.xlabel("Date"); plt.ylabel("NAV (USD)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_rsi_buyhist(rsi_buys: pd.DataFrame, out_path: Path):
    plt.figure(figsize=(12,6))
    plt.hist(rsi_buys["ret_30d"].dropna(), bins=30)
    plt.title(f"30-Day Return after Each Buy — RSI≤{RSI_OVERSOLD} (N={len(rsi_buys)})")
    plt.xlabel("Return over next 30 trading days"); plt.ylabel("Count")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

# Hook into main()
def _main_with_rsi():
    # Reuse main()'s setup
    px = load_price(DATA_PATH)
    # compute bias (for bias strategies)
    px_bias = compute_bias(px, MA_WINDOW)

    # DCA baseline on original prices
    dca_eq = backtest_dca(px)

    # Bias strategies
    bias_eq_map, buys_map = {}, {}
    for thr in THRESHOLDS:
        eq, buys = backtest_bias(px_bias, thr)
        bias_eq_map[thr] = eq
        buys_map[thr] = buys

    # Save bias tri-panels
    plot_compare_3panel_equity(px_bias, dca_eq, bias_eq_map, THRESHOLDS, OUT_DIR / "plot_compare_3panel_equity.png")
    plot_compare_3panel_buyhist(buys_map, THRESHOLDS, OUT_DIR / "plot_compare_3panel_buy30_hist.png")

    # RSI strategy
    rsi_df = compute_rsi(px)
    rsi_sig_mask = rsi_oversold_first_cross_mask(rsi_df, RSI_OVERSOLD)
    rsi_eq, rsi_buys = backtest_signal_strategy(px, rsi_sig_mask)

    # RSI figures
    plot_rsi_equity(dca_eq, rsi_eq, OUT_DIR / "plot_rsi_equity.png")
    plot_rsi_buyhist(rsi_buys, OUT_DIR / "plot_rsi_buy30_hist.png")

# Replace old main() by calling our new flow
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _main_with_rsi()
if __name__ == "__main__":
    main()

