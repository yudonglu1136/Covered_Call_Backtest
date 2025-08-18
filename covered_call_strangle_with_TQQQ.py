# -*- coding: utf-8 -*-
"""
Covered Strangle Backtest (QQQ) — CALLs + cash-secured PUTs, with TQQQ buys on extreme fear
Additions in this version:
  - Output directory: output/covered_call_strangle_TQQQ
  - Extra chart for TQQQ buy/sell events (scatter markers) saved alongside main chart.
  - HTML report embeds both figures.

Core rules (unchanged from your previous request):
  - CALLs: 15–18 DTE, IV->delta targeting with a 106% strike floor; do NOT sell a new CALL on an assignment day.
  - PUTs: single cash-secured ATM PUT (1–3 DTE); if expires OTM -> re-sell; if assigned -> acquire QQQ.
  - TQQQ: when Fear & Greed (FG) < 15, deploy ALL *free* cash (cash minus reserved PUT collateral) to buy integer shares;
           track batches; for each batch, if price >= 2x entry (>= +100%), sell that batch (take profit).
  - DCA benchmark: buy-and-hold QQQ in 100-share lots when cash allows.

Inputs under ./data/ :
  - options_with_iv_delta.csv
  - QQQ_ohlcv_1d.csv
  - TQQQ_ohlcv_1d.csv
  - QQQ_dividends.csv
  - Fear_and_greed.csv   # columns: Date, Value (0~100)
"""

from pathlib import Path
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# If you run on a headless server, uncomment and remove plt.show():
# import matplotlib
# matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent      # .../data_gen
ROOT = HERE.parent                          # project root (sibling of data_gen and src)
sys.path.insert(0, str(ROOT))               # make `src` importable as a top-level package

# Now import the shared utilities
from src.quant_utils import (
    normalize_date_series,
    iv_to_delta,
    price_on_or_before,
    shares_affordable_for_put,
    sharpe_ratio,
)
# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/covered_call_strangle_TQQQ")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63   # quarterly contribution (approx.)
DCA_AMOUNT = 15_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00          # set nonzero if desired (e.g., 0.03)

FG_BUY_THRESHOLD = 15            # FG < 15 triggers TQQQ buy
PUT_MIN_DTE = 1
PUT_MAX_DTE = 3
CALL_MIN_DTE = 15
CALL_MAX_DTE = 18
DELTA_BAND = 0.01
CALL_STRIKE_FLOOR_PCT = 1.055     # 106% of spot


# ============================
# Load Data
# ============================
df_options = pd.read_csv("data/options_with_iv_delta.csv")
qqq_raw    = pd.read_csv("data/QQQ_ohlcv_1d.csv")   # expects: date + open
tqqq_raw   = pd.read_csv("data/TQQQ_ohlcv_1d.csv")  # expects: date + open
div_df     = pd.read_csv("data/QQQ_dividends.csv")
fg_raw     = pd.read_csv("data/Fear_and_greed.csv") # columns: Date, Value

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

# Standardize price columns
def prep_price_df(raw: pd.DataFrame) -> pd.DataFrame:
    if "date" not in raw.columns:
        raise KeyError(f"'date' column not found: {list(raw.columns)}")
    if "Open" not in raw.columns:
        if "open" in raw.columns:
            raw = raw.rename(columns={"open": "Open"})
        else:
            for alt in ["Close","close","Adj Close","adj_close","adjClose"]:
                if alt in raw.columns:
                    raw = raw.rename(columns={alt: "Open"})
                    break
    if "Open" not in raw.columns:
        raise KeyError(f"No 'Open' or 'open'-like column found: {list(raw.columns)}")
    raw["date"] = normalize_date_series(raw["date"])
    raw["Open"] = pd.to_numeric(raw["Open"], errors="coerce")
    return raw[["date","Open"]].dropna()

qqq_df  = prep_price_df(qqq_raw)
tqqq_df = prep_price_df(tqqq_raw)

# Keep only the option dates universe
opt_dates = pd.Index(df_options["date"].dropna().unique())
qqq_df  = qqq_df[qqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
tqqq_df = tqqq_df[tqqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
dates = qqq_df.index

# Fear & Greed series aligned to dates (forward-fill)
fg_raw = fg_raw.rename(columns={"Date":"date","date":"date","Value":"value","value":"value"})
if "date" not in fg_raw.columns or "value" not in fg_raw.columns:
    raise KeyError(f"Fear_and_greed.csv must have columns like ['Date','Value']; got: {list(fg_raw.columns)}")
fg_raw["date"] = normalize_date_series(fg_raw["date"])
fg_raw["value"] = pd.to_numeric(fg_raw["value"], errors="coerce")
fg_series = fg_raw.set_index("date")["value"].sort_index().reindex(dates).ffill()

# ============================
# State Variables
# ============================
cash = float(INITIAL_CASH)
shares = 0                        # QQQ shares
reserved_put_cash = 0.0           # reserve cash to back the active PUT (strike*100*contracts)
total_premium = 0.0

# Active options
active_calls = []                 # list of dicts: strike, expiration, contracts, type="call", exercised
active_put   = None               # single dict or None: strike, expiration, contracts, type="put", exercised

# TQQQ holdings tracked by batches + trade events for plotting
tqqq_batches = []                 # list of dicts: {"qty": int, "entry": float, "date": Timestamp}
tqqq_realized_pnl = 0.0
tqqq_buy_events = []              # list of (Timestamp, price, qty)
tqqq_sell_events = []             # list of (Timestamp, price, qty, pnl)

# DCA benchmark
cash_bh = float(INITIAL_CASH)
shares_bh = 0

# Logs (English)
log_lines = []

# Equity trackers
portfolio_value = []
buy_hold_value = []

# ============================
# Main Loop
# ============================
for i, current_date in enumerate(dates):
    qqq_price  = float(qqq_df.loc[current_date, "Open"])
    tqqq_price = float(tqqq_df.loc[current_date, "Open"])
    df_today = df_options[df_options["date"] == current_date]
    assignment_today = False  # block CALL sale today if any assignment happened

    # 1) Quarterly contribution
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash balance (Strategy): ${cash:,.2f}")

    # 2) Dividends on QQQ
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty and shares > 0:
        dividend_per_share = float(div_row["dividend"].values[0])
        credited = dividend_per_share * shares
        cash += credited
        log_lines.append(f"[{current_date.date()}] 📦 Dividend credited +${credited:,.2f} (${dividend_per_share:.4f}/share on {shares} shares)")

    # 3) Handle expirations / assignments
    # 3a) CALLs
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(qqq_df, opt["expiration"], qqq_price)
            contracts = int(opt["contracts"])
            if price_at_exp > float(opt["strike"]):
                # Called away
                cash += float(opt["strike"]) * contracts * 100
                shares -= contracts * 100
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ⚠️ CALL assigned: -{contracts*100} QQQ shares @ ${float(opt['strike']):.2f}")
            opt["exercised"] = True

    # 3b) PUT (cash-secured)
    if active_put is not None and (current_date >= active_put["expiration"]) and (not active_put["exercised"]):
        price_at_exp = price_on_or_before(qqq_df, active_put["expiration"], qqq_price)
        contracts = int(active_put["contracts"])
        strike = float(active_put["strike"])
        cost = strike * contracts * 100

        if price_at_exp < strike:
            # Assignment: buy QQQ via PUT assignment (consumes reserved cash)
            if cost <= cash:
                shares += contracts * 100
                cash -= cost
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} QQQ shares @ ${strike:.2f}")
            else:
                log_lines.append(f"[{current_date.date()}] ⚠️ PUT assignment failed (insufficient cash). Needed ${cost:,.2f}, have ${cash:,.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] ⏳ PUT expired worthless @ strike ${strike:.2f}")

        # Release reserved cash in either case
        reserved_put_cash = 0.0
        active_put["exercised"] = True
        active_put = None

    # 4) TQQQ actions driven by Fear & Greed (FG)
    fg_val = fg_series.loc[current_date] if current_date in fg_series.index else np.nan

    # 4a) If FG < 15, deploy ALL *free* cash into TQQQ integer shares
    #     free cash = cash - reserved_put_cash (never touch the reserved amount)
    if pd.notna(fg_val) and fg_val < FG_BUY_THRESHOLD and tqqq_price > 0:
        free_cash = max(0.0, cash - reserved_put_cash)
        qty = int(free_cash // tqqq_price)  # integer shares (not 100-lots)
        if qty > 0:
            cost = qty * tqqq_price
            cash -= cost
            tqqq_batches.append({"qty": qty, "entry": tqqq_price, "date": current_date})
            tqqq_buy_events.append((current_date, tqqq_price, qty))
            log_lines.append(
                f"[{current_date.date()}] 🛒 TQQQ buy (FG={fg_val:.0f}) {qty} shares @ ${tqqq_price:.2f}; "
                f"cash left: ${cash:,.2f} (reserved: ${reserved_put_cash:,.2f})"
            )

    # 4b) Check TQQQ batches for +100% take-profit
    if tqqq_batches:
        kept_batches = []
        for b in tqqq_batches:
            entry = float(b["entry"])
            qty   = int(b["qty"])
            if tqqq_price >= 2.0 * entry and qty > 0:
                proceeds = qty * tqqq_price
                pnl = qty * (tqqq_price - entry)
                cash += proceeds
                tqqq_realized_pnl += pnl
                tqqq_sell_events.append((current_date, tqqq_price, qty, pnl))
                log_lines.append(
                    f"[{current_date.date()}] ✅ TQQQ take-profit: sell {qty} @ ${tqqq_price:.2f} "
                    f"(entry ${entry:.2f}, +{(tqqq_price/entry-1):.0%}), P&L +${pnl:,.2f}"
                )
            else:
                kept_batches.append(b)
        tqqq_batches = kept_batches

    # 5) Sell a new cash-secured PUT (ATM, 1–3 DTE) *after* TQQQ actions
    has_active_put = active_put is not None
    if (not has_active_put) and (not df_today.empty):
        df_puts = df_today[
            (df_today["type"].str.lower() == "put") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=PUT_MIN_DTE)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=PUT_MAX_DTE))
        ].copy()
        if not df_puts.empty:
            df_puts["strike"] = pd.to_numeric(df_puts["strike"], errors="coerce")
            df_puts = df_puts.dropna(subset=["strike"])
            # ATM = minimize |strike - spot|
            df_puts["atm_gap"] = (df_puts["strike"] - qqq_price).abs()
            put_opt = df_puts.sort_values(["atm_gap","expiration"]).iloc[0]
            put_strike = float(put_opt["strike"])

            # contracts sized by *free* cash (enforce cash-backed)
            free_cash = max(0.0, cash - reserved_put_cash)  # should be 0 if a PUT exists; here no active PUT
            max_contracts = shares_affordable_for_put(free_cash, put_strike)
            if max_contracts > 0:
                premium = float(put_opt["vw"]) * 100 * max_contracts
                cash += premium
                total_premium += premium
                reserved_put_cash = put_strike * 100 * max_contracts  # reserve full notional
                active_put = {
                    "strike": put_strike,
                    "expiration": pd.to_datetime(put_opt["expiration"]),
                    "contracts": int(max_contracts),
                    "type": "put",
                    "exercised": False,
                }
                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold cash-secured PUT +${premium:,.2f} "
                    f"@ strike ${put_strike:.2f}, expiry {pd.to_datetime(put_opt['expiration']).date()}, "
                    f"contracts {max_contracts}; reserved ${reserved_put_cash:,.2f}"
                )

    # 6) Covered CALL (15–18 DTE) with 106% strike floor — only if not assigned today
    has_active_call = any((o["type"] == "call") and (not o["exercised"]) for o in active_calls)
    if (not assignment_today) and (not has_active_call) and (not df_today.empty) and shares >= 100:
        df_calls = df_today[
            (df_today["type"].str.lower() == "call") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=CALL_MIN_DTE)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=CALL_MAX_DTE))
        ].copy()
        if not df_calls.empty:
            df_calls["strike"] = pd.to_numeric(df_calls["strike"], errors="coerce")
            df_calls["delta"]  = pd.to_numeric(df_calls["delta"], errors="coerce")
            df_calls = df_calls.dropna(subset=["strike","delta"])

            iv_today = float(df_calls["iv"].mean())
            target_delta = iv_to_delta(iv_today)
            strike_floor = round(CALL_STRIKE_FLOOR_PCT * qqq_price, 2)

            # (1) delta filter + strike >= floor
            candidates = df_calls[
                (df_calls["delta"] <= target_delta + DELTA_BAND) &
                (df_calls["strike"] >= strike_floor)
            ].copy()
            # (2) relax delta but keep strike floor
            if candidates.empty:
                candidates = df_calls[df_calls["strike"] >= strike_floor].copy()
            # (3) if still none, fall back to delta filter
            if candidates.empty:
                candidates = df_calls[df_calls["delta"] <= target_delta + DELTA_BAND].copy()

            if not candidates.empty:
                candidates["delta_gap"] = (candidates["delta"] - target_delta).abs()
                if (candidates["strike"] >= strike_floor).any():
                    candidates = candidates.sort_values(by=["strike","delta_gap"], ascending=[True,True])
                else:
                    candidates = candidates.sort_values(by=["delta_gap","strike"], ascending=[True,True])

                option = candidates.iloc[0]
                chosen_strike = float(option["strike"])
                chosen_delta  = float(option["delta"])
                contracts = int(shares // 100)
                premium = float(option["vw"]) * contracts * 100
                cash += premium
                total_premium += premium

                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold CALL +${premium:,.2f} "
                    f"@ strike ${chosen_strike:.2f} (floor ${strike_floor:.2f}), Δ={chosen_delta:.3f}, "
                    f"expiry {pd.to_datetime(option['expiration']).date()}, contracts {contracts}"
                )

                active_calls.append({
                    "strike": chosen_strike,
                    "expiration": pd.to_datetime(option["expiration"]),
                    "contracts": int(contracts),
                    "type": "call",
                    "exercised": False,
                })

    # 7) DCA benchmark: always uses 100-share lots when possible
    if cash_bh >= qqq_price * 100:
        can_buy_bh = int(cash_bh // qqq_price) // 100 * 100
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * qqq_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 8) Track equity (QQQ + cash + TQQQ)
    tqqq_position_qty = sum(int(b["qty"]) for b in tqqq_batches) if tqqq_batches else 0
    equity_strategy = shares * qqq_price + cash + tqqq_position_qty * tqqq_price
    portfolio_value.append(equity_strategy)
    buy_hold_value.append(shares_bh * qqq_price + cash_bh)

# ============================
# Statistics & Output
# ============================
# Equity curves
curve_cc = pd.Series(portfolio_value, index=dates, name="Strategy")
curve_bh = pd.Series(buy_hold_value, index=dates, name="DCA")

final_value_cc = float(curve_cc.iloc[-1])
final_value_bh = float(curve_bh.iloc[-1])

# Total invested capital = initial + contributions (for both strategies)
contrib_cnt = (len(dates) - 1) // DCA_INTERVAL_TRADING_DAYS
total_invested = INITIAL_CASH + contrib_cnt * DCA_AMOUNT

years = (dates[-1] - dates[0]).days / 365.0 if len(dates) > 1 else 0.0
cagr_cc = (final_value_cc / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
cagr_bh = (final_value_bh / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
excess = final_value_cc - final_value_bh
excess_per_year = excess / years if years > 0 else 0.0

# Sharpe ratios
sharpe_cc = sharpe_ratio(curve_cc, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)
sharpe_bh = sharpe_ratio(curve_bh, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)

# TQQQ position snapshot
tqqq_qty_now = sum(int(b["qty"]) for b in tqqq_batches) if tqqq_batches else 0

summary = f"""
Backtest Summary (Covered Strangle + TQQQ on FG<15)
----------------------------
Final equity (Strategy):    ${final_value_cc:,.2f}
Final equity (DCA):         ${final_value_bh:,.2f}
Total invested capital:     ${total_invested:,.2f}

Total return (Strategy):    {(final_value_cc / total_invested - 1.0):.2%}
Total return (DCA):         {(final_value_bh / total_invested - 1.0):.2%}
CAGR (Strategy):            {cagr_cc:.2%}
CAGR (DCA):                 {cagr_bh:.2%}

Sharpe (Strategy):          {sharpe_cc:.2f}
Sharpe (DCA):               {sharpe_bh:.2f}

Option premium collected:   ${total_premium:,.2f}
TQQQ realized P&L:          ${tqqq_realized_pnl:,.2f}
TQQQ open position:         {tqqq_qty_now} shares

Excess over DCA:            ${excess:,.2f}
Excess per year:            ${excess_per_year:,.2f}
----------------------------
"""
print(summary)

# Write logs & artifacts
log_path = OUTPUT_DIR / "covered_call_strangle.log"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n" + summary)

# Equity curves CSV
eq_csv = OUTPUT_DIR / "equity_curves.csv"
pd.DataFrame({"date": dates, "strategy_value": curve_cc.values, "dca_value": curve_bh.values}).to_csv(eq_csv, index=False)

# TQQQ trades CSV
tqqq_trades_csv = OUTPUT_DIR / "tqqq_trades.csv"
tqqq_trades_rows = []
for d, p, q in tqqq_buy_events:
    tqqq_trades_rows.append({"date": d, "side": "BUY", "price": p, "qty": q, "pnl": np.nan})
for d, p, q, pnl in tqqq_sell_events:
    tqqq_trades_rows.append({"date": d, "side": "SELL", "price": p, "qty": q, "pnl": pnl})
pd.DataFrame(tqqq_trades_rows).to_csv(tqqq_trades_csv, index=False)

# Save a plain-text summary
with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

# ============================
# Plots (save first, then show)
# ============================
ts_label = dates[-1].date()
main_png = OUTPUT_DIR / f"strategy_comparison_{ts_label}.png"
main_pdf = OUTPUT_DIR / f"strategy_comparison_{ts_label}.pdf"
tqqq_png = OUTPUT_DIR / f"tqqq_trades_{ts_label}.png"
tqqq_pdf = OUTPUT_DIR / f"tqqq_trades_{ts_label}.pdf"
html_path = OUTPUT_DIR / "report.html"

# 1) Main equity curve figure
plt.figure(figsize=(12, 6))
plt.plot(dates, curve_cc.values, label="Strategy: Covered Strangle + TQQQ(FG<15)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("📊 Strategy Comparison: Strangle + TQQQ (FG trigger) vs DCA")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(main_png, dpi=150, bbox_inches="tight")
plt.savefig(main_pdf, bbox_inches="tight")
plt.close()

# 2) TQQQ trades figure (price line + buy/sell scatter markers)
plt.figure(figsize=(12, 6))
plt.plot(dates, tqqq_df.loc[dates, "Open"].values, label="TQQQ Open")
if tqqq_buy_events:
    buy_dates = [d for d, _, _ in tqqq_buy_events]
    buy_prices = [p for _, p, _ in tqqq_buy_events]
    plt.scatter(buy_dates, buy_prices, marker="^", s=64, label="TQQQ BUY")
if tqqq_sell_events:
    sell_dates = [d for d, _, _, _ in tqqq_sell_events]
    sell_prices = [p for _, p, _, _ in tqqq_sell_events]
    plt.scatter(sell_dates, sell_prices, marker="v", s=64, label="TQQQ SELL")
plt.title("📈 TQQQ Trades (FG<15 buys, 100% take-profit sells)")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(tqqq_png, dpi=150, bbox_inches="tight")
plt.savefig(tqqq_pdf, bbox_inches="tight")
plt.close()

# HTML report with both figures
html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial; max-width: 900px; margin: 40px auto;">
  <h1>Strategy Comparison: Strangle + TQQQ (FG trigger) vs DCA</h1>
  <pre style="background:#f6f8fa; padding:16px; border-radius:8px;">{summary}</pre>

  <h2>Main Equity Curves</h2>
  <figure>
    <img src="{main_png.name}" alt="Strategy Comparison" style="max-width:100%; height:auto;">
    <figcaption>Saved figure: {main_png.name}</figcaption>
  </figure>

  <h2>TQQQ Buy/Sell Events</h2>
  <figure>
    <img src="{tqqq_png.name}" alt="TQQQ Trades" style="max-width:100%; height:auto;">
    <figcaption>Saved figure: {tqqq_png.name}</figcaption>
  </figure>

  <p>Equity curves CSV: {eq_csv.name}</p>
  <p>TQQQ trades CSV: {tqqq_trades_csv.name}</p>
  <p>Log file: {log_path.name}</p>
</body>
</html>"""
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Figures saved to: {main_png}, {main_pdf}, {tqqq_png}, {tqqq_pdf}")
print(f"HTML report saved to: {html_path}")
plt.show()  # On headless servers, comment this out and use Agg backend

