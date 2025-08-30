# -*- coding: utf-8 -*-
"""
Covered Call Backtest (QQQ) — Calls only, with:
  - Early take-profit (>=60%) using option_ticker daily prices
  - T+2 repurchase after assignment with cash lock
  - NEW: 60-day-high-aware selling rule:
      * If current price is within 5% of last 60 trading days' high -> use 7% floor
      * If drawdown >5% and <=8% -> use 5% floor
      * If drawdown >8% -> skip selling today
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from datetime import datetime
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.quant_utils import (
    normalize_date_series,
    iv_to_delta, price_on_or_before,
    sharpe_ratio, shares_affordable, deploy_cash_into_shares
)

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/covered_call")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63   # ~ quarterly
DCA_AMOUNT = 0
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00

# Gap filters
SELL_ONLY_ON_GAP_DOWN = False
SELL_ONLY_ON_GAP_UP   = False

# DTE window
DTE_MIN, DTE_MAX = 28, 31

# Strike floor (base switch; dynamic value decided per day below)
USE_STRIKE_FLOOR = True

# Early close rule
EARLY_CLOSE_MIN_DAYS    = 20    # days left > 15
EARLY_CLOSE_TAKE_PROFIT = 0.80  # >=60% profit

# 60-day-high dynamic floor rule
LOOKBACK_HIGH_N        = 100
NEAR_HIGH_THRESHOLD    = 0.07   # within 5% of 60d high -> conservative 7% floor
DEEP_DROP_THRESHOLD    = 0.30   # drop >8% from 60d high -> do NOT sell
FLOOR_NEAR_HIGH        = 0.07
FLOOR_DEFAULT          = 0.07

# ============================
# Load Data
# ============================
DATA_DIR = Path("data")

df_options = pd.read_csv(DATA_DIR / "options_with_iv_delta.csv")
price_raw  = pd.read_csv(DATA_DIR / "QQQ_ohlcv_1d.csv")    # date, Open/open, close
div_df     = pd.read_csv(DATA_DIR / "QQQ_dividends.csv")   # date, dividend

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

# Price prep + gap label
if "date" not in price_raw.columns:
    raise KeyError("price csv must contain a 'date' column")
if "Open" not in price_raw.columns:
    if "open" in price_raw.columns:
        price_raw = price_raw.rename(columns={"open": "Open"})
    else:
        raise KeyError("price csv must contain 'Open' (or 'open') column")
if "close" not in price_raw.columns:
    raise KeyError("QQQ_ohlcv_1d.csv must contain 'close'")

price_raw["date"]  = normalize_date_series(price_raw["date"])
price_raw["Open"]  = pd.to_numeric(price_raw["Open"], errors="coerce")
price_raw = price_raw.sort_values("date").set_index("date")

price_raw["prev_close"] = price_raw["close"].shift(1)
price_raw["gap_label"] = pd.NA
price_raw.loc[price_raw["Open"] > price_raw["prev_close"], "gap_label"] = "Gap Up"
price_raw.loc[price_raw["Open"] < price_raw["prev_close"], "gap_label"] = "Gap Down"

# ⬅️ NEW: rolling 60d high (use close)
price_raw["rolling_high_60"] = price_raw["close"].rolling(window=LOOKBACK_HIGH_N, min_periods=1).max()

# Keep only dates with option quotes
opt_dates = pd.Index(df_options["date"].unique())
price_df  = price_raw.loc[price_raw.index.isin(opt_dates), ["Open","gap_label","rolling_high_60"]].copy()

print(f"[INFO] price_df rows kept: {len(price_df):,} | first: {price_df.index.min().date()} | last: {price_df.index.max().date()}")

dates = price_df.index

# ============== Option daily price index (option_ticker + date) ==============
use_cols = []
for c in ["option_ticker", "date", "vw", "c"]:
    if c in df_options.columns:
        use_cols.append(c)
missing = set(["option_ticker","date"]).difference(use_cols)
if missing:
    raise KeyError(f"options_with_iv_delta.csv missing columns: {missing}")

opt_px = df_options[use_cols].set_index(["option_ticker","date"]).sort_index()

def get_option_price(option_ticker: str, date: pd.Timestamp, last_known: float | None = None) -> float | None:
    """Get option price for ticker+date, prefer vw then c, else fallback to last_known."""
    try:
        row = opt_px.loc[(option_ticker, date)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        price = row["vw"] if ("vw" in row and pd.notna(row["vw"])) else (row["c"] if "c" in row else np.nan)
        if pd.isna(price):
            return last_known
        return float(price)
    except KeyError:
        return last_known

# ============================
# State & Stats
# ============================
cash = float(INITIAL_CASH)
shares = 0
active_calls = []  # dict(strike, expiration, contracts, type, exercised, option_ticker, sold_price, open_date, last_price)
total_premium = 0.0
contribution_count = 0

cash_bh = float(INITIAL_CASH)
shares_bh = 0

log_lines = []
portfolio_value = []
buy_hold_value = []

assignment_count = 0
total_assignment_loss = 0.0
uncovered_days = 0
held_100plus_days = 0
floor_enforced_count = 0

# New stats
profit_take_count = 0
total_buyback_cost = 0.0
floor7_used_count = 0
floor5_used_count = 0
deep_drop_skip_count = 0

# T+2 repurchase after assignment
repurchase_schedule = []          # list[pd.Timestamp]
cash_locks = []                   # list[{"unlock_date": pd.Timestamp, "amount": float}]

# ============================
# Helpers
# ============================
def rule_allows_today(gap_label) -> bool:
    if pd.isna(gap_label):
        return not (SELL_ONLY_ON_GAP_UP or SELL_ONLY_ON_GAP_DOWN)
    label = str(gap_label)
    if SELL_ONLY_ON_GAP_UP and SELL_ONLY_ON_GAP_DOWN:
        return (label == "Gap Up") or (label == "Gap Down")
    if SELL_ONLY_ON_GAP_UP:
        return label == "Gap Up"
    if SELL_ONLY_ON_GAP_DOWN:
        return label == "Gap Down"
    return True

# ============================
# Main Loop
# ============================
for i, current_date in enumerate(dates):
    row_today = price_df.loc[current_date]
    current_price = float(row_today["Open"])
    gap_today = row_today["gap_label"]

    # 60d high logic (use price_raw to ensure full continuity)
    rolling_high = float(price_raw.loc[:current_date, "rolling_high_60"].iloc[-1])
    drawdown = (rolling_high - current_price) / rolling_high if rolling_high > 0 else 0.0

    df_today = df_options[df_options["date"] == current_date]
    assignment_today = False

    # 1) contributions (0 here)
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        contribution_count += 1
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash (Strategy): ${cash:,.2f}")

    # 2) dividends
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty:
        dividend_per_share = float(div_row["dividend"].values[0])
        if shares > 0:
            credited = dividend_per_share * shares
            cash += credited
            log_lines.append(f"[{current_date.date()}] 📦 Dividend +${credited:,.2f} (${dividend_per_share:.4f}/sh × {shares})")
        if shares_bh > 0:
            cash_bh += shares_bh * dividend_per_share

    # 3) expirations & assignments
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(price_df, opt["expiration"], current_price)
            contracts = int(opt["contracts"])
            if price_at_exp > float(opt["strike"]):
                proceeds = float(opt["strike"]) * contracts * 100
                cash += proceeds
                shares -= contracts * 100
                assignment_today = True
                assignment_count += 1
                total_assignment_loss += (price_at_exp - float(opt["strike"])) * contracts * 100
                log_lines.append(f"[{current_date.date()}] ⚠️ Assigned: -{contracts*100} sh @ ${float(opt['strike']):.2f}")

                # T+2 repurchase & lock
                if i + 2 < len(dates):
                    buyback_day = dates[i + 2]
                else:
                    buyback_day = dates[-1]
                repurchase_schedule.append(buyback_day)
                cash_locks.append({"unlock_date": buyback_day, "amount": proceeds})
                log_lines.append(f"[{current_date.date()}] 🗓️ T+2 repurchase on {buyback_day.date()}, locked ${proceeds:,.2f}")
            opt["exercised"] = True

    # 3.3) Early close check
    for opt in active_calls:
        if (opt["type"] == "call") and (not opt["exercised"]) and (not opt.get("closed", False)):
            days_left = (opt["expiration"] - current_date).days
            if days_left > EARLY_CLOSE_MIN_DAYS:
                curr = get_option_price(opt["option_ticker"], current_date, last_known=opt.get("last_price"))
                if curr is not None:
                    opt["last_price"] = curr
                    threshold = opt["sold_price"] * (1.0 - EARLY_CLOSE_TAKE_PROFIT)  # 40% of sold price
                    if curr <= threshold:
                        buyback_cost = curr * opt["contracts"] * 100
                        cash -= buyback_cost
                        total_buyback_cost += buyback_cost
                        profit_take_count += 1
                        opt["closed"] = True
                        log_lines.append(
                            f"[{current_date.date()}] ✅ Early close (TP {EARLY_CLOSE_TAKE_PROFIT:.0%}): "
                            f"buyback ${curr:.2f} vs sold ${opt['sold_price']:.2f}, contracts {opt['contracts']}, days_left {days_left}"
                        )

    # remove closed options from active list (so we can sell new one today)
    active_calls = [o for o in active_calls if not o.get("closed", False)]

    # 3.5) T+2 repurchase day
    if repurchase_schedule and any(d == current_date for d in repurchase_schedule):
        unlock_total = sum(lk["amount"] for lk in cash_locks if lk["unlock_date"] == current_date)
        if unlock_total > 0:
            log_lines.append(f"[{current_date.date()}] 🔓 Unlock cash ${unlock_total:,.2f} for repurchase.")
        cash_locks = [lk for lk in cash_locks if lk["unlock_date"] != current_date]

        cash, shares, bought = deploy_cash_into_shares(cash, shares, current_price)
        if bought > 0:
            log_lines.append(f"[{current_date.date()}] 🔁 T+2 repurchase: +{bought} sh @ ${current_price:.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] 🔁 T+2 repurchase attempted but 0 bought.")

        repurchase_schedule = [d for d in repurchase_schedule if d != current_date]

    # 4) Idle cash deployment (skip on assignment day; exclude locked cash)
    skip_idle_buy_today = assignment_today
    locked_amt = sum(lk["amount"] for lk in cash_locks if current_date < lk["unlock_date"])
    if not skip_idle_buy_today:
        avail_cash = cash - locked_amt
        if avail_cash >= current_price:
            new_cash, shares, bought = deploy_cash_into_shares(avail_cash, shares, current_price)
            if bought > 0:
                cash = new_cash + locked_amt
                log_lines.append(f"[{current_date.date()}] 🛒 Idle buy {bought} sh @ ${current_price:.2f} | locked ${locked_amt:,.2f}")
    else:
        log_lines.append(f"[{current_date.date()}] ⏸️ Skip buying due to same-day assignment; T+2 scheduled.")

    # 5) SELL covered call?
    has_active_call_now = any((o["type"] == "call") and (not o["exercised"]) for o in active_calls)
    allow_sell_today = (not assignment_today) and (not has_active_call_now) and shares >= 100

    if allow_sell_today and (not df_today.empty) and rule_allows_today(gap_today):
        # === NEW: 60d-high aware switch ===
        dynamic_floor_pct = None
        if drawdown > DEEP_DROP_THRESHOLD:
            # Deep drop: skip selling
            deep_drop_skip_count += 1
            log_lines.append(
                f"[{current_date.date()}] ⏸️ Skip selling: price {current_price:.2f} is "
                f"{drawdown:.1%} below 60d high ({rolling_high:.2f}) > {DEEP_DROP_THRESHOLD:.0%}."
            )
        else:
            if drawdown <= NEAR_HIGH_THRESHOLD:
                dynamic_floor_pct = FLOOR_NEAR_HIGH   # conservative 7%
                floor7_used_count += 1
            else:
                dynamic_floor_pct = FLOOR_DEFAULT     # default 5%
                floor5_used_count += 1

            # candidate calls in DTE window
            df_calls = df_today[
                (df_today["type"].str.lower() == "call") &
                (df_today["expiration"] >= current_date + pd.Timedelta(days=DTE_MIN)) &
                (df_today["expiration"] <= current_date + pd.Timedelta(days=DTE_MAX))
            ].copy()

            if not df_calls.empty:
                if "option_ticker" not in df_calls.columns:
                    raise KeyError("options_with_iv_delta.csv must include 'option_ticker'.")

                iv_today = float(df_calls["iv"].mean())
                target_delta = iv_to_delta(iv_today)

                subset = df_calls[df_calls["delta"] <= target_delta + 0.01].copy()
                chosen = None
                reason = "delta-match"

                if not subset.empty:
                    subset["delta_gap"] = (subset["delta"] - target_delta).abs()
                    chosen = subset.sort_values("delta_gap").iloc[0]

                # Apply dynamic floor
                if USE_STRIKE_FLOOR and dynamic_floor_pct is not None:
                    floor_strike = current_price * (1.0 + dynamic_floor_pct)
                    if (chosen is None) or (float(chosen["strike"]) < floor_strike):
                        floor_cands = df_calls[df_calls["strike"] >= floor_strike].copy()
                        if not floor_cands.empty:
                            floor_cands["strike_gap"] = (floor_cands["strike"] - floor_strike).abs()
                            chosen = floor_cands.sort_values("strike_gap").iloc[0]
                            reason = f"floor-enforced({dynamic_floor_pct:.0%})"
                            floor_enforced_count += 1
                        else:
                            log_lines.append(
                                f"[{current_date.date()}] ⏸️ No CC sold: no contract meets dynamic floor "
                                f"(floor {dynamic_floor_pct:.0%}, need ≥ {floor_strike:.2f})."
                            )
                            chosen = None

                if chosen is not None:
                    contracts = int(shares // 100)
                    sold_price = float(chosen["vw"]) if pd.notna(chosen["vw"]) else (float(chosen["c"]) if "c" in chosen else 0.0)
                    premium = sold_price * contracts * 100
                    cash += premium
                    total_premium += premium
                    why = "Gap-Up day" if gap_today == "Gap Up" else ("Gap-Down day" if gap_today=="Gap Down" else "rule off")
                    more = f", dyn_floor={dynamic_floor_pct:.0%}" if (USE_STRIKE_FLOOR and dynamic_floor_pct is not None) else ""
                    log_lines.append(
                        f"[{current_date.date()}] 💰 Sold CALL +${premium:,.2f} "
                        f"@ strike ${float(chosen['strike']):.2f}, Δ={float(chosen['delta']):.3f}, "
                        f"expiry {pd.to_datetime(chosen['expiration']).date()}, contracts {contracts} "
                        f"(reason: {why}, pick={reason}{more}; 60dHi={rolling_high:.2f}, dd={drawdown:.1%})"
                    )
                    active_calls.append({
                        "strike": float(chosen["strike"]),
                        "expiration": pd.to_datetime(chosen["expiration"]),
                        "contracts": int(contracts),
                        "type": "call",
                        "exercised": False,
                        "option_ticker": str(chosen["option_ticker"]),
                        "sold_price": float(sold_price),
                        "open_date": current_date,
                        "last_price": float(sold_price),
                    })
    elif allow_sell_today and (not df_today.empty) and not rule_allows_today(gap_today):
        note = "not Gap-Up" if SELL_ONLY_ON_GAP_UP else "not Gap-Down"
        log_lines.append(f"[{current_date.date()}] ⏸️ Skip selling: {note} day.")

    # 6) DCA benchmark
    if cash_bh >= current_price * 100:
        can_buy_bh = shares_affordable(cash_bh, current_price)
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * current_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 7) Track equity + uncovered days
    portfolio_value.append(shares * current_price + cash)
    buy_hold_value.append(shares_bh * current_price + cash_bh)

    has_active_call_end = any((o["type"] == "call") and (not o["exercised"]) for o in active_calls)
    if shares >= 100:
        held_100plus_days += 1
        if not has_active_call_end:
            uncovered_days += 1

# ============================
# Statistics & Output
# ============================
total_invested = INITIAL_CASH + contribution_count * DCA_AMOUNT
curve_cc = pd.Series(portfolio_value, index=dates, name="Strategy")
curve_bh = pd.Series(buy_hold_value, index=dates, name="DCA")

final_value_cc = float(curve_cc.iloc[-1])
final_value_bh = float(curve_bh.iloc[-1])

years = (dates[-1] - dates[0]).days / 365.0
cagr_cc = (final_value_cc / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
cagr_bh = (final_value_bh / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0

excess = final_value_cc - final_value_bh
excess_per_year = excess / years if years > 0 else 0.0

sharpe_cc = sharpe_ratio(curve_cc, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)
sharpe_bh = sharpe_ratio(curve_bh, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)

period_text = f"{dates.min().date()} → {dates.max().date()}"
uncovered_ratio = (uncovered_days / held_100plus_days) if held_100plus_days > 0 else 0.0

net_premium_after_costs = total_premium - total_buyback_cost - total_assignment_loss

summary = f"""
Backtest Summary (Calls only)
Rules:
  - Early close: days_left > {EARLY_CLOSE_MIN_DAYS} & profit ≥ {EARLY_CLOSE_TAKE_PROFIT:.0%}
  - Dynamic floor by 60d high:
      within 5% -> floor 7%
      5%~8% drop -> floor 5%
      >8% drop -> skip sell
  - T+2 repurchase after assignment (cash locked until T+2)
Period: {period_text}
--------------------------------------------------
Option premium collected:    ${total_premium:,.2f}
Buyback cost (early close):  ${total_buyback_cost:,.2f}
Assignment count:            {assignment_count}
Total assignment loss:       ${total_assignment_loss:,.2f}
Net premium after costs:     ${net_premium_after_costs:,.2f}
Early take-profit count:     {profit_take_count}
Floor used 7% days:          {floor7_used_count}
Floor used 5% days:          {floor5_used_count}
Skip (deep drop >8%) days:   {deep_drop_skip_count}

Days ≥100 sh w/o CC:         {uncovered_days}  (out of {held_100plus_days}, {uncovered_ratio:.1%})
Final equity (Strategy):     ${final_value_cc:,.2f}
Final equity (DCA):          ${final_value_bh:,.2f}
Total invested capital:      ${total_invested:,.2f}

Total return (Strategy):     {(final_value_cc / total_invested - 1.0):.2%}
Total return (DCA):          {(final_value_bh / total_invested - 1.0):.2%}
CAGR (Strategy):             {cagr_cc:.2%}
CAGR (DCA):                  {cagr_bh:.2%}

Sharpe (Strategy):           {sharpe_cc:.2f}
Sharpe (DCA):                {sharpe_bh:.2f}

Excess over DCA:             ${excess:,.2f}
Excess per year:             ${excess_per_year:,.2f}
--------------------------------------------------
"""
print(summary)

# Outputs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
log_path = OUTPUT_DIR / "strategy_covered_call.log"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n" + summary)

eq_csv = OUTPUT_DIR / "equity_curves.csv"
pd.DataFrame({"date": dates, "strategy_value": curve_cc.values, "dca_value": curve_bh.values}).to_csv(eq_csv, index=False)

with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

ts_label = dates[-1].date()
png_path = OUTPUT_DIR / f"strategy_comparison_{ts_label}.png"
pdf_path = OUTPUT_DIR / f"strategy_comparison_{ts_label}.pdf"
html_path = OUTPUT_DIR / "report.html"

plt.figure(figsize=(12, 6))
plt.plot(dates, curve_cc.values, label="Covered Call Strategy (early TP + T+2 + 60dHi floor)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("Strategy Comparison: Covered Call vs DCA")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial; max-width: 900px; margin: 40px auto;">
  <h1>Strategy Comparison: Covered Call vs DCA</h1>
  <pre style="background:#f6f8fa; padding:16px; border-radius:8px;">{summary}</pre>
  <figure>
    <img src="{png_path.name}" alt="Strategy Comparison" style="max-width:100%; height:auto;">
    <figcaption>Saved figure: {png_path.name}</figcaption>
  </figure>
  <p>Equity curves CSV: {eq_csv.name}</p>
  <p>Log file: {log_path.name}</p>
</body>
</html>"""
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Figure saved to: {png_path} and {pdf_path}")
print(f"HTML report saved to: {html_path}")
plt.show()
