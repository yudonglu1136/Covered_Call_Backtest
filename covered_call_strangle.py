# -*- coding: utf-8 -*-
"""
Covered Strangle Backtest (QQQ) — Covered CALLs + Cash-Secured PUTs (CSP) for (re)building shares
Key rules vs. your "calls-only" baseline:
  - Do NOT directly buy shares with cash.
  - To add shares or rebuild after assignment, SELL an ATM cash-secured PUT (1–3 DTE).
  - Only ONE active PUT at a time (cash-backed). If it expires OTM, immediately re-sell another.
  - If PUT is assigned, shares are acquired via assignment (no market buy).
  - Keep your CALL logic: 15–18 DTE, delta targeting via IV->delta, with a 106% strike floor.
  - Do NOT sell a new CALL on the same day when any assignment (CALL or PUT) occurs.

All comments/logs/prints are in English.
Outputs are saved under output/covered_call_strangle/ before showing the figure.
Includes Sharpe ratios for Strategy and DCA benchmark.

Data files expected under ./data/ :
  - options_with_iv_delta.csv   # columns: date, expiration, type(call/put), strike, vw, iv, delta
  - QQQ_ohlcv_1d.csv            # columns: date, open, high, low, close, ...
  - QQQ_dividends.csv           # columns: date, dividend
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# If you run on a headless server, uncomment the next two lines and remove plt.show():
# import matplotlib
# matplotlib.use("Agg")

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/covered_call_strangle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63  # quarterly contribution in trading days (approx.)
DCA_AMOUNT = 15_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00  # set to a nonzero annual risk-free rate if desired (e.g., 0.03)

# ============================
# Helpers
# ============================
def normalize_date_series(s: pd.Series) -> pd.Series:
    """Return a Timestamp-normalized (00:00) date series."""
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def iv_to_delta(iv: float, steepness: float = 10, mid: float = 0.27,
                min_delta: float = 0.018, max_delta: float = 0.15) -> float:
    """Map IV to a target call delta via a logistic curve."""
    norm = 1.0 / (1.0 + math.exp(-steepness * (mid - float(iv))))
    delta_val = min_delta + (max_delta - min_delta) * norm
    return round(float(delta_val), 4)

def price_on_or_before(idx_price: pd.DataFrame, ts: pd.Timestamp, current_fallback: float) -> float:
    """Get 'Open' price on ts if available; otherwise closest prior trading day's price.
    Fallback to first available or current_fallback if necessary."""
    if ts in idx_price.index:
        return float(idx_price.loc[ts, "Open"])
    earlier = idx_price.index[idx_price.index <= ts]
    if len(earlier) > 0:
        return float(idx_price.loc[earlier[-1], "Open"])
    return float(current_fallback)

def shares_affordable(cash: float, strike: float) -> int:
    """Max 100-share contracts buyable at assignment (cash-secured)."""
    if strike <= 0:
        return 0
    contracts = int(cash // (strike * 100))
    return int(contracts)

def sharpe_ratio(equity_curve: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    """Compute annualized Sharpe ratio from a daily equity curve."""
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return 0.0
    rf_per_period = rf_annual / periods_per_year
    excess = rets - rf_per_period
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))

# ============================
# Load Data
# ============================
df_options = pd.read_csv("data/options_with_iv_delta.csv")
price_raw  = pd.read_csv("data/QQQ_ohlcv_1d.csv")  # expects: date, open, high, low, close, ...
div_df     = pd.read_csv("data/QQQ_dividends.csv")

# Normalize options/dividend dates
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

# Normalize price df: ensure 'date' & 'Open'
if "date" not in price_raw.columns:
    raise KeyError(f"'date' column not found in QQQ_ohlcv_1d.csv. Columns: {list(price_raw.columns)}")
if "Open" not in price_raw.columns:
    if "open" in price_raw.columns:
        price_raw = price_raw.rename(columns={"open": "Open"})
    else:
        # Fallback to 'close' if 'Open' truly missing; comment out if strict
        for alt in ["Close", "close", "Adj Close", "adj_close", "adjClose"]:
            if alt in price_raw.columns:
                price_raw = price_raw.rename(columns={alt: "Open"})
                break
if "Open" not in price_raw.columns:
    raise KeyError(f"No 'Open' or 'open' column in QQQ_ohlcv_1d.csv. Columns: {list(price_raw.columns)}")

price_raw["date"] = normalize_date_series(price_raw["date"])
price_raw["Open"] = pd.to_numeric(price_raw["Open"], errors="coerce")

# Build clean price_df and keep only option dates
price_df = price_raw[["date", "Open"]].dropna(subset=["date", "Open"]).copy()
opt_dates = pd.Index(df_options["date"].dropna().unique())
price_df = price_df[price_df["date"].isin(opt_dates)].sort_values("date").set_index("date")

dates = price_df.index

# ============================
# State Variables
# ============================
cash = float(INITIAL_CASH)
shares = 0

# Active options
active_calls = []  # dicts: strike, expiration, contracts, type="call", exercised
active_put   = None  # single dict or None: strike, expiration, contracts, type="put", exercised

total_premium = 0.0
contribution_count = 0

# DCA benchmark (unchanged)
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
    current_price = float(price_df.loc[current_date, "Open"])
    df_today = df_options[df_options["date"] == current_date]

    assignment_today = False  # block new CALLs on assignment day

    # 1) Quarterly contribution (to both strategies)
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        contribution_count += 1
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash balance (Strategy): ${cash:,.2f}")

    # 2) Dividends (paid on strategy shares)
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty and shares > 0:
        dividend_per_share = float(div_row["dividend"].values[0])
        credited = dividend_per_share * shares
        cash += credited
        log_lines.append(f"[{current_date.date()}] 📦 Dividend credited +${credited:,.2f} (${dividend_per_share:.4f}/share on {shares} shares)")

    # 3) Handle option expirations / assignments
    # 3a) CALLs
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(price_df, opt["expiration"], current_price)
            contracts = int(opt["contracts"])
            if price_at_exp > float(opt["strike"]):
                # CALL assignment
                cash += float(opt["strike"]) * contracts * 100
                shares -= contracts * 100
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ⚠️ CALL assigned: -{contracts*100} shares @ ${float(opt['strike']):.2f}")
            opt["exercised"] = True

    # 3b) Single PUT (cash-secured)
    if active_put is not None and (current_date >= active_put["expiration"]) and (not active_put["exercised"]):
        price_at_exp = price_on_or_before(price_df, active_put["expiration"], current_price)
        contracts = int(active_put["contracts"])
        strike = float(active_put["strike"])
        if price_at_exp < strike:
            # PUT assignment -> buy shares via assignment
            cost = strike * contracts * 100
            # should always be affordable because we sized at sale; still guard
            if cost <= cash:
                shares += contracts * 100
                cash -= cost
                assignment_today = True  # block CALL sale today
                log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} shares @ ${strike:.2f}")
            else:
                log_lines.append(f"[{current_date.date()}] ⚠️ PUT assignment skipped (insufficient cash), needed ${cost:,.2f}, have ${cash:,.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] ⏳ PUT expired worthless at strike ${strike:.2f}")
        active_put["exercised"] = True
        active_put = None  # free to sell another CSP below

    # 4) SELL a new cash-secured PUT (ATM, 1–3 DTE) if:
    #    - We want to build/rebuild shares (i.e., have cash and either we don't own enough or just prefer CSP route)
    #    - No active PUT currently
    #    - Have enough cash to secure at least 1 contract at ATM strike
    has_active_put = active_put is not None
    if (not has_active_put) and (not df_today.empty):
        # Candidate PUTs: 1–3 DTE, ATM by closest |strike - spot|
        df_puts = df_today[
            (df_today["type"].str.lower() == "put") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=1)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=3))
        ].copy()
        if not df_puts.empty:
            df_puts["strike"] = pd.to_numeric(df_puts["strike"], errors="coerce")
            df_puts = df_puts.dropna(subset=["strike"])
            # pick ATM strike
            df_puts["atm_gap"] = (df_puts["strike"] - current_price).abs()
            put_opt = df_puts.sort_values(["atm_gap", "expiration"]).iloc[0]
            put_strike = float(put_opt["strike"])
            # contracts sized by strike (cash-secured)
            max_contracts = shares_affordable(cash, put_strike)
            if max_contracts > 0:
                premium = float(put_opt["vw"]) * 100 * max_contracts
                cash += premium
                total_premium += premium
                active_put = {
                    "strike": put_strike,
                    "expiration": pd.to_datetime(put_opt["expiration"]),
                    "contracts": int(max_contracts),
                    "type": "put",
                    "exercised": False,
                }
                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold cash-secured PUT +${premium:,.2f} "
                    f"@ strike ${put_strike:.2f}, expiry {pd.to_datetime(put_opt['expiration']).date()}, contracts {max_contracts}"
                )

    # 5) Covered CALL (15–18 DTE) with 106% strike floor — only if not assigned today
    has_active_call = any((o["type"] == "call") and (not o["exercised"]) for o in active_calls)
    if (not assignment_today) and (not has_active_call) and (not df_today.empty) and shares >= 100:
        df_calls = df_today[
            (df_today["type"].str.lower() == "call") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=15)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=18))
        ].copy()
        if not df_calls.empty:
            # ensure numeric
            df_calls["strike"] = pd.to_numeric(df_calls["strike"], errors="coerce")
            df_calls["delta"]  = pd.to_numeric(df_calls["delta"], errors="coerce")
            df_calls = df_calls.dropna(subset=["strike", "delta"])

            iv_today = float(df_calls["iv"].mean())
            target_delta = iv_to_delta(iv_today)
            strike_floor = round(1.06 * current_price, 2)  # 106% of spot

            # (1) delta filter + strike >= floor
            candidates = df_calls[
                (df_calls["delta"] <= target_delta + 0.01) &
                (df_calls["strike"] >= strike_floor)
            ].copy()
            # (2) relax delta but keep strike floor
            if candidates.empty:
                candidates = df_calls[df_calls["strike"] >= strike_floor].copy()
            # (3) if still none, fall back to delta filter
            if candidates.empty:
                candidates = df_calls[df_calls["delta"] <= target_delta + 0.01].copy()

            if not candidates.empty:
                candidates["delta_gap"] = (candidates["delta"] - target_delta).abs()
                if (candidates["strike"] >= strike_floor).any():
                    candidates = candidates.sort_values(by=["strike", "delta_gap"], ascending=[True, True])
                else:
                    candidates = candidates.sort_values(by=["delta_gap", "strike"], ascending=[True, True])

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

    # 6) DCA benchmark buys 100-share lots whenever cash allows (unchanged)
    if cash_bh >= current_price * 100:
        can_buy_bh = int(cash_bh // current_price) // 100 * 100
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * current_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 7) Track equity (after all actions of the day)
    portfolio_value.append(shares * current_price + cash)
    buy_hold_value.append(shares_bh * current_price + cash_bh)

# ============================
# Statistics & Output
# ============================
# Total invested capital = initial + quarterly contributions
total_invested = INITIAL_CASH + contribution_count * DCA_AMOUNT

# Equity curves
curve_cc = pd.Series(portfolio_value, index=dates, name="Strategy")
curve_bh = pd.Series(buy_hold_value, index=dates, name="DCA")

final_value_cc = float(curve_cc.iloc[-1])
final_value_bh = float(curve_bh.iloc[-1])

years = (dates[-1] - dates[0]).days / 365.0
cagr_cc = (final_value_cc / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
cagr_bh = (final_value_bh / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0

excess = final_value_cc - final_value_bh
excess_per_year = excess / years if years > 0 else 0.0

# Sharpe ratios (annualized)
sharpe_cc = sharpe_ratio(curve_cc, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)
sharpe_bh = sharpe_ratio(curve_bh, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)

# Text summary (English)
summary = f"""
Backtest Summary (Covered Strangle: CALLs + cash-secured PUTs)
----------------------------
Option premium collected:   ${total_premium:,.2f}
Final equity (Strategy):    ${final_value_cc:,.2f}
Final equity (DCA):         ${final_value_bh:,.2f}
Total invested capital:     ${total_invested:,.2f}

Total return (Strategy):    {(final_value_cc / total_invested - 1.0):.2%}
Total return (DCA):         {(final_value_bh / total_invested - 1.0):.2%}
CAGR (Strategy):            {cagr_cc:.2%}
CAGR (DCA):                 {cagr_bh:.2%}

Sharpe (Strategy):          {sharpe_cc:.2f}
Sharpe (DCA):               {sharpe_bh:.2f}

Excess over DCA:            ${excess:,.2f}
Excess per year:            ${excess_per_year:,.2f}
----------------------------
"""
print(summary)

# Append summary to logs (English)
log_path = OUTPUT_DIR / "covered_call_strangle.log"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n" + summary)

# Save equity curves to CSV for further analysis
eq_csv = OUTPUT_DIR / "equity_curves.csv"
pd.DataFrame({"date": dates, "strategy_value": curve_cc.values, "dca_value": curve_bh.values}).to_csv(eq_csv, index=False)

# Save a plain-text summary
with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

# ============================
# Plot (save first, then show)
# ============================
ts_label = dates[-1].date()
png_path = OUTPUT_DIR / f"strategy_comparison_{ts_label}.png"
pdf_path = OUTPUT_DIR / f"strategy_comparison_{ts_label}.pdf"
html_path = OUTPUT_DIR / "report.html"

plt.figure(figsize=(12, 6))
plt.plot(dates, curve_cc.values, label="Covered Strangle Strategy (CALLs + CSP)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("Strategy Comparison: Covered Strangle vs DCA")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save to disk BEFORE showing
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

# Simple HTML report
html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Backtest Report</title></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Arial; max-width: 900px; margin: 40px auto;">
  <h1>Strategy Comparison: Covered Strangle vs DCA</h1>
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
plt.show()  # On headless servers, comment this out and use Agg backend
