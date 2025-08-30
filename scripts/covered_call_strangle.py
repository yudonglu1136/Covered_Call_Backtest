# -*- coding: utf-8 -*-
"""
Covered Call Backtest (QQQ)
Variant: CC baseline + CASH DEPLOYMENT via CSP (ATM 1–3 DTE)
AND "rebuy-via-PUT" after CALL assignment, with loss computed as:
  loss = (rebuy_price) - (assigned CALL strike), realized only when PUT is assigned.

Changes from previous:
  - Removed MTM loss on CALL assignment day.
  - Added realized, reacquisition-based loss when the tagged PUT is assigned.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.quant_utils import (
    normalize_date_series,
    iv_to_delta, price_on_or_before, shares_affordable_for_put,
    sharpe_ratio, shares_affordable
)

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/covered_call")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63   # ~ quarterly
DCA_AMOUNT = 15_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00

# ---- CALL leg ----
SELL_ONLY_ON_GAP_DOWN = False
SELL_ONLY_ON_GAP_UP   = False
DTE_MIN, DTE_MAX = 28, 31

USE_STRIKE_FLOOR = True
STRIKE_FLOOR_PCT = 0.07  # floor = 7% OTM

# ---- CSP window (1–3 DTE) ----
PUT_DTE_MIN, PUT_DTE_MAX = 1, 3

# ============================
# Load Data
# ============================
DATA_DIR = Path("data")
df_options = pd.read_csv(DATA_DIR / "options_with_iv_delta.csv")
price_raw  = pd.read_csv(DATA_DIR / "QQQ_ohlcv_1d.csv")
div_df     = pd.read_csv(DATA_DIR / "QQQ_dividends.csv")

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

# Prepare price & gap label
if "date" not in price_raw.columns:
    raise KeyError("price csv must contain a 'date' column")
if "Open" not in price_raw.columns:
    if "open" in price_raw.columns:
        price_raw = price_raw.rename(columns={"open": "Open"})
    else:
        raise KeyError("price csv must contain 'Open' (or 'open') column")
if "close" not in price_raw.columns:
    raise KeyError("QQQ_ohlcv_1d.csv must contain 'close' for Gap calc")

price_raw["date"]  = normalize_date_series(price_raw["date"])
price_raw["Open"]  = pd.to_numeric(price_raw["Open"], errors="coerce")
price_raw = price_raw.sort_values("date").set_index("date")
price_raw["prev_close"] = price_raw["close"].shift(1)
price_raw["gap_label"] = pd.NA
price_raw.loc[price_raw["Open"] > price_raw["prev_close"], "gap_label"] = "Gap Up"
price_raw.loc[price_raw["Open"] < price_raw["prev_close"], "gap_label"] = "Gap Down"

opt_dates = pd.Index(df_options["date"].unique())
price_df  = price_raw.loc[price_raw.index.isin(opt_dates), ["Open","gap_label"]].copy()

print(f"[INFO] price_df rows kept: {len(price_df):,} | first: {price_df.index.min().date()} | last: {price_df.index.max().date()}")

dates = price_df.index

# ============================
# State & Stats
# ============================
cash = float(INITIAL_CASH)
shares = 0
active_calls = []
active_puts  = []   # each: {strike, expiration, contracts, type='put', exercised, rebuy_ref_strike: float|None}

reserved_collateral = 0.0

# Premium buckets
total_premium = 0.0
call_premium_collected = 0.0
put_premium_collected  = 0.0

# Contributions
contribution_count = 0

# Benchmark (DCA)
cash_bh = float(INITIAL_CASH)
shares_bh = 0

# Logs & curves
log_lines = []
portfolio_value = []
buy_hold_value = []

# Stats
call_assignment_count = 0
put_assignment_count  = 0

# >>> Realized loss based on reacquisition price <<<
total_call_assignment_loss = 0.0  # accumulates only when a tagged PUT is assigned

uncovered_days = 0
held_100plus_days = 0
floor_enforced_count = 0

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

def choose_put_for_assignment(df_today_puts, current_date, target_strike):
    """Pick 1–3 DTE PUT: prefer exact strike==target_strike; else nearest, earliest expiry."""
    cands = df_today_puts[
        (df_today_puts["type"].str.lower() == "put") &
        (df_today_puts["expiration"] >= current_date + pd.Timedelta(days=PUT_DTE_MIN)) &
        (df_today_puts["expiration"] <= current_date + pd.Timedelta(days=PUT_DTE_MAX))
    ].copy()
    if cands.empty:
        return None
    exact = cands[np.isclose(cands["strike"].astype(float), float(target_strike), atol=1e-6)]
    if not exact.empty:
        return exact.sort_values("expiration").iloc[0]
    cands["gap"] = (cands["strike"].astype(float) - float(target_strike)).abs()
    return cands.sort_values(["gap","expiration"]).iloc[0]

# ============================
# Main Loop
# ============================
for i, current_date in enumerate(dates):
    row_today = price_df.loc[current_date]
    current_price = float(row_today["Open"])
    gap_today = row_today["gap_label"]
    df_today = df_options[df_options["date"] == current_date]
    assignment_today = False  # block selling new CALL the same day

    # 1) Contributions
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        contribution_count += 1
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash=${cash:,.2f}")

    # 2) Dividends
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty:
        dps = float(div_row["dividend"].values[0])
        if shares > 0:
            credited = dps * shares
            cash += credited
            log_lines.append(f"[{current_date.date()}] 📦 Dividend +${credited:,.2f} (${dps:.4f}/sh x {shares})")
        if shares_bh > 0:
            cash_bh += dps * shares_bh

    # 3) Expirations — CALL first
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(price_df, opt["expiration"], current_price)
            contracts = int(opt["contracts"])
            strike_val = float(opt["strike"])
            if price_at_exp > strike_val:
                # Called away
                proceeds = strike_val * contracts * 100
                cash += proceeds
                shares -= contracts * 100
                assignment_today = True
                call_assignment_count += 1
                log_lines.append(f"[{current_date.date()}] ⚠️ CALL assigned: -{contracts*100} sh @ ${strike_val:.2f} (px@exp={price_at_exp:.2f})")

                # Rebuy via 1–3DTE PUT at same strike (or nearest)
                if not df_today.empty:
                    put_row = choose_put_for_assignment(df_today, current_date, strike_val)
                    if put_row is not None:
                        chosen_strike = float(put_row["strike"])
                        expiry = pd.to_datetime(put_row["expiration"])
                        # size up to assigned contracts, subject to collateral
                        cash_available = cash - reserved_collateral
                        max_by_collat = int(cash_available // (chosen_strike * 100))
                        target_contracts = min(contracts, max_by_collat)
                        if target_contracts > 0:
                            premium = float(put_row["vw"]) * target_contracts * 100
                            cash += premium
                            total_premium += premium
                            put_premium_collected += premium
                            reserved_collateral += chosen_strike * target_contracts * 100
                            active_puts.append({
                                "strike": chosen_strike,
                                "expiration": expiry,
                                "contracts": int(target_contracts),
                                "type": "put",
                                "exercised": False,
                                # tag this PUT as a rebuy for loss accounting
                                "rebuy_ref_strike": float(strike_val),
                            })
                            clip_note = "" if target_contracts == contracts else f" (clipped by collateral from {contracts}→{target_contracts})"
                            exact_note = "exact-strike" if np.isclose(chosen_strike, strike_val, atol=1e-6) else f"nearest-strike to {strike_val:.2f}"
                            log_lines.append(
                                f"[{current_date.date()}] 🔁 Rebuy-via-PUT: Sold {target_contracts} CSP @ ${chosen_strike:.2f} "
                                f"(1–3DTE {expiry.date()}), +${premium:,.2f}; reserve ${chosen_strike*target_contracts*100:,.2f}{clip_note} [{exact_note}]"
                            )
                        else:
                            log_lines.append(f"[{current_date.date()}] ⏸️ Rebuy-via-PUT skipped: insufficient collateral for strike ${chosen_strike:.2f}")
                    else:
                        log_lines.append(f"[{current_date.date()}] ⏸️ Rebuy-via-PUT skipped: no 1–3DTE PUT near strike ${strike_val:.2f}")
                else:
                    log_lines.append(f"[{current_date.date()}] ⏸️ Rebuy-via-PUT skipped: no option quotes today")
            opt["exercised"] = True

    # 3b) Expirations — PUTs
    for put in active_puts[:]:
        if (current_date >= put["expiration"]) and (not put["exercised"]):
            price_at_exp = price_on_or_before(price_df, put["expiration"], current_price)
            contracts = int(put["contracts"])
            strike_val = float(put["strike"])
            collat = strike_val * contracts * 100
            if price_at_exp < strike_val:
                # Assigned — buy shares using collateral
                cash -= collat
                shares += contracts * 100
                put_assignment_count += 1

                # If this PUT was tagged as "rebuy for a CALL assignment", realize loss = rebuy_price - call_strike
                ref = put.get("rebuy_ref_strike", None)
                if ref is not None:
                    realized = (strike_val - float(ref)) * contracts * 100
                    total_call_assignment_loss += realized
                    sign = "loss" if realized >= 0 else "gain"
                    log_lines.append(
                        f"[{current_date.date()}] ✅ PUT assigned (rebuy): +{contracts*100} sh @ ${strike_val:.2f}; "
                        f"reacq-{sign}: ${realized:,.2f} (rebuy {strike_val:.2f} - call {float(ref):.2f})"
                    )
                else:
                    log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} sh @ ${strike_val:.2f} (px@exp={price_at_exp:.2f})")
            else:
                log_lines.append(f"[{current_date.date()}] ✅ PUT expired OTM: release collateral on {contracts}x @ ${strike_val:.2f}")

            reserved_collateral -= collat
            put["exercised"] = True

    # 4) CASH DEPLOYMENT — idle cash via ATM CSP (untagged)
    cash_available = cash - reserved_collateral
    if cash_available >= current_price * 100 and not df_today.empty:
        df_puts_idle = df_today[
            (df_today["type"].str.lower() == "put") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=PUT_DTE_MIN)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=PUT_DTE_MAX))
        ].copy()
        if not df_puts_idle.empty:
            df_puts_idle["atm_gap"] = (df_puts_idle["strike"] - current_price).abs()
            put_choice = df_puts_idle.sort_values(["atm_gap","expiration"]).iloc[0]
            strike_val = float(put_choice["strike"])
            max_contracts = int(cash_available // (strike_val * 100))
            if max_contracts > 0:
                premium = float(put_choice["vw"]) * max_contracts * 100
                cash += premium
                total_premium += premium
                put_premium_collected += premium
                reserved_collateral += strike_val * max_contracts * 100
                active_puts.append({
                    "strike": strike_val,
                    "expiration": pd.to_datetime(put_choice["expiration"]),
                    "contracts": int(max_contracts),
                    "type": "put",
                    "exercised": False,
                    "rebuy_ref_strike": None,  # idle-cash CSP not tied to CALL loss accounting
                })
                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold PUT (idle cash) +${premium:,.2f} "
                    f"@ strike ${strike_val:.2f}, expiry {pd.to_datetime(put_choice['expiration']).date()}, "
                    f"contracts {max_contracts} (ATM 1–3DTE)"
                )
            else:
                log_lines.append(f"[{current_date.date()}] ⏸️ CSP skipped (idle): collateral not enough for strike {strike_val:.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] ⏸️ CSP skipped (idle): no 1–3DTE ATM puts available")

    # 5) Sell covered CALL (unchanged rule)
    has_active_call_now = any((o["type"] == "call") and (not o["exercised"]) for o in active_calls)
    allow_sell_call_today = (not assignment_today) and (not has_active_call_now) and shares >= 100
    if allow_sell_call_today and (not df_today.empty) and rule_allows_today(gap_today):
        df_calls = df_today[
            (df_today["type"].str.lower() == "call") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=DTE_MIN)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=DTE_MAX))
        ]
        if not df_calls.empty:
            iv_today = float(df_calls["iv"].mean())
            target_delta = iv_to_delta(iv_today)

            subset = df_calls[df_calls["delta"] <= target_delta + 0.01].copy()
            chosen = None
            reason = "delta-match"
            if not subset.empty:
                subset["delta_gap"] = (subset["delta"] - target_delta).abs()
                chosen = subset.sort_values("delta_gap").iloc[0]

            if USE_STRIKE_FLOOR:
                floor_strike = current_price * (1.0 + STRIKE_FLOOR_PCT)
                if (chosen is None) or (float(chosen["strike"]) < floor_strike):
                    floor_cands = df_calls[df_calls["strike"] >= floor_strike].copy()
                    if not floor_cands.empty:
                        floor_cands["strike_gap"] = (floor_cands["strike"] - floor_strike).abs()
                        chosen = floor_cands.sort_values("strike_gap").iloc[0]
                        reason = "floor-enforced"
                        floor_enforced_count += 1
                    else:
                        log_lines.append(
                            f"[{current_date.date()}] ⏸️ No CC sold: no contract meets strike floor "
                            f"(floor {STRIKE_FLOOR_PCT:.0%}, needed ≥ {floor_strike:.2f})."
                        )
                        chosen = None

            if chosen is not None:
                contracts = int(shares // 100)
                premium = float(chosen["vw"]) * contracts * 100
                cash += premium
                total_premium += premium
                call_premium_collected += premium
                why = ("Gap-Up day" if gap_today == "Gap Up" else
                       "Gap-Down day" if gap_today == "Gap Down" else "rule off")
                more = f", floor={STRIKE_FLOOR_PCT:.0%}" if (USE_STRIKE_FLOOR and reason == "floor-enforced") else ""
                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold CALL +${premium:,.2f} "
                    f"@ strike ${float(chosen['strike']):.2f}, Δ={float(chosen['delta']):.3f}, "
                    f"expiry {pd.to_datetime(chosen['expiration']).date()}, contracts {contracts} "
                    f"(reason: {why}, pick={reason}{more})"
                )
                active_calls.append({
                    "strike": float(chosen["strike"]),
                    "expiration": pd.to_datetime(chosen["expiration"]),
                    "contracts": int(contracts),
                    "type": "call",
                    "exercised": False,
                })
    elif allow_sell_call_today and (not df_today.empty) and not rule_allows_today(gap_today):
        note = "not Gap-Up" if SELL_ONLY_ON_GAP_UP else "not Gap-Down"
        log_lines.append(f"[{current_date.date()}] ⏸️ Skip selling CALL: {note} day.")

    # 6) DCA benchmark
    if cash_bh >= current_price * 100:
        can_buy_bh = shares_affordable(cash_bh, current_price)
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * current_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 7) Curves & uncovered days
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

summary = f"""
Backtest Summary — CC + CSP, with CALL-assignment rebuy via PUT @ same strike (1–3 DTE)
Period: {period_text}
CALL strike floor: {"ON" if USE_STRIKE_FLOOR else "OFF"}{(f" (≥ {STRIKE_FLOOR_PCT:.0%}, enforced {floor_enforced_count}×)") if USE_STRIKE_FLOOR else ""}
DCA contrib: ${DCA_AMOUNT:,.0f} every {DCA_INTERVAL_TRADING_DAYS} trading days
--------------------------------------------------
CALL premium collected:       ${call_premium_collected:,.2f}
PUT premium collected:        ${put_premium_collected:,.2f}
Total option premium:         ${total_premium:,.2f}

CALL assignment count:        {call_assignment_count}
CALL assignment loss (reacq): ${total_call_assignment_loss:,.2f}

PUT assignment count:         {put_assignment_count}
Reserved collateral (final):  ${reserved_collateral:,.2f}

Days ≥100sh w/o CC:           {uncovered_days}  (out of {held_100plus_days}, {uncovered_ratio:.1%})

Final equity (Strategy):      ${final_value_cc:,.2f}
Final equity (DCA):           ${final_value_bh:,.2f}
Total invested capital:       ${total_invested:,.2f}

Total return (Strategy):      {(final_value_cc / total_invested - 1.0):.2%}
Total return (DCA):           {(final_value_bh / total_invested - 1.0):.2%}
CAGR (Strategy):              {cagr_cc:.2%}
CAGR (DCA):                   {cagr_bh:.2%}

Sharpe (Strategy):            {sharpe_cc:.2f}
Sharpe (DCA):                 {sharpe_bh:.2f}

Excess over DCA:              ${excess:,.2f}  |  per year: ${excess_per_year:,.2f}
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
plt.plot(dates, curve_cc.values, label="Strategy (CC + CSP; rebuy-via-PUT after CALL assignment)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("Strategy Comparison: CC + CSP vs DCA")
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
  <h1>Strategy: CC + CSP (rebuy via PUT after CALL assignment)</h1>
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


