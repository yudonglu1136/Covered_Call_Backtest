# -*- coding: utf-8 -*-
"""
Covered Call Backtest (QQQ) — Calls only baseline,
with CASH DEPLOYMENT via CSP (ATM 1–3 DTE) + TQQQ add-on under Fear & Greed.

Changes relative to your last version:
  - When cash can buy ≥ 1 lot (100 shares) of QQQ, DO NOT buy shares immediately.
    Instead, sell ATM cash-secured PUT(s) with 1–3 DTE. Use their strike*100*contracts
    as reserved collateral (not available to anything else).
  - If PUT expires ITM => assigned => +shares; OTM => keep premium, release collateral.
  - On days when Fear & Greed (FG) < 15, deploy ALL *free* cash (i.e., cash - reserved_put_cash)
    into TQQQ integer shares (not lots). Take-profit each batch at +100%.
  - Robust CSV loading for Fear_and_greed.csv and TQQQ_ohlcv_1d.csv (case-insensitive headers).

All other CC mechanics, statistics, plotting, and summary remain the same style.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/covered_call")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63   # ~ quarterly
DCA_AMOUNT = 15000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00

# ---- Strategy toggles for CALL leg (unchanged) ----
SELL_ONLY_ON_GAP_DOWN = False
SELL_ONLY_ON_GAP_UP   = False
DTE_MIN, DTE_MAX = 28, 31

# Strike floor for CALL（原有机制，未更动）
USE_STRIKE_FLOOR = True
STRIKE_FLOOR_PCT = 0.07  # floor = 7% OTM

# ---- CSP deployment window ----
PUT_DTE_MIN, PUT_DTE_MAX = 1, 3   # 卖现金担保PUT的到期区间（1-3天）

# ---- TQQQ add-on under Fear & Greed ----
FG_BUY_THRESHOLD = 15  # FG < 15 才买 TQQQ
TP_MULTIPLE = 2.5      # +100% take-profit
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.quant_utils import (
    normalize_date_series,
    iv_to_delta, price_on_or_before, shares_affordable_for_put,
    sharpe_ratio, compute_mdd,fmt_currency,load_fear_greed_csv, load_tqqq_open_csv,shares_affordable
)
# ============================
# Helpers
# ============================


# ============================
# Load Core Data
# ============================

DATA_DIR = Path("data")

df_options = pd.read_csv(DATA_DIR/"options_with_iv_delta.csv")
price_raw  = pd.read_csv(DATA_DIR/"QQQ_ohlcv_1d.csv")  # has: date, open, high, low, close, ...
div_df     = pd.read_csv(DATA_DIR/"QQQ_dividends.csv")

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

# --- Prepare price dataframe & compute Gap label (before filtering) ---
if "date" not in price_raw.columns:
    raise KeyError("QQQ_ohlcv_1d.csv must contain a 'date' column")
if "Open" not in price_raw.columns:
    if "open" in price_raw.columns:
        price_raw = price_raw.rename(columns={"open": "Open"})
    else:
        raise KeyError("QQQ_ohlcv_1d.csv must contain 'Open' (or 'open') column")

price_raw["date"]  = normalize_date_series(price_raw["date"])
price_raw["Open"]  = pd.to_numeric(price_raw["Open"], errors="coerce")
if "close" not in price_raw.columns:
    raise KeyError("QQQ_ohlcv_1d.csv must contain 'close' for Gap calc")

# Gap label on full daily series
price_raw = price_raw.sort_values("date").set_index("date")
price_raw["prev_close"] = price_raw["close"].shift(1)
price_raw["gap_label"] = pd.NA
price_raw.loc[price_raw["Open"] > price_raw["prev_close"], "gap_label"] = "Gap Up"
price_raw.loc[price_raw["Open"] < price_raw["prev_close"], "gap_label"] = "Gap Down"

# Filter prices to only option dates (for speed/consistency)
opt_dates = pd.Index(df_options["date"].unique())
price_df  = price_raw.loc[price_raw.index.isin(opt_dates), ["Open","gap_label"]].copy()

print(f"[INFO] price_df rows kept: {len(price_df):,} | first: {price_df.index.min().date()} | last: {price_df.index.max().date()}")

dates = price_df.index

# ---- Load Fear & Greed + TQQQ (robust) ----
fg_series   = load_fear_greed_csv(Path("data/Fear_and_greed.csv"))
tqqq_open_s = load_tqqq_open_csv(Path("data/TQQQ_ohlcv_1d.csv"))

# ============================
# State & Stats
# ============================
cash = float(INITIAL_CASH)
shares = 0                                 # QQQ shares
active_calls = []
total_premium = 0.0
contribution_count = 0

# CSP state
active_puts = []                 # dict: strike, expiration, contracts, exercised
reserved_collateral = 0.0        # sum of strike*100*contracts for open CSP

# Benchmark
cash_bh = float(INITIAL_CASH)
shares_bh = 0

log_lines = []
portfolio_value = []
buy_hold_value = []

assignment_count = 0            # CALL assignments
total_assignment_loss = 0.0
uncovered_days = 0
held_100plus_days = 0
floor_enforced_count = 0

# TQQQ state
tqqq_batches = []               # [{qty, entry, date}]
tqqq_realized_pnl = 0.0
tqqq_buy_events = []
tqqq_sell_events = []

# ============================
# Helper for rule gating
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
    df_today = df_options[df_options["date"] == current_date]
    assignment_today = False  # 阻止当日新卖CALL（保持你的原逻辑）

    # 1) Quarterly contribution
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        contribution_count += 1
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash balance (Strategy): ${cash:,.2f}")

    # 2) Dividends
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty and shares > 0:
        dividend_per_share = float(div_row["dividend"].values[0])
        credited = dividend_per_share * shares
        cash += credited
        log_lines.append(f"[{current_date.date()}] 📦 Dividend credited +${credited:,.2f} (${dividend_per_share:.4f}/share on {shares} shares)")

    # 3) Handle expirations — CALL
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(price_df, opt["expiration"], current_price)
            contracts = int(opt["contracts"])
            if price_at_exp > float(opt["strike"]):
                # Called away
                cash += float(opt["strike"]) * contracts * 100
                shares -= contracts * 100
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ⚠️ CALL assigned: -{contracts*100} shares @ ${float(opt['strike']):.2f}")
                assignment_count += 1
                total_assignment_loss += (price_at_exp - float(opt["strike"])) * contracts * 100

                # Repurchase immediately（保持原逻辑：若现金够，再买回尽量多）
                lots = shares_affordable(cash, current_price)
                if lots > 0:
                    cost = lots * current_price
                    cash -= cost
                    shares += lots
                    log_lines.append(f"[{current_date.date()}] 🔁 Post-assignment repurchase {lots} shares @ ${current_price:.2f}")
            opt["exercised"] = True

    # 3b) Handle expirations — PUT
    for put in active_puts[:]:
        if (current_date >= put["expiration"]) and (not put["exercised"]):
            price_at_exp = price_on_or_before(price_df, put["expiration"], current_price)
            contracts = int(put["contracts"])
            strike_val = float(put["strike"])
            collat = strike_val * contracts * 100
            if price_at_exp < strike_val:
                # PUT assigned: buy shares at strike using reserved collateral
                cash -= collat
                shares += contracts * 100
                log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} shares @ ${strike_val:.2f}")
            else:
                # Expired worthless, keep premium
                log_lines.append(f"[{current_date.date()}] ✅ PUT expired OTM: collateral released on {contracts} contract(s) @ ${strike_val:.2f}")
            # Release collateral in both cases
            reserved_collateral -= collat
            put["exercised"] = True

    # 4) CASH DEPLOYMENT — CSP instead of immediate QQQ buy
    cash_available = cash - reserved_collateral  # free cash not tied in CSP collateral
    if cash_available >= current_price * 100 and not df_today.empty:
        # Choose 1–3 DTE ATM puts
        df_puts = df_today[
            (df_today["type"].str.lower() == "put") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=PUT_DTE_MIN)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=PUT_DTE_MAX))
        ].copy()

        if not df_puts.empty:
            df_puts["atm_gap"] = (df_puts["strike"] - current_price).abs()
            put_choice = df_puts.sort_values(["atm_gap","expiration"]).iloc[0]
            strike_val = float(put_choice["strike"])
            if strike_val > 0:
                max_contracts = int(cash_available // (strike_val * 100))
                if max_contracts > 0:
                    premium = float(put_choice["vw"]) * max_contracts * 100
                    cash += premium
                    total_premium += premium
                    reserved_collateral += strike_val * max_contracts * 100
                    active_puts.append({
                        "strike": strike_val,
                        "expiration": pd.to_datetime(put_choice["expiration"]),
                        "contracts": int(max_contracts),
                        "type": "put",
                        "exercised": False,
                    })
                    log_lines.append(
                        f"[{current_date.date()}] 💰 Sold PUT (CSP) +${premium:,.2f} "
                        f"@ strike ${strike_val:.2f}, expiry {pd.to_datetime(put_choice['expiration']).date()}, "
                        f"contracts {max_contracts} (ATM, DTE {PUT_DTE_MIN}-{PUT_DTE_MAX})"
                    )
                else:
                    log_lines.append(f"[{current_date.date()}] ⏸️ CSP skipped: collateral not enough for strike {strike_val:.2f}")
            else:
                log_lines.append(f"[{current_date.date()}] ⏸️ CSP skipped: invalid strike")
        else:
            log_lines.append(f"[{current_date.date()}] ⏸️ CSP skipped: no 1–3DTE ATM puts available")

    # 4a) TQQQ buy when FG < threshold — only use FREE cash (not collateral)
    fg_val = fg_series.loc[current_date] if current_date in fg_series.index else np.nan
    tqqq_price = float(tqqq_open_s.loc[current_date]) if current_date in tqqq_open_s.index else np.nan

    if pd.notna(fg_val) and fg_val < FG_BUY_THRESHOLD and pd.notna(tqqq_price) and tqqq_price > 0:
        free_cash = max(0.0, cash - reserved_collateral)
        qty = int(free_cash // tqqq_price)  # integer shares
        if qty > 0:
            cost = qty * tqqq_price
            cash -= cost
            tqqq_batches.append({"qty": qty, "entry": tqqq_price, "date": current_date})
            tqqq_buy_events.append((current_date, tqqq_price, qty))
            log_lines.append(
                f"[{current_date.date()}] 🛒 TQQQ buy (FG={fg_val:.0f}) {qty} shares @ ${tqqq_price:.2f}; "
                f"cash left: ${cash:,.2f} (reserved: ${reserved_collateral:,.2f})"
            )

    # 4b) TQQQ take-profit at +100%
    if pd.notna(tqqq_price) and tqqq_batches:
        kept_batches = []
        for b in tqqq_batches:
            entry = float(b["entry"]); qty = int(b["qty"])
            if tqqq_price >= TP_MULTIPLE * entry and qty > 0:
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

    # 5) Sell covered CALL —（与你原逻辑一致）
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
        log_lines.append(f"[{current_date.date()}] ⏸️ Skip selling: {note} day.")

    # 6) DCA benchmark — unchanged
    if cash_bh >= current_price * 100:
        can_buy_bh = shares_affordable(cash_bh, current_price)
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * current_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 7) Track equity + uncovered days
    # unrealized TQQQ value (mark at today's open if available)
    tqqq_qty_total = sum(int(b["qty"]) for b in tqqq_batches)
    tqqq_mtm_val = (tqqq_qty_total * tqqq_price) if (pd.notna(tqqq_price) and tqqq_qty_total > 0) else 0.0

    portfolio_value.append(shares * current_price + cash + tqqq_mtm_val)
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

# rule 描述
if SELL_ONLY_ON_GAP_UP and not SELL_ONLY_ON_GAP_DOWN:
    rule_desc_core = f"ONLY on Gap-Up days"
elif SELL_ONLY_ON_GAP_DOWN and not SELL_ONLY_ON_GAP_UP:
    rule_desc_core = f"ONLY on Gap-Down days"
elif (not SELL_ONLY_ON_GAP_UP) and (not SELL_ONLY_ON_GAP_DOWN):
    rule_desc_core = f"Every eligible day"
else:
    rule_desc_core = f"Gap-Up or Gap-Down days (both toggles ON)"
rule_desc = f"{rule_desc_core}; monthly {DTE_MIN}-{DTE_MAX} DTE"

period_text = f"{dates.min().date()} → {dates.max().date()}"
net_premium_after_loss = total_premium - total_assignment_loss
uncovered_ratio = (uncovered_days / held_100plus_days) if held_100plus_days > 0 else 0.0

# ==== Summary ====
summary = f"""
Backtest Summary (CC + CSP + TQQQ)
Rule: Sell new call {rule_desc}
Cash deployment: Sell ATM CSP (PUT), DTE {PUT_DTE_MIN}-{PUT_DTE_MAX}; FG<{FG_BUY_THRESHOLD} buy TQQQ (integer shares), TP at +100%
Period: {period_text}
Strike floor (CALL): {"ON" if USE_STRIKE_FLOOR else "OFF"}{(f" (≥ {STRIKE_FLOOR_PCT:.0%}, enforced {floor_enforced_count}×)") if USE_STRIKE_FLOOR else ""}
--------------------------------------------------
Option premium collected (CALL+PUT): ${total_premium:,.2f}
CALL assignment count:                {assignment_count}
CALL total assignment loss:           ${total_assignment_loss:,.2f}
Net premium after CALL loss:          ${net_premium_after_loss:,.2f}

TQQQ batches:      {len(tqqq_batches)} open
TQQQ realized PnL: ${tqqq_realized_pnl:,.2f}
Days holding ≥100 shares w/o CC: {uncovered_days}  (out of {held_100plus_days}, {uncovered_ratio:.1%})

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
--------------------------------------------------
"""
print(summary)

# Logs & outputs
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
plt.plot(dates, curve_cc.values, label="Strategy (CC + CSP cash deployment + TQQQ FG add-on)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("Strategy Comparison: CC + CSP + TQQQ-FG vs DCA")
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
  <h1>Strategy Comparison: CC + CSP + TQQQ-FG vs DCA</h1>
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

