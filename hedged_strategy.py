# -*- coding: utf-8 -*-
"""
Covered Strangle Backtest (QQQ) — CALLs + cash-secured PUTs, with TQQQ buys on FG<15
+ Hedge v2: allow stacking multiple long PUT hedges (one per signal day), settle each precisely.

Outputs saved to: output/hedged_strategy/
All comments/logs/prints are in English.

Inputs under ./data/ :
  - options_with_iv_delta.csv  # columns: date, expiration, type(call/put), strike, vw, iv, delta
  - QQQ_ohlcv_1d.csv           # columns: date, open/... (Open will be normalized)
  - TQQQ_ohlcv_1d.csv          # columns: date, open/...
  - QQQ_dividends.csv          # columns: date, dividend
  - Fear_and_greed.csv         # columns: Date, Value (0~100)
  - put_signals.csv            # columns: (date, signal)  -- flexible names supported
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/hedged_strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CASH = 300_000
DCA_INTERVAL_TRADING_DAYS = 63   # quarterly contribution (approx.)
DCA_AMOUNT = 15_000
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_ANNUAL = 0.00

FG_BUY_THRESHOLD = 15            # FG < 15 triggers TQQQ buy
PUT_MIN_DTE = 1
PUT_MAX_DTE = 3
CALL_MIN_DTE = 15
CALL_MAX_DTE = 18
DELTA_BAND = 0.01
CALL_STRIKE_FLOOR_PCT = 1.06     # 106% of spot

# Hedge parameters
HEDGE_DTE_LOWER = 28
HEDGE_DTE_UPPER = 31

# ============================
# Helpers
# ============================
def normalize_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.normalize()

def iv_to_delta(iv: float, steepness: float = 10, mid: float = 0.27,
                min_delta: float = 0.018, max_delta: float = 0.15) -> float:
    norm = 1.0 / (1.0 + math.exp(-steepness * (mid - float(iv))))
    delta_val = min_delta + (max_delta - min_delta) * norm
    return round(float(delta_val), 4)

def price_on_or_before(idx_price: pd.DataFrame, ts: pd.Timestamp, current_fallback: float) -> float:
    if ts in idx_price.index:
        return float(idx_price.loc[ts, "Open"])
    earlier = idx_price.index[idx_price.index <= ts]
    if len(earlier) > 0:
        return float(idx_price.loc[earier[-1], "Open"])
    return float(current_fallback)

def shares_affordable_for_put(cash_free: float, strike: float) -> int:
    if strike <= 0:
        return 0
    return int(cash_free // (strike * 100))

def sharpe_ratio(equity_curve: pd.Series, rf_annual: float = 0.0, periods_per_year: int = 252) -> float:
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return 0.0
    rf_per_period = rf_annual / periods_per_year
    excess = rets - rf_per_period
    std = excess.std()
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))

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

def pick_atm_option(df_chain: pd.DataFrame, spot: float, d1: int, d2: int):
    """Pick ATM option within [d1, d2] DTE; fallback widen ranges if empty."""
    def _pick(df):
        if df.empty:
            return None
        df = df.copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"])
        df["atm_gap"] = (df["strike"] - spot).abs()
        df = df.sort_values(["expiration", "atm_gap"])
        return df.iloc[0]
    # exact range
    df = df_chain[(df_chain["expiration"] >= current_date + pd.Timedelta(days=d1)) &
                  (df_chain["expiration"] <= current_date + pd.Timedelta(days=d2))]
    pick = _pick(df)
    if pick is not None:
        return pick
    # widen 25-35
    df = df_chain[(df_chain["expiration"] >= current_date + pd.Timedelta(days=25)) &
                  (df_chain["expiration"] <= current_date + pd.Timedelta(days=35))]
    pick = _pick(df)
    if pick is not None:
        return pick
    # widen 20-40
    df = df_chain[(df_chain["expiration"] >= current_date + pd.Timedelta(days=20)) &
                  (df_chain["expiration"] <= current_date + pd.Timedelta(days=40))]
    pick = _pick(df)
    return pick

# ============================
# Load Data
# ============================
df_options = pd.read_csv("data/options_with_iv_delta.csv")
qqq_raw    = pd.read_csv("data/QQQ_ohlcv_1d.csv")
tqqq_raw   = pd.read_csv("data/TQQQ_ohlcv_1d.csv")
div_df     = pd.read_csv("data/QQQ_dividends.csv")
fg_raw     = pd.read_csv("data/Fear_and_greed.csv")
put_sig    = pd.read_csv("data/put_signals.csv")  # flexible column names handled below

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])

qqq_df  = prep_price_df(qqq_raw)
tqqq_df = prep_price_df(tqqq_raw)

# Keep only the option dates universe for prices
opt_dates = pd.Index(df_options["date"].dropna().unique())
qqq_df  = qqq_df[qqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
tqqq_df = tqqq_df[tqqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
dates = qqq_df.index

# Fear & Greed aligned
fg_raw = fg_raw.rename(columns={"Date":"date","date":"date","Value":"value","value":"value"})
if "date" not in fg_raw.columns or "value" not in fg_raw.columns:
    raise KeyError(f"Fear_and_greed.csv must have ['Date','Value']; got: {list(fg_raw.columns)}")
fg_raw["date"] = normalize_date_series(fg_raw["date"])
fg_raw["value"] = pd.to_numeric(fg_raw["value"], errors="coerce")
fg_series = fg_raw.set_index("date")["value"].sort_index().reindex(dates).ffill()

# Put signal normalize
sig_date_col = None
for c in ["date","Date","DATE","datetime","time","timestamp"]:
    if c in put_sig.columns:
        sig_date_col = c; break
if sig_date_col is None:
    raise KeyError(f"No date-like column in put_signals.csv: {list(put_sig.columns)}")
put_sig["date"] = normalize_date_series(put_sig[sig_date_col])

sig_col = None
for c in ["signal","Signal","hedge","Hedge","sig","Sig"]:
    if c in put_sig.columns:
        sig_col = c; break
if sig_col is None:
    numeric_cols = [c for c in put_sig.columns if c != "date" and pd.api.types.is_numeric_dtype(put_sig[c])]
    if not numeric_cols:
        raise KeyError("put_signals.csv must contain a numeric signal column.")
    sig_col = numeric_cols[0]

put_sig = put_sig[["date", sig_col]].rename(columns={sig_col: "signal"}).dropna(subset=["date"])
put_sig["signal"] = (pd.to_numeric(put_sig["signal"], errors="coerce") > 0).astype(int)
put_signal_series = put_sig.set_index("date")["signal"].reindex(dates).fillna(0).astype(int)

# ============================
# State Variables
# ============================
cash = float(INITIAL_CASH)
shares = 0                        # QQQ shares
reserved_put_cash = 0.0           # collateral for short PUT
total_premium = 0.0               # SHORT option premiums + CALL premiums received

# Active options
active_calls = []                 # short covered calls
active_put   = None               # short cash-secured put (to (re)build QQQ)

# Hedge v2: allow multiple long puts at the same time
active_hedges = []                # list of dicts: {"id","buy_date","strike","expiration","premium","contracts","settled"}
hedge_id_seq = 0                  # incremental ID to pair buy & expiry
hedge_premium_paid = 0.0
hedge_payout_received = 0.0
# Events with IDs
# BUY:    (buy_date, id, strike, expiry, premium)
# EXPIRE: (expire_date, id, strike, expiry, payoff, itm, premium)
hedge_buy_events = []
hedge_expire_events = []

# TQQQ holdings
tqqq_batches = []                 # {"qty": int, "entry": float, "date": Timestamp}
tqqq_realized_pnl = 0.0
tqqq_buy_events = []              # (date, price, qty)
tqqq_sell_events = []             # (date, price, qty, pnl)

# DCA benchmark
cash_bh = float(INITIAL_CASH)
shares_bh = 0

# Logs & equity trackers
log_lines = []
portfolio_value = []
buy_hold_value = []

# ============================
# Main Loop
# ============================
for i, current_date in enumerate(dates):
    qqq_price  = float(qqq_df.loc[current_date, "Open"])
    tqqq_price = float(tqqq_df.loc[current_date, "Open"])
    df_today = df_options[df_options["date"] == current_date]
    assignment_today = False

    # 1) Contribution
    if i > 0 and i % DCA_INTERVAL_TRADING_DAYS == 0:
        cash += DCA_AMOUNT
        cash_bh += DCA_AMOUNT
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash: ${cash:,.2f}")

    # 2) Dividends
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty and shares > 0:
        dps = float(div_row["dividend"].values[0])
        cred = dps * shares
        cash += cred
        log_lines.append(f"[{current_date.date()}] 📦 Dividend +${cred:,.2f} (${dps:.4f}/sh on {shares})")

    # 3) Expirations / assignments
    # 3a) Covered CALLs
    for opt in active_calls[:]:
        if (current_date >= opt["expiration"]) and (not opt["exercised"]):
            price_at_exp = price_on_or_before(qqq_df, opt["expiration"], qqq_price)
            contracts = int(opt["contracts"])
            if price_at_exp > float(opt["strike"]):
                cash += float(opt["strike"]) * contracts * 100
                shares -= contracts * 100
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ⚠️ CALL assigned: -{contracts*100} @ ${float(opt['strike']):.2f}")
            opt["exercised"] = True

    # 3b) Short PUT (cash-secured)
    if active_put is not None and (current_date >= active_put["expiration"]) and (not active_put["exercised"]):
        price_at_exp = price_on_or_before(qqq_df, active_put["expiration"], qqq_price)
        contracts = int(active_put["contracts"])
        strike = float(active_put["strike"])
        cost = strike * contracts * 100

        if price_at_exp < strike:
            if cost <= cash:
                shares += contracts * 100
                cash -= cost
                assignment_today = True
                log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} @ ${strike:.2f}")
            else:
                log_lines.append(f"[{current_date.date()}] ⚠️ PUT assignment failed; need ${cost:,.2f}, have ${cash:,.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] ⏳ PUT expired worthless @ ${strike:.2f}")

        reserved_put_cash = 0.0
        active_put["exercised"] = True
        active_put = None

    # 3c) Hedge v2: settle any hedge(s) reaching expiration today or earlier
    if active_hedges:
        still_open = []
        for h in active_hedges:
            if (current_date >= h["expiration"]) and (not h.get("settled", False)):
                strike = float(h["strike"])
                exp = pd.to_datetime(h["expiration"])
                price_at_exp = price_on_or_before(qqq_df, exp, qqq_price)
                payoff = max(0.0, (strike - price_at_exp)) * 100.0 * int(h.get("contracts", 1))
                cash += payoff
                hedge_payout_received += payoff
                itm = payoff > 0.0
                hedge_expire_events.append((current_date, h["id"], strike, exp, payoff, itm, h["premium"]))
                h["settled"] = True
                log_lines.append(
                    f"[{current_date.date()}] 🛡️ Hedge PUT expired (id={h['id']}): "
                    f"strike ${strike:.2f}, payoff ${payoff:,.2f}, ITM={itm}"
                )
            # keep only non-settled hedges
            if not h.get("settled", False):
                still_open.append(h)
        active_hedges = still_open

    # 4) TQQQ actions driven by FG
    fg_val = fg_series.loc[current_date] if current_date in fg_series.index else np.nan
    if pd.notna(fg_val) and fg_val < FG_BUY_THRESHOLD and tqqq_price > 0:
        free_cash = max(0.0, cash - reserved_put_cash)  # do not use reserved collateral
        qty = int(free_cash // tqqq_price)
        if qty > 0:
            cost = qty * tqqq_price
            cash -= cost
            tqqq_batches.append({"qty": qty, "entry": tqqq_price, "date": current_date})
            tqqq_buy_events.append((current_date, tqqq_price, qty))
            log_lines.append(f"[{current_date.date()}] 🛒 TQQQ buy(FG={fg_val:.0f}) {qty} @ ${tqqq_price:.2f}, cash ${cash:,.2f}")

    if tqqq_batches:
        kept = []
        for b in tqqq_batches:
            entry = float(b["entry"]); qty = int(b["qty"])
            if tqqq_price >= 2.0 * entry and qty > 0:
                proceeds = qty * tqqq_price
                pnl = qty * (tqqq_price - entry)
                cash += proceeds
                tqqq_realized_pnl += pnl
                tqqq_sell_events.append((current_date, tqqq_price, qty, pnl))
                log_lines.append(f"[{current_date.date()}] ✅ TQQQ TP: sell {qty} @ ${tqqq_price:.2f} (entry ${entry:.2f}), P&L +${pnl:,.2f}")
            else:
                kept.append(b)
        tqqq_batches = kept

    # 5) Hedge v2: on signal==1, try to buy 1x ATM PUT (28–31 DTE) — stacking allowed
    if int(put_signal_series.loc[current_date]) == 1:
        df_puts = df_today[df_today["type"].str.lower() == "put"].copy()
        if df_puts.empty:
            log_lines.append(f"[{current_date.date()}] 🛡️ Hedge signal=1 but skipped: no PUT rows for today.")
        else:
            df_puts["strike"] = pd.to_numeric(df_puts["strike"], errors="coerce")
            df_puts = df_puts.dropna(subset=["strike"])
            pick = pick_atm_option(df_puts, qqq_price, HEDGE_DTE_LOWER, HEDGE_DTE_UPPER)
            if pick is None:
                log_lines.append(
                    f"[{current_date.date()}] 🛡️ Hedge signal=1 but skipped: no ATM candidate within DTE windows."
                )
            else:
                exp = pd.to_datetime(pick["expiration"])
                strike = float(pick["strike"])
                vw = pd.to_numeric(pd.Series([pick.get("vw", np.nan)]), errors="coerce").iloc[0]
                if not np.isfinite(vw):
                    log_lines.append(
                        f"[{current_date.date()}] 🛡️ Hedge skipped: invalid vw for selected option (strike {strike:.2f}, exp {exp.date()})."
                    )
                else:
                    premium = float(vw) * 100.0  # 1 contract
                    if premium <= cash:  # keep cash check
                        cash -= premium
                        hedge_premium_paid += premium
                        hedge_id_seq += 1
                        h = {
                            "id": hedge_id_seq,
                            "buy_date": current_date,
                            "strike": strike,
                            "expiration": exp,
                            "premium": premium,
                            "contracts": 1,
                            "settled": False,
                        }
                        active_hedges.append(h)
                        hedge_buy_events.append((current_date, h["id"], strike, exp, premium))
                        log_lines.append(
                            f"[{current_date.date()}] 🛡️ Buy Hedge PUT id={h['id']} @ strike ${strike:.2f}, "
                            f"expiry {exp.date()}, vw ${vw:.2f}, premium ${premium:,.2f}"
                        )
                    else:
                        log_lines.append(
                            f"[{current_date.date()}] 🛡️ Hedge skipped: insufficient cash. Need ${premium:,.2f}, have ${cash:,.2f}."
                        )

    # 6) Short cash-secured PUT to (re)build shares (ATM, 1–3 DTE), after TQQQ actions
    has_active_put = active_put is not None
    if (not has_active_put) and (not df_today.empty):
        df_puts2 = df_today[
            (df_today["type"].str.lower() == "put") &
            (df_today["expiration"] >= current_date + pd.Timedelta(days=PUT_MIN_DTE)) &
            (df_today["expiration"] <= current_date + pd.Timedelta(days=PUT_MAX_DTE))
        ].copy()
        if not df_puts2.empty:
            df_puts2["strike"] = pd.to_numeric(df_puts2["strike"], errors="coerce")
            df_puts2 = df_puts2.dropna(subset=["strike"])
            df_puts2["atm_gap"] = (df_puts2["strike"] - qqq_price).abs()
            put_opt = df_puts2.sort_values(["atm_gap","expiration"]).iloc[0]
            put_strike = float(put_opt["strike"])

            free_cash = max(0.0, cash - reserved_put_cash)
            max_contracts = shares_affordable_for_put(free_cash, put_strike)
            if max_contracts > 0:
                premium = float(put_opt["vw"]) * 100 * max_contracts
                cash += premium
                total_premium += premium
                reserved_put_cash = put_strike * 100 * max_contracts
                active_put = {
                    "strike": put_strike,
                    "expiration": pd.to_datetime(put_opt["expiration"]),
                    "contracts": int(max_contracts),
                    "type": "put",
                    "exercised": False,
                }
                log_lines.append(
                    f"[{current_date.date()}] 💰 Sold CSP +${premium:,.2f} @ ${put_strike:.2f}, "
                    f"expiry {pd.to_datetime(put_opt['expiration']).date()}, contracts {max_contracts}; "
                    f"reserved ${reserved_put_cash:,.2f}"
                )

    # 7) Covered CALL (15–18 DTE) with 106% strike floor (skip on assignment day)
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

            candidates = df_calls[
                (df_calls["delta"] <= target_delta + DELTA_BAND) &
                (df_calls["strike"] >= strike_floor)
            ].copy()
            if candidates.empty:
                candidates = df_calls[df_calls["strike"] >= strike_floor].copy()
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
                    f"[{current_date.date()}] 💰 Sold CALL +${premium:,.2f} @ ${chosen_strike:.2f} "
                    f"(floor ${strike_floor:.2f}), Δ={chosen_delta:.3f}, "
                    f"expiry {pd.to_datetime(option['expiration']).date()}, contracts {contracts}"
                )

                active_calls.append({
                    "strike": chosen_strike,
                    "expiration": pd.to_datetime(option["expiration"]),
                    "contracts": int(contracts),
                    "type": "call",
                    "exercised": False,
                })

    # 8) DCA benchmark
    if cash_bh >= qqq_price * 100:
        can_buy_bh = int(cash_bh // qqq_price) // 100 * 100
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * qqq_price
            shares_bh += can_buy_bh
            cash_bh -= cost_bh

    # 9) Equity
    tqqq_qty = sum(int(b["qty"]) for b in tqqq_batches) if tqqq_batches else 0
    equity_strategy = shares * qqq_price + cash + tqqq_qty * tqqq_price
    portfolio_value.append(equity_strategy)
    buy_hold_value.append(shares_bh * qqq_price + cash_bh)

# ============================
# Stats & Output
# ============================
curve_cc = pd.Series(portfolio_value, index=dates, name="Strategy")
curve_bh = pd.Series(buy_hold_value, index=dates, name="DCA")

final_value_cc = float(curve_cc.iloc[-1])
final_value_bh = float(curve_bh.iloc[-1])

contrib_cnt = (len(dates) - 1) // DCA_INTERVAL_TRADING_DAYS
total_invested = INITIAL_CASH + contrib_cnt * DCA_AMOUNT

years = (dates[-1] - dates[0]).days / 365.0 if len(dates) > 1 else 0.0
cagr_cc = (final_value_cc / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
cagr_bh = (final_value_bh / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
excess = final_value_cc - final_value_bh
excess_per_year = excess / years if years > 0 else 0.0

sharpe_cc = sharpe_ratio(curve_cc, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)
sharpe_bh = sharpe_ratio(curve_bh, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)

hedge_net_pnl = hedge_payout_received - hedge_premium_paid
tqqq_qty_now = sum(int(b["qty"]) for b in tqqq_batches) if tqqq_batches else 0

summary = f"""
Backtest Summary (Covered Strangle + TQQQ on FG<15 + Hedge PUTs stacked on signals)
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

Hedge premium paid:         ${hedge_premium_paid:,.2f}
Hedge payouts received:     ${hedge_payout_received:,.2f}
Hedge net P&L:              ${hedge_net_pnl:,.2f}
Open hedge count:           {len(active_hedges)}
----------------------------
"""
print(summary)

# Logs
log_path = OUTPUT_DIR / "strategy_hedged.log"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines:
        f.write(line + "\n")
    f.write("\n" + summary)

# CSVs
eq_csv = OUTPUT_DIR / "equity_curves.csv"
pd.DataFrame({"date": dates, "strategy_value": curve_cc.values, "dca_value": curve_bh.values}).to_csv(eq_csv, index=False)

tqqq_trades_csv = OUTPUT_DIR / "tqqq_trades.csv"
rows = []
for d,p,q in tqqq_buy_events:
    rows.append({"date": d, "side": "BUY", "price": p, "qty": q, "pnl": np.nan})
for d,p,q,pnl in tqqq_sell_events:
    rows.append({"date": d, "side": "SELL", "price": p, "qty": q, "pnl": pnl})
pd.DataFrame(rows).to_csv(tqqq_trades_csv, index=False)

# Hedge trades CSV with IDs and full info
hedge_trades_csv = OUTPUT_DIR / "hedge_puts.csv"
rows = []
for d, hid, strk, exp, prem in hedge_buy_events:
    rows.append({
        "event": "BUY_HEDGE_PUT", "id": hid, "date": d, "strike": strk,
        "expiry": exp, "premium": prem, "payoff": np.nan, "itm": np.nan
    })
for d, hid, strk, exp, pay, itm, prem in hedge_expire_events:
    rows.append({
        "event": "EXPIRE", "id": hid, "date": d, "strike": strk,
        "expiry": exp, "premium": prem, "payoff": pay, "itm": int(itm)
    })
pd.DataFrame(rows).sort_values(["date","event","id"]).to_csv(hedge_trades_csv, index=False)

with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)

# ============================
# Plots
# ============================
ts_label = dates[-1].date()
main_png = OUTPUT_DIR / f"strategy_comparison_{ts_label}.png"
main_pdf = OUTPUT_DIR / f"strategy_comparison_{ts_label}.pdf"
tqqq_png = OUTPUT_DIR / f"tqqq_trades_{ts_label}.png"
tqqq_pdf = OUTPUT_DIR / f"tqqq_trades_{ts_label}.pdf"
hedge_png = OUTPUT_DIR / f"hedge_overview_{ts_label}.png"
hedge_pdf = OUTPUT_DIR / f"hedge_overview_{ts_label}.pdf"
html_path = OUTPUT_DIR / "report.html"

# 1) Main equity curves
plt.figure(figsize=(12, 6))
plt.plot(dates, curve_cc.values, label="Strategy (hedged)")
plt.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
plt.title("📊 Strategy vs DCA")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (USD)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(main_png, dpi=150, bbox_inches="tight"); plt.savefig(main_pdf, bbox_inches="tight"); plt.close()

# 2) TQQQ trades
plt.figure(figsize=(12, 6))
plt.plot(dates, tqqq_df.loc[dates, "Open"].values, label="TQQQ Open")
if tqqq_buy_events:
    bd = [d for d,_,_ in tqqq_buy_events]; bp = [p for _,p,_ in tqqq_buy_events]
    plt.scatter(bd, bp, marker="^", s=64, label="TQQQ BUY")
if tqqq_sell_events:
    sd = [d for d,_,_,_ in tqqq_sell_events]; sp = [p for _,p,_,_ in tqqq_sell_events]
    plt.scatter(sd, sp, marker="v", s=64, label="TQQQ SELL")
plt.title("📈 TQQQ Trades (FG<15 buys, +100% take-profit)")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(tqqq_png, dpi=150, bbox_inches="tight"); plt.savefig(tqqq_pdf, bbox_inches="tight"); plt.close()

# 3) Hedge overview: QQQ price + hedge events + cumulative hedge P&L + signal
# Build cumulative hedge P&L series for plotting
hedge_pnl_curve = []
cum = 0.0
sig_vals = []
prices = []
for d in dates:
    prices.append(qqq_df.loc[d, "Open"])
    sig_vals.append(put_signal_series.loc[d] if d in put_signal_series.index else 0)
    pnl_delta = 0.0
    # premiums when bought (multiple possible per day)
    for bd, hid, strk, exp, prem in [x for x in hedge_buy_events if x[0] == d]:
        pnl_delta -= prem
    # payoff on expiry (multiple possible per day)
    for ed, hid, strk, exp, pay, itm, prem in [x for x in hedge_expire_events if x[0] == d]:
        pnl_delta += pay
    cum += pnl_delta
    hedge_pnl_curve.append(cum)

plt.figure(figsize=(13, 7))
ax1 = plt.gca()
ax1.plot(dates, prices, label="QQQ Open")
# mark hedge buys
if hedge_buy_events:
    hd = [d for d,_,_,_,_ in hedge_buy_events]
    hp = [qqq_df.loc[d, "Open"] for d in hd]
    ax1.scatter(hd, hp, marker="o", s=48, label="Hedge PUT Buy")
# mark expiries: ITM vs OTM
if hedge_expire_events:
    ed_itm = [d for d,_,_,_,_,itm,_ in hedge_expire_events if itm]
    ep_itm = [qqq_df.loc[d, "Open"] for d in ed_itm]
    ed_otm = [d for d,_,_,_,_,itm,_ in hedge_expire_events if not itm]
    ep_otm = [qqq_df.loc[d, "Open"] for d in ed_otm]
    if ed_itm:
        ax1.scatter(ed_itm, ep_itm, marker="*", s=90, label="Hedge Expire ITM")
    if ed_otm:
        ax1.scatter(ed_otm, ep_otm, marker="x", s=60, label="Hedge Expire OTM")
ax1.set_xlabel("Date"); ax1.set_ylabel("QQQ Price (USD)")
ax1.grid(True); ax1.legend(loc="upper left")

# add cumulative hedge P&L (secondary axis)
ax2 = ax1.twinx()
ax2.plot(dates, hedge_pnl_curve, label="Cumulative Hedge P&L", linestyle="--")
ax2.set_ylabel("Cumulative Hedge P&L (USD)")
ax2.legend(loc="upper right")

# overlay signal (scaled)
sig_scaled = np.array(sig_vals, dtype=float)
if sig_scaled.max() > 0:
    sig_scaled = sig_scaled / sig_scaled.max() * max(1.0, np.nanmax(hedge_pnl_curve) * 0.2 if len(hedge_pnl_curve) else 1.0)
    ax2.plot(dates, sig_scaled, label="PUT Signal (scaled)", alpha=0.6)
    ax2.legend(loc="lower right")

plt.title("🛡️ Hedge Overview: QQQ, Hedge Events, Cumulative Hedge P&L, Signal")
plt.tight_layout()
plt.savefig(hedge_png, dpi=150, bbox_inches="tight")
plt.savefig(hedge_pdf, bbox_inches="tight")
plt.close()

# HTML report
html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Hedged Strategy Report</title></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;max-width:1000px;margin:40px auto;">
  <h1>Hedged Strategy — Covered Strangle + TQQQ(FG) + PUT Hedge (stacked)</h1>
  <pre style="background:#f6f8fa;padding:16px;border-radius:8px;white-space:pre-wrap;">{summary}</pre>

  <h2>Main Equity Curves</h2>
  <figure><img src="{main_png.name}" style="max-width:100%;height:auto;"><figcaption>{main_png.name}</figcaption></figure>

  <h2>TQQQ Buy/Sell Events</h2>
  <figure><img src="{tqqq_png.name}" style="max-width:100%;height:auto;"><figcaption>{tqqq_png.name}</figcaption></figure>

  <h2>Hedge Overview</h2>
  <figure><img src="{hedge_png.name}" style="max-width:100%;height:auto;"><figcaption>{hedge_png.name}</figcaption></figure>

  <p>Equity curves: {eq_csv.name}</p>
  <p>TQQQ trades: {tqqq_trades_csv.name}</p>
  <p>Hedge trades: {hedge_trades_csv.name}</p>
  <p>Log: {log_path.name}</p>
</body>
</html>"""
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"Figures saved: {main_png}, {tqqq_png}, {hedge_png}")
print(f"HTML report saved: {html_path}")


