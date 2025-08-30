# -*- coding: utf-8 -*-
"""
Covered Strangle Backtest (QQQ) — CALLs + cash-secured PUTs, with TQQQ buys on FG<15
+ Hedge v2: stacking long PUTs (one per signal day), precise settlement.
+ Costs: commissions & slippage for options.
+ Cap hedge size: open long PUT contracts <= floor(shares/100 * 0.5)
+ MDD annotation, yearly hedge stats table.

Inputs under ./data/ :
  - options_with_iv_delta.csv  # columns: date, expiration, type(call/put), strike, vw, iv, delta
  - QQQ_ohlcv_1d.csv           # columns: date, open/... (Open will be normalized)
  - TQQQ_ohlcv_1d.csv          # columns: date, open/...
  - QQQ_dividends.csv          # columns: date, dividend
  - Fear_and_greed.csv         # columns: Date, Value (0~100)
  - put_signals.csv            # columns: (date, signal)
Outputs to: output/hedged_strategy/
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.quant_utils import (
    parse_date_or_none, open_hedge_contracts, normalize_date_series,
    iv_to_delta, price_on_or_before, shares_affordable_for_put,
    sharpe_ratio, prep_price_df, compute_mdd, fmt_currency
)

# ============================
# Config
# ============================
OUTPUT_DIR = Path("output/hedged_strategy")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- Backtest window (inclusive). Set to "" or None to use full range. --------
START_DATE = "2021-07-26"   # e.g. "2024-01-01" or None / ""
END_DATE   = "2025-08-21"   # e.g. "2025-12-31" or None / ""

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
HEDGE_DTE_LOWER = 7
HEDGE_DTE_UPPER = 10

# --------- Costs (options only) ---------
COMMISSION_PER_CONTRACT = 0.65           # $/contract
BUY_SLIPPAGE_PCT  = 0.15                 # long PUT buy: paid = premium*(1+pct)
SELL_SLIPPAGE_PCT = 0.10                 # short options sell: received = premium*(1-pct)

# ============================
# Helpers
# ============================


START_TS = parse_date_or_none(START_DATE)
END_TS   = parse_date_or_none(END_DATE)


def pick_atm_option(df_chain: pd.DataFrame, spot: float, d1: int, d2: int):
    def _pick(df):
        if df.empty: return None
        df = df.copy()
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df = df.dropna(subset=["strike"])
        df["atm_gap"] = (df["strike"] - spot).abs()
        df = df.sort_values(["expiration", "atm_gap"])
        return df.iloc[0]
    df = df_chain[(df_chain["expiration"] >= current_date + pd.Timedelta(days=d1)) &
                  (df_chain["expiration"] <= current_date + pd.Timedelta(days=d2))]
    pick = _pick(df)
    if pick is not None: return pick
    df = df_chain[(df_chain["expiration"] >= current_date + pd.Timedelta(days=25)) &
                  (df_chain["expiration"] <= current_date + pd.Timedelta(days=35))]
    pick = _pick(df)
    if pick is not None: return pick
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
put_sig    = pd.read_csv("data/put_signals_T1.csv")

# Normalize
df_options["date"] = normalize_date_series(df_options["date"])
df_options["expiration"] = normalize_date_series(df_options["expiration"])
div_df["date"] = normalize_date_series(div_df["date"])
qqq_df  = prep_price_df(qqq_raw)
tqqq_df = prep_price_df(tqqq_raw)
fg_raw = fg_raw.rename(columns={"Date":"date","date":"date","Value":"value","value":"value"})
fg_raw["date"] = normalize_date_series(fg_raw["date"])
fg_raw["value"] = pd.to_numeric(fg_raw["value"], errors="coerce")
# signal
sig_date_col = None
for c in ["date","Date","DATE","datetime","time","timestamp"]:
    if c in put_sig.columns: sig_date_col = c; break
if sig_date_col is None: raise KeyError(f"No date-like column in put_signals.csv: {list(put_sig.columns)}")
put_sig["date"] = normalize_date_series(put_sig[sig_date_col])
sig_col = None
for c in ["signal","Signal","hedge","Hedge","sig","Sig"]:
    if c in put_sig.columns: sig_col = c; break
if sig_col is None:
    numeric_cols = [c for c in put_sig.columns if c != "date" and pd.api.types.is_numeric_dtype(put_sig[c])]
    if not numeric_cols: raise KeyError("put_signals.csv must contain a numeric signal column.")
    sig_col = numeric_cols[0]
put_sig = put_sig[["date", sig_col]].rename(columns={sig_col: "signal"}).dropna(subset=["date"])
put_sig["signal"] = (pd.to_numeric(put_sig["signal"], errors="coerce") > 0).astype(int)

# -------- Apply backtest window BEFORE building the trading date index --------
def clip_df_by_window(df, date_col="date"):
    if START_TS is not None: df = df[df[date_col] >= START_TS]
    if END_TS   is not None: df = df[df[date_col] <= END_TS]
    return df

df_options = clip_df_by_window(df_options, "date")
qqq_df     = clip_df_by_window(qqq_df, "date")
tqqq_df    = clip_df_by_window(tqqq_df, "date")
div_df     = clip_df_by_window(div_df, "date")
fg_raw     = clip_df_by_window(fg_raw, "date")
put_sig    = clip_df_by_window(put_sig, "date")

# Keep only the option dates universe for prices
opt_dates = pd.Index(df_options["date"].dropna().unique())
qqq_df  = qqq_df[qqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
tqqq_df = tqqq_df[tqqq_df["date"].isin(opt_dates)].sort_values("date").set_index("date")
dates = qqq_df.index

# Align FG and signals
fg_series = fg_raw.set_index("date")["value"].sort_index().reindex(dates).ffill()
put_signal_series = put_sig.set_index("date")["signal"].reindex(dates).fillna(0).astype(int)

# ============================
# State Variables
# ============================
cash = float(INITIAL_CASH)
shares = 0
reserved_put_cash = 0.0
total_premium = 0.0

active_calls = []
active_put   = None

active_hedges = []
hedge_id_seq = 0
hedge_premium_paid = 0.0
hedge_payout_received = 0.0
hedge_buy_events = []
hedge_expire_events = []

tqqq_batches = []
tqqq_realized_pnl = 0.0
tqqq_buy_events = []
tqqq_sell_events = []

cash_bh = float(INITIAL_CASH)
shares_bh = 0

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
        cash += DCA_AMOUNT;  cash_bh += DCA_AMOUNT
        log_lines.append(f"[{current_date.date()}] 💰 Contribution +${DCA_AMOUNT:,.0f}, Cash: ${cash:,.2f}")

    # 2) Dividends
    div_row = div_df[div_df["date"] == current_date]
    if not div_row.empty and shares > 0:
        dps = float(div_row["dividend"].values[0]); cred = dps * shares
        cash += cred
        log_lines.append(f"[{current_date.date()}] 📦 Dividend +${cred:,.2f} (${dps:.4f}/sh on {shares})")

    # 3) Expirations / assignments
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

    if active_put is not None and (current_date >= active_put["expiration"]) and (not active_put["exercised"]):
        price_at_exp = price_on_or_before(qqq_df, active_put["expiration"], qqq_price)
        contracts = int(active_put["contracts"]); strike = float(active_put["strike"]); cost = strike * contracts * 100
        if price_at_exp < strike:
            if cost <= cash:
                shares += contracts * 100; cash -= cost; assignment_today = True
                log_lines.append(f"[{current_date.date()}] ✅ PUT assigned: +{contracts*100} @ ${strike:.2f}")
            else:
                log_lines.append(f"[{current_date.date()}] ⚠️ PUT assignment failed; need ${cost:,.2f}, have ${cash:,.2f}")
        else:
            log_lines.append(f"[{current_date.date()}] ⏳ PUT expired worthless @ ${strike:.2f}")
        reserved_put_cash = 0.0; active_put["exercised"] = True; active_put = None

    if active_hedges:
        still_open = []
        for h in active_hedges:
            if (current_date >= h["expiration"]) and (not h.get("settled", False)):
                strike = float(h["strike"]); exp = pd.to_datetime(h["expiration"])
                price_at_exp = price_on_or_before(qqq_df, exp, qqq_price)
                payoff = max(0.0, (strike - price_at_exp)) * 100.0 * int(h.get("contracts", 1))
                cash += payoff; hedge_payout_received += payoff
                itm = payoff > 0.0
                hedge_expire_events.append((current_date, h["id"], strike, exp, payoff, itm, h["premium"]))
                h["settled"] = True
                log_lines.append(f"[{current_date.date()}] 🛡️ Hedge PUT expired (id={h['id']}): strike ${strike:.2f}, payoff ${payoff:,.2f}, ITM={itm}")
            if not h.get("settled", False): still_open.append(h)
        active_hedges = still_open

    # 4) TQQQ actions
    fg_val = fg_series.loc[current_date] if current_date in fg_series.index else np.nan
    if pd.notna(fg_val) and fg_val < FG_BUY_THRESHOLD and tqqq_price > 0:
        free_cash = max(0.0, cash - reserved_put_cash)
        qty = int(free_cash // tqqq_price)
        if qty > 0:
            cost = qty * tqqq_price; cash -= cost
            tqqq_batches.append({"qty": qty, "entry": tqqq_price, "date": current_date})
            tqqq_buy_events.append((current_date, tqqq_price, qty))
            log_lines.append(f"[{current_date.date()}] 🛒 TQQQ buy(FG={fg_val:.0f}) {qty} @ ${tqqq_price:.2f}, cash ${cash:,.2f}")

    if tqqq_batches:
        kept = []
        for b in tqqq_batches:
            entry = float(b["entry"]); qty = int(b["qty"])
            if tqqq_price >= 2.0 * entry and qty > 0:
                proceeds = qty * tqqq_price; pnl = qty * (tqqq_price - entry)
                cash += proceeds; tqqq_realized_pnl += pnl
                tqqq_sell_events.append((current_date, tqqq_price, qty, pnl))
                log_lines.append(f"[{current_date.date()}] ✅ TQQQ TP: sell {qty} @ ${tqqq_price:.2f} (entry ${entry:.2f}), P&L +${pnl:,.2f}")
            else: kept.append(b)
        tqqq_batches = kept

    # 5) Hedge buy (cap <= 0.5 * lots)
    if int(put_signal_series.loc[current_date]) == 1:
        max_allowed = int((shares // 100) * 0.5)
        currently_open = open_hedge_contracts(active_hedges)
        if max_allowed <= 0:
            log_lines.append(f"[{current_date.date()}] 🛡️ Hedge skipped: no equity lots (shares={shares}), cap=0.")
        elif currently_open >= max_allowed:
            log_lines.append(f"[{current_date.date()}] 🛡️ Hedge skipped: cap reached ({currently_open}/{max_allowed}).")
        else:
            df_puts = df_today[df_today["type"].str.lower() == "put"].copy()
            if df_puts.empty:
                log_lines.append(f"[{current_date.date()}] 🛡️ Hedge signal=1 but no PUT rows.")
            else:
                df_puts["strike"] = pd.to_numeric(df_puts["strike"], errors="coerce")
                df_puts = df_puts.dropna(subset=["strike"])
                pick = pick_atm_option(df_puts, qqq_price, HEDGE_DTE_LOWER, HEDGE_DTE_UPPER)
                if pick is None:
                    log_lines.append(f"[{current_date.date()}] 🛡️ Hedge signal=1 but no ATM candidate.")
                else:
                    exp = pd.to_datetime(pick["expiration"]); strike = float(pick["strike"])
                    vw = pd.to_numeric(pd.Series([pick.get("vw", np.nan)]), errors="coerce").iloc[0]
                    if not np.isfinite(vw):
                        log_lines.append(f"[{current_date.date()}] 🛡️ Hedge skipped: invalid vw.")
                    else:
                        premium = float(vw) * 100.0
                        premium_paid = premium * (1.0 + BUY_SLIPPAGE_PCT) + COMMISSION_PER_CONTRACT * 1
                        if premium_paid > 0:
                            cash -= premium_paid; hedge_premium_paid += premium_paid
                            hedge_id_seq += 1
                            h = {"id": hedge_id_seq, "buy_date": current_date, "strike": strike,
                                 "expiration": exp, "premium": premium_paid, "contracts": 1, "settled": False}
                            active_hedges.append(h)
                            hedge_buy_events.append((current_date, h["id"], strike, exp, premium_paid))
                            log_lines.append(f"[{current_date.date()}] 🛡️ Buy Hedge PUT id={h['id']} strike ${strike:.2f} exp {exp.date()} cost {fmt_currency(premium_paid)}")
                        else:
                            log_lines.append(f"[{current_date.date()}] 🛡️ Hedge skipped: need {fmt_currency(premium_paid)}, have {fmt_currency(cash)}.")

    # 6) Cash-secured PUT sell (1–3 DTE)
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
                vw = float(put_opt["vw"]); gross = vw * 100 * max_contracts
                premium_rcv = gross * (1.0 - SELL_SLIPPAGE_PCT) - COMMISSION_PER_CONTRACT * max_contracts
                cash += premium_rcv; total_premium += premium_rcv
                reserved_put_cash = put_strike * 100 * max_contracts
                active_put = {"strike": put_strike, "expiration": pd.to_datetime(put_opt["expiration"]),
                              "contracts": int(max_contracts), "type": "put", "exercised": False}
                log_lines.append(f"[{current_date.date()}] 💰 Sold CSP +{fmt_currency(premium_rcv)} @ ${put_strike:.2f} exp {pd.to_datetime(put_opt['expiration']).date()} "
                                 f"contracts {max_contracts}; reserved {fmt_currency(reserved_put_cash)}")

    # 7) Covered CALL
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
            iv_today = float(df_calls["iv"].mean()); target_delta = iv_to_delta(iv_today)
            strike_floor = round(CALL_STRIKE_FLOOR_PCT * qqq_price, 2)
            candidates = df_calls[(df_calls["delta"] <= target_delta + DELTA_BAND) & (df_calls["strike"] >= strike_floor)].copy()
            if candidates.empty: candidates = df_calls[df_calls["strike"] >= strike_floor].copy()
            if candidates.empty: candidates = df_calls[df_calls["delta"] <= target_delta + DELTA_BAND].copy()
            if not candidates.empty:
                candidates["delta_gap"] = (candidates["delta"] - target_delta).abs()
                if (candidates["strike"] >= strike_floor).any():
                    candidates = candidates.sort_values(by=["strike","delta_gap"], ascending=[True,True])
                else:
                    candidates = candidates.sort_values(by=["delta_gap","strike"], ascending=[True,True])
                option = candidates.iloc[0]
                chosen_strike = float(option["strike"]); chosen_delta = float(option["delta"])
                contracts = int(shares // 100); vw = float(option["vw"]); gross = vw * contracts * 100
                premium_rcv = gross * (1.0 - SELL_SLIPPAGE_PCT) - COMMISSION_PER_CONTRACT * contracts
                cash += premium_rcv; total_premium += premium_rcv
                log_lines.append(f"[{current_date.date()}] 💰 Sold CALL +{fmt_currency(premium_rcv)} @ ${chosen_strike:.2f} Δ={chosen_delta:.3f} "
                                 f"exp {pd.to_datetime(option['expiration']).date()} contracts {contracts}")
                active_calls.append({"strike": chosen_strike, "expiration": pd.to_datetime(option["expiration"]),
                                     "contracts": int(contracts), "type": "call", "exercised": False})

    # 8) DCA benchmark
    if cash_bh >= qqq_price * 100:
        can_buy_bh = int(cash_bh // qqq_price) // 100 * 100
        if can_buy_bh > 0:
            cost_bh = can_buy_bh * qqq_price; shares_bh += can_buy_bh; cash_bh -= cost_bh

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

final_value_cc = float(curve_cc.iloc[-1]); final_value_bh = float(curve_bh.iloc[-1])
contrib_cnt = (len(dates) - 1) // DCA_INTERVAL_TRADING_DAYS
total_invested = INITIAL_CASH + contrib_cnt * DCA_AMOUNT
years = (dates[-1] - dates[0]).days / 365.0 if len(dates) > 1 else 0.0
cagr_cc = (final_value_cc / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
cagr_bh = (final_value_bh / total_invested) ** (1.0 / years) - 1.0 if years > 0 else 0.0
sharpe_cc = sharpe_ratio(curve_cc, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)
sharpe_bh = sharpe_ratio(curve_bh, rf_annual=RISK_FREE_ANNUAL, periods_per_year=TRADING_DAYS_PER_YEAR)

def _fmt_date(d): return d.date().isoformat() if d is not None else "N/A"
mdd_cc, peak_cc, trough_cc, rec_cc = compute_mdd(curve_cc)
mdd_bh, peak_bh, trough_bh, rec_bh = compute_mdd(curve_bh)

hedge_net_pnl = hedge_payout_received - hedge_premium_paid
tqqq_qty_now = sum(int(b["qty"]) for b in tqqq_batches) if tqqq_batches else 0

window_str = f"{START_TS.date() if START_TS else 'BEGIN'} → {END_TS.date() if END_TS else 'END'}"
summary = f"""
Backtest Summary (Costs ON)  |  Window: {window_str}
---------------------------------------------------------------------------
Costs: Commission ${COMMISSION_PER_CONTRACT:.2f}/ctr | BUY slip +{BUY_SLIPPAGE_PCT:.0%} | SELL slip -{SELL_SLIPPAGE_PCT:.0%}

Final equity (Strategy):    {fmt_currency(final_value_cc)}
Final equity (DCA):         {fmt_currency(final_value_bh)}
Total invested capital:     {fmt_currency(total_invested)}

Total return (Strategy):    {(final_value_cc / total_invested - 1.0):.2%}
Total return (DCA):         {(final_value_bh / total_invested - 1.0):.2%}
CAGR (Strategy):            {cagr_cc:.2%}
CAGR (DCA):                 {cagr_bh:.2%}
Sharpe (Strategy):          {sharpe_cc:.2f}
Sharpe (DCA):               {sharpe_bh:.2f}

Max Drawdown (Strategy):    {mdd_cc:.2%}  (peak {_fmt_date(peak_cc)} → trough {_fmt_date(trough_cc)}{'' if rec_cc is None else ' → rec ' + _fmt_date(rec_cc)})
Max Drawdown (DCA):         {mdd_bh:.2%}  (peak {_fmt_date(peak_bh)} → trough {_fmt_date(trough_bh)}{'' if rec_bh is None else ' → rec ' + _fmt_date(rec_bh)})

Option premium collected (net):   {fmt_currency(total_premium)}
TQQQ realized P&L:                {fmt_currency(tqqq_realized_pnl)}
TQQQ open position:               {tqqq_qty_now} shares

Hedge premium paid (after costs): {fmt_currency(hedge_premium_paid)}
Hedge payouts received:           {fmt_currency(hedge_payout_received)}
Hedge net P&L:                    {fmt_currency(hedge_net_pnl)}
Open hedge count:                 {len(active_hedges)}
---------------------------------------------------------------------------
"""
print(summary)

# Logs
log_path = OUTPUT_DIR / "strategy_hedged.log"
with open(log_path, "w", encoding="utf-8") as f:
    for line in log_lines: f.write(line + "\n")
    f.write("\n" + summary)

# ---------- Hedge yearly stats ----------
hedge_trades_df = []
buy_map = {}
for (d, hid, strk, exp, prem) in hedge_buy_events:
    buy_map[hid] = {"buy_date": pd.to_datetime(d), "strike": float(strk), "expiry": pd.to_datetime(exp), "premium": float(prem)}
for (d, hid, strk, exp, payoff, itm, prem) in hedge_expire_events:
    d = pd.to_datetime(d); exp = pd.to_datetime(exp)
    row = {
        "id": hid,
        "buy_date": buy_map.get(hid, {}).get("buy_date", pd.NaT),
        "expiry_date": exp,
        "strike": float(strk),
        "premium_paid": float(buy_map.get(hid, {}).get("premium", prem)),
        "payoff": float(payoff),
        "pnl": float(payoff) - float(buy_map.get(hid, {}).get("premium", prem)),
        "itm": int(bool(itm)),
    }
    hedge_trades_df.append(row)
hedge_trades_df = pd.DataFrame(hedge_trades_df)

def compute_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["year","trades","wins","losses","win_rate","gross_profit","gross_loss","net_pnl"])
    df["year"] = df["expiry_date"].dt.year
    grp = df.groupby("year")
    trades = grp.size().rename("trades")
    wins = grp.apply(lambda x: (x["pnl"] > 0).sum()).rename("wins")
    losses = grp.apply(lambda x: (x["pnl"] <= 0).sum()).rename("losses")
    win_rate = (wins / trades).fillna(0.0).rename("win_rate")
    gross_profit = grp.apply(lambda x: x.loc[x["pnl"] > 0, "pnl"].sum()).rename("gross_profit")
    gross_loss   = grp.apply(lambda x: -x.loc[x["pnl"] < 0, "pnl"].sum()).rename("gross_loss")
    net_pnl      = grp["pnl"].sum().rename("net_pnl")
    out = pd.concat([trades, wins, losses, win_rate, gross_profit, gross_loss, net_pnl], axis=1).reset_index()
    all_row = pd.DataFrame({
        "year": ["ALL"],
        "trades": [trades.sum()],
        "wins": [wins.sum()],
        "losses": [losses.sum()],
        "win_rate": [ (wins.sum() / max(1, trades.sum())) if trades.sum()>0 else 0.0 ],
        "gross_profit": [gross_profit.sum()],
        "gross_loss": [gross_loss.sum()],
        "net_pnl": [net_pnl.sum()],
    })
    return pd.concat([out, all_row], ignore_index=True)

yearly_stats = compute_yearly_stats(hedge_trades_df)
yr_csv = OUTPUT_DIR / "hedge_put_yearly_stats.csv"
yearly_stats.to_csv(yr_csv, index=False)

def save_table_png(df: pd.DataFrame, path_png: Path, title="Short / Hedge PUT — Yearly Stats"):
    fig, ax = plt.subplots(figsize=(12, 4 + 0.35*len(df))); ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    df_fmt = df.copy(); df_fmt["win_rate"] = (df_fmt["win_rate"]*100.0).map(lambda x: f"{x:.2f}%")
    for col in ["gross_profit","gross_loss","net_pnl"]: df_fmt[col] = df_fmt[col].map(lambda x: fmt_currency(x))
    tbl = ax.table(cellText=df_fmt.values, colLabels=df_fmt.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(12); tbl.scale(1, 1.2)
    plt.tight_layout(); plt.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close()

yr_png = OUTPUT_DIR / "hedge_put_yearly_stats.png"
save_table_png(yearly_stats, yr_png)

# ============================
# CSVs & Plots
# ============================
eq_csv = OUTPUT_DIR / "equity_curves.csv"
pd.DataFrame({"date": dates, "strategy_value": curve_cc.values, "dca_value": curve_bh.values}).to_csv(eq_csv, index=False)

tqqq_trades_csv = OUTPUT_DIR / "tqqq_trades.csv"
rows = []
for d,p,q in tqqq_buy_events: rows.append({"date": d, "side": "BUY", "price": p, "qty": q, "pnl": np.nan})
for d,p,q,pnl in tqqq_sell_events: rows.append({"date": d, "side": "SELL", "price": p, "qty": q, "pnl": pnl})
pd.DataFrame(rows).to_csv(tqqq_trades_csv, index=False)

hedge_trades_csv = OUTPUT_DIR / "hedge_puts.csv"
rows = []
for d, hid, strk, exp, prem in hedge_buy_events:
    rows.append({"event": "BUY_HEDGE_PUT", "id": hid, "date": d, "strike": strk, "expiry": exp, "premium_paid": prem})
for d, hid, strk, exp, pay, itm, prem in hedge_expire_events:
    rows.append({"event": "EXPIRE", "id": hid, "date": d, "strike": strk, "expiry": exp, "premium_paid": prem, "payoff": pay, "itm": int(itm)})
pd.DataFrame(rows).sort_values(["date","event","id"]).to_csv(hedge_trades_csv, index=False)

# Plots
ts_label = f"{(START_TS.date() if START_TS else 'BEGIN')}_to_{(END_TS.date() if END_TS else 'END')}"
main_png = OUTPUT_DIR / f"strategy_comparison_{ts_label}.png"
main_pdf = OUTPUT_DIR / f"strategy_comparison_{ts_label}.pdf"
tqqq_png = OUTPUT_DIR / f"tqqq_trades_{ts_label}.png"
tqqq_pdf = OUTPUT_DIR / f"tqqq_trades_{ts_label}.pdf"
hedge_png = OUTPUT_DIR / f"hedge_overview_{ts_label}.png"
hedge_pdf = OUTPUT_DIR / f"hedge_overview_{ts_label}.pdf"
html_path = OUTPUT_DIR / "report.html"

plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.plot(dates, curve_cc.values, label="Strategy (hedged)")
ax.plot(dates, curve_bh.values, label="Buy & Hold (Quarterly DCA)")
if (peak_cc is not None) and (trough_cc is not None):
    x_end = rec_cc if rec_cc is not None else dates[-1]
    ax.axvspan(peak_cc, x_end, color="tab:red", alpha=0.12, label=f"Strategy MDD {mdd_cc:.1%}")
    ax.scatter([trough_cc], [curve_cc.loc[trough_cc]], marker="v", s=60, color="tab:red")
if (peak_bh is not None) and (trough_bh is not None):
    x_end_b = rec_bh if rec_bh is not None else dates[-1]
    ax.axvspan(peak_bh, x_end_b, color="tab:blue", alpha=0.12, label=f"DCA MDD {mdd_bh:.1%}")
    ax.scatter([trough_bh], [curve_bh.loc[trough_bh]], marker="v", s=60, color="tab:blue")
ax.set_title(f"Strategy vs DCA (MDD)  | Window: {window_str}")
ax.set_xlabel("Date"); ax.set_ylabel("Portfolio Value (USD)")
ax.legend(); ax.grid(True); plt.tight_layout()
plt.savefig(main_png, dpi=150, bbox_inches="tight"); plt.savefig(main_pdf, bbox_inches="tight"); plt.close()

plt.figure(figsize=(12, 6))
plt.plot(dates, tqqq_df.loc[dates, "Open"].values, label="TQQQ Open")
if tqqq_buy_events:
    bd = [d for d,_,_ in tqqq_buy_events]; bp = [p for _,p,_ in tqqq_buy_events]
    plt.scatter(bd, bp, marker="^", s=64, label="TQQQ BUY")
if tqqq_sell_events:
    sd = [d for d,_,_,_ in tqqq_sell_events]; sp = [p for _,p,_,_ in tqqq_sell_events]
    plt.scatter(sd, sp, marker="v", s=64, label="TQQQ SELL")
plt.title(f"TQQQ Trades (FG<15 buys, +100% TP)  | Window: {window_str}")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(tqqq_png, dpi=150, bbox_inches="tight"); plt.savefig(tqqq_pdf, bbox_inches="tight"); plt.close()

hedge_pnl_curve = []; cum = 0.0; sig_vals = []; prices = []
for d in dates:
    prices.append(qqq_df.loc[d, "Open"])
    sig_vals.append(put_signal_series.loc[d] if d in put_signal_series.index else 0)
    pnl_delta = 0.0
    for bd, hid, strk, exp, prem in [x for x in hedge_buy_events if x[0] == d]: pnl_delta -= prem
    for ed, hid, strk, exp, pay, itm, prem in [x for x in hedge_expire_events if x[0] == d]: pnl_delta += pay
    cum += pnl_delta; hedge_pnl_curve.append(cum)

plt.figure(figsize=(13, 7))
ax1 = plt.gca()
ax1.plot(dates, prices, label="QQQ Open")
if hedge_buy_events:
    hd = [d for d,_,_,_,_ in hedge_buy_events]; hp = [qqq_df.loc[d, "Open"] for d in hd]
    ax1.scatter(hd, hp, marker="o", s=48, label="Hedge PUT Buy")
if hedge_expire_events:
    ed_itm = [d for d,_,_,_,_,itm,_ in hedge_expire_events if itm]; ep_itm = [qqq_df.loc[d, "Open"] for d in ed_itm]
    ed_otm = [d for d,_,_,_,_,itm,_ in hedge_expire_events if not itm]; ep_otm = [qqq_df.loc[d, "Open"] for d in ed_otm]
    if ed_itm: ax1.scatter(ed_itm, ep_itm, marker="*", s=90, label="Hedge Expire ITM")
    if ed_otm: ax1.scatter(ed_otm, ep_otm, marker="x", s=60, label="Hedge Expire OTM")
ax1.set_xlabel("Date"); ax1.set_ylabel("QQQ Price (USD)")
ax1.grid(True); ax1.legend(loc="upper left")
ax2 = ax1.twinx()
ax2.plot(dates, hedge_pnl_curve, label="Cumulative Hedge P&L", linestyle="--")
ax2.set_ylabel("Cumulative Hedge P&L (USD)"); ax2.legend(loc="upper right")
sig_scaled = np.array(sig_vals, dtype=float)
if sig_scaled.max() > 0:
    sig_scaled = sig_scaled / sig_scaled.max() * max(1.0, np.nanmax(hedge_pnl_curve) * 0.2 if len(hedge_pnl_curve) else 1.0)
    ax2.plot(dates, sig_scaled, label="PUT Signal (scaled)", alpha=0.6); ax2.legend(loc="lower right")
plt.title(f"Hedge Overview  | Window: {window_str}")
plt.tight_layout(); plt.savefig(hedge_png, dpi=150, bbox_inches="tight"); plt.savefig(hedge_pdf, bbox_inches="tight"); plt.close()

html = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>Hedged Strategy Report</title></head>
<body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial;max-width:1000px;margin:40px auto;">
  <h1>Hedged Strategy — Covered Strangle + TQQQ(FG) + PUT Hedge (stacked)</h1>
  <p><strong>Backtest Window:</strong> {window_str}</p>
  <pre style="background:#f6f8fa;padding:16px;border-radius:8px;white-space:pre-wrap;">{summary}</pre>

  <h2>Main Equity Curves</h2>
  <figure><img src="{main_png.name}" style="max-width:100%;height:auto;"><figcaption>{main_png.name}</figcaption></figure>

  <h2>TQQQ Buy/Sell Events</h2>
  <figure><img src="{tqqq_png.name}" style="max-width:100%;height:auto;"><figcaption>{tqqq_png.name}</figcaption></figure>

  <h2>Hedge Overview</h2>
  <figure><img src="{hedge_png.name}" style="max-width:100%;height:auto;"><figcaption>{hedge_png.name}</figcaption></figure>

  <h2>Short / Hedge PUT — Yearly Stats</h2>
  <figure><img src="{yr_png.name}" style="max-width:100%;height:auto;"><figcaption>{yr_png.name}</figcaption></figure>

  <p>Equity curves: {eq_csv.name}</p>
  <p>TQQQ trades: {tqqq_trades_csv.name}</p>
  <p>Hedge trades: {hedge_trades_csv.name}</p>
  <p>Yearly stats: {yr_csv.name}</p>
  <p>Log: {log_path.name}</p>
</body>
</html>"""
with open(html_path, "w", encoding="utf-8") as f: f.write(html)

print(f"Figures saved: {main_png}, {tqqq_png}, {hedge_png}, {yr_png}")
print(f"HTML report saved: {html_path}")


