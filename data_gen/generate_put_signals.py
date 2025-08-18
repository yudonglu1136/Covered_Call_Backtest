# daily_divergence_with_adx_fixed.py  (place under data_gen/)
import ssl
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

# Make pandas behavior explicit
pd.set_option('future.no_silent_downcasting', True)

# ========= Path bases (script under data_gen) =========
HERE = Path(__file__).resolve().parent            # .../data_gen
ROOT = HERE.parent                                # project root
DATA_DIR = ROOT / "data"                          # input data and signal CSV outputs
OUTPUT_BASE = HERE / "output"                     # figures/reports saved under data_gen/output
(OUTPUT_BASE / "put_signals").mkdir(parents=True, exist_ok=True)

# ========= Optional: bypass SSL verification (debug only) =========
DISABLE_SSL_VERIFY = False
if DISABLE_SSL_VERIFY:
    ssl._create_default_https_context = ssl._create_unverified_context

# ===================== Parameters =====================
START_DATE   = "2018-01-01"

# —— Trigger: price first-breakout + breadth lags —— #
LB        = 60        # lookback window (days)
PB        = 0.01      # price breakout threshold vs previous rolling max (+1.0%)
BGAP      = 0.003     # breadth lag threshold vs previous breadth max (-0.3%)
COOLDOWN  = 5         # minimum calendar-day spacing between triggers

# —— ADX filters —— #
ADX_PERIOD          = 14
ADX_WEAK_TH         = 25
ADX_STRONG_TH       = 30
USE_STRONG_SUPPRESS = True
USE_ADX_WEAK        = True
USE_ADX_FALLING     = True
ADX_FALLING_N       = 3

# —— Execution: structural confirmation —— #
EMA_EXEC  = 10       # must close below EMA within K days
K         = 4

# —— MA60 filter (use PRIOR day’s MA to avoid look-ahead) —— #
MA_N        = 60
FILTER_ON   = "trigger"  # "trigger" or "hit"
PLOT_MA60   = True
PRINT_DEBUG = False

# —— Only accept signals close to ATH —— #
ATH_PROX_PCT  = 0.10          # accept if price >= (1-0.05)*ATH up to the reference day
ATH_FILTER_ON = "trigger"     # "trigger" or "hit": which day to check proximity

# ===================== Helpers =====================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def compute_adx(df_ohlc: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    for col in ("High", "Low", "Close"):
        if col not in df_ohlc.columns:
            raise ValueError(f"compute_adx() requires column: {col}")
    high = df_ohlc["High"].astype(float)
    low  = df_ohlc["Low"].astype(float)
    close= df_ohlc["Close"].astype(float)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    up_move   = (high - high.shift(1)).to_numpy()
    down_move = (low.shift(1) - low).to_numpy()
    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0),  up_move, 0.0).ravel(),
                         index=df_ohlc.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0).ravel(),
                         index=df_ohlc.index)

    tr_n       = tr.rolling(n).sum()
    plus_dm_n  = plus_dm.rolling(n).sum()
    minus_dm_n = minus_dm.rolling(n).sum()

    plus_di  = 100.0 * (plus_dm_n / tr_n)
    minus_di = 100.0 * (minus_dm_n / tr_n)
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)) * 100.0
    adx = dx.rolling(n).mean()

    return pd.DataFrame({"+DI": plus_di, "-DI": minus_di, "ADX": adx}, index=df_ohlc.index)

def normalize_index(idx) -> pd.DatetimeIndex:
    di = pd.DatetimeIndex(pd.to_datetime(idx))
    try:
        di = di.tz_convert(None)
    except Exception:
        try:
            di = di.tz_localize(None)
        except Exception:
            pass
    return di.normalize()

# ===================== 1) S&P 500 components & A/D Line =====================
spx_table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
tickers = [t.replace('.', '-') for t in spx_table['Symbol'].astype(str).tolist()]
print(f"S&P 500 symbols loaded: {len(tickers)}")

spx = yf.download(
    tickers,
    start=START_DATE,
    group_by="ticker",
    progress=False,
    auto_adjust=True,
    timeout=30,
    threads=False,
)

close_matrix = {}
available = set(spx.columns.get_level_values(0)) if isinstance(spx.columns, pd.MultiIndex) else set()
for t in tickers:
    if t in available:
        s = spx[t]["Adj Close"] if "Adj Close" in spx[t].columns else spx[t]["Close"]
        s.index = normalize_index(s.index)
        close_matrix[t] = s

adj_close = pd.DataFrame(close_matrix).dropna(how="all").dropna(axis=1, how="all")
adj_close.index = normalize_index(adj_close.index)

prev = adj_close.shift(1)
up_count   = (adj_close.gt(prev)).sum(axis=1)
down_count = (adj_close.lt(prev)).sum(axis=1)
breadth    = pd.Series((up_count - down_count).cumsum(), name="A/D Line")
breadth.index = normalize_index(breadth.index)

# ===================== 2) QQQ price & ADX =====================
qqq_raw = yf.download("QQQ", start=breadth.index.min().strftime("%Y-%m-%d"),
                      progress=False, auto_adjust=True, timeout=30)
qqq_raw.index = normalize_index(qqq_raw.index)

qqq_close = (qqq_raw["Adj Close"] if "Adj Close" in qqq_raw.columns else qqq_raw["Close"])
qqq_close = qqq_close.reindex(breadth.index).ffill()

qqq_ohlc  = qqq_raw[["High","Low","Close"]].reindex(breadth.index).ffill()
adx_df    = compute_adx(qqq_ohlc, n=ADX_PERIOD)
adx       = adx_df["ADX"]

# ===================== 3) Trigger: first breakout + breadth lag + cooldown =====================
p_rollmax_prev = qqq_close.shift(1).rolling(LB, min_periods=LB).max()
b_rollmax_prev = breadth.shift(1).rolling(LB, min_periods=LB).max()

price_above    = (qqq_close > p_rollmax_prev * (1 + PB)).astype(bool)
price_breakout = (price_above & (~price_above.shift(1).fillna(False).astype(bool))).astype(bool)

breadth_lag = (breadth <= b_rollmax_prev * (1 - BGAP)).astype(bool)
raw_trig    = (price_breakout & breadth_lag).fillna(False).astype(bool)

# cooldown
trigger_raw = pd.Series(False, index=raw_trig.index)
last = None
for dt in raw_trig[raw_trig].index:
    if (last is None) or ((dt - last).days > COOLDOWN):
        trigger_raw.loc[dt] = True
        last = dt

# ===================== 4) ADX filters =====================
mask = trigger_raw.copy().astype(bool)

if USE_STRONG_SUPPRESS:
    strong_up = ((adx > ADX_STRONG_TH) & (adx_df["+DI"] > adx_df["-DI"])).reindex(mask.index).fillna(False)
    mask = (mask & (~strong_up)).astype(bool)

if USE_ADX_WEAK:
    weak_ok = (adx < ADX_WEAK_TH).reindex(mask.index).fillna(False)
    mask = (mask & weak_ok).astype(bool)

if USE_ADX_FALLING:
    falling_n = (adx.diff() < 0).rolling(ADX_FALLING_N).sum().ge(ADX_FALLING_N)
    falling_n = falling_n.reindex(mask.index).fillna(False)
    mask = (mask & falling_n).astype(bool)

trigger_mask = mask.astype(bool)

# ===================== 5) Execution: must close below EMA within K days =====================
def ema(series: pd.Series, span: int) -> pd.Series:  # re-define local to avoid accidental import mismatch
    return series.ewm(span=span, adjust=False).mean()

ema_exec   = ema(qqq_close, EMA_EXEC)
below_exec = (qqq_close < ema_exec).astype(bool)

final_mask = pd.Series(False, index=qqq_close.index)
hit_by_trigger = {}  # {hit_date: trigger_date}

for d in trigger_mask[trigger_mask].index:
    win = below_exec.loc[d : d + pd.Timedelta(days=K)]
    hit = win[win].index.min()
    if pd.notna(hit):
        final_mask.loc[hit] = True
        hit_by_trigger[hit] = d

orig_hits = list(hit_by_trigger.keys())
print(f"[Fixed Divergence + ADX] Executed signals (before MA60): {len(orig_hits)}")

# ===================== 6) MA60 filter (previous day MA) =====================
ma60          = qqq_close.rolling(MA_N, min_periods=MA_N).mean()
ma60_prev     = ma60.shift(1)
valid_ma_prev = (qqq_close.rolling(MA_N).count() >= MA_N).shift(1).fillna(False)

ok_prev = (valid_ma_prev & (qqq_close >= ma60_prev))
if isinstance(ok_prev, pd.DataFrame):
    ok_prev = ok_prev.iloc[:, 0]
ok_prev.index = normalize_index(ok_prev.index)
ok_prev = ok_prev.astype(bool)

def ok_at(date) -> bool:
    d = pd.Timestamp(date).normalize()
    v = ok_prev.get(d, False)
    if isinstance(v, (pd.Series, np.ndarray, list, tuple)):
        return bool(np.any(v))
    return bool(v)

if FILTER_ON.lower() == "trigger":
    kept_hits = [h for h, d in hit_by_trigger.items() if ok_at(d)]
else:  # "hit"
    kept_hits = [h for h in orig_hits if ok_at(h)]

# ===================== 6.5) ATH proximity filter =====================
ath_to_date = qqq_close.cummax()

def near_ath(day: pd.Timestamp) -> bool:
    px  = float(qqq_close.get(day, np.nan))
    ath = float(ath_to_date.get(day, np.nan))
    if np.isnan(px) or np.isnan(ath) or ath <= 0:
        return False
    return px >= (1.0 - ATH_PROX_PCT) * ath

if ATH_FILTER_ON.lower() == "trigger":
    kept_hits = [h for h in kept_hits if near_ath(hit_by_trigger.get(h, h))]
else:
    kept_hits = [h for h in kept_hits if near_ath(h)]

filtered_out = set(orig_hits) - set(kept_hits)
print(f"[ATH filter] accepted within {ATH_PROX_PCT:.1%} of ATH on {ATH_FILTER_ON}: {len(kept_hits)} (filtered out: {len(filtered_out)})")

# Rebuild final mask with ATH filter applied
final_mask = pd.Series(False, index=qqq_close.index)
final_mask.loc[kept_hits] = True

signal_dates = pd.Index(kept_hits).sort_values()
print(f"[Final] Executed signals after all filters: {len(signal_dates)}")

# ===================== 7) Visualization (3-panel diagnostic) =====================
plt.figure(figsize=(11, 10))

ax1 = plt.subplot(3,1,1)
ax1.plot(qqq_close.index, qqq_close.values, label="QQQ (Close)")
ax1.scatter(signal_dates, qqq_close.reindex(signal_dates), marker='v', s=70,
            label=f"Final Exit (LB={LB}, PB={PB*100:.1f}%, BGAP={BGAP*100:.1f}%, CD={COOLDOWN}, EMA{EMA_EXEC}, K={K}, MA{MA_N}(prev)@{FILTER_ON}, ATH±{ATH_PROX_PCT*100:.0f}% on {ATH_FILTER_ON})")
if PLOT_MA60:
    ax1.plot(ma60.index, ma60.values, linestyle="--", alpha=0.7, label=f"MA{MA_N}")
ax1.set_title("QQQ vs S&P500 A/D — First-Break Divergence + ADX + EMA Execution + MA(prev) + ATH Proximity")
ax1.set_ylabel("QQQ Price")
ax1.grid(alpha=0.3)
ax1.legend(loc="upper left")

ax2 = plt.subplot(3,1,2, sharex=ax1)
ax2.plot(breadth.index, breadth.values, label="S&P 500 A/D Line", linewidth=1.1)
ax2.set_ylabel("A/D Line (Cumulative)")
ax2.grid(alpha=0.3)
ax2.legend(loc="upper left")

ax3 = plt.subplot(3,1,3, sharex=ax1)
ax3.plot(adx.index, adx.values, label=f"ADX ({ADX_PERIOD})", linewidth=1.1)
ax3.axhline(ADX_WEAK_TH,   linestyle="--", alpha=0.5, label=f"ADX weak={ADX_WEAK_TH}")
ax3.axhline(ADX_STRONG_TH, linestyle="--", alpha=0.5, label=f"ADX strong={ADX_STRONG_TH}")
ax3.set_ylabel("ADX")
ax3.set_xlabel("Date")
ax3.grid(alpha=0.3)
ax3.legend(loc="upper left")

plt.tight_layout()

# ===================== 8) Save signals to CSV (robust when zero signals) =====================
OUTPUT_EVENTS_CSV = DATA_DIR / "signals_events.csv"
OUTPUT_SERIES_CSV = DATA_DIR / "put_signals.csv"

events_cols = [
    "symbol", "signal", "hit_date", "trigger_date",
    "close_at_hit", "ema_exec_at_hit", "ma60_prev_at_hit", "ge_ma60_prev_at_hit",
    "adx_at_trigger", "plusDI_at_trigger", "minusDI_at_trigger",
    "ath_filter_on", "ath_to_date", "close_on_ref", "pct_from_ath_on_ref",
    "LB","PB","BGAP","COOLDOWN",
    "ADX_PERIOD","ADX_WEAK_TH","ADX_STRONG_TH","USE_STRONG_SUPPRESS","USE_ADX_WEAK","USE_ADX_FALLING","ADX_FALLING_N",
    "EMA_EXEC","K","MA_N","FILTER_ON","ATH_PROX_PCT","ATH_FILTER_ON","generated_at"
]

rows = []
for h in sorted(signal_dates):
    d = hit_by_trigger.get(h, pd.NaT)
    ath_ref_day = d if ATH_FILTER_ON.lower() == "trigger" else h
    rows.append({
        "symbol": "QQQ",
        "signal": "EXIT",
        "hit_date": pd.Timestamp(h).strftime("%Y-%m-%d"),
        "trigger_date": (pd.Timestamp(d).strftime("%Y-%m-%d") if pd.notna(d) else None),
        "close_at_hit": float(qqq_close.get(h, np.nan)),
        "ema_exec_at_hit": float(ema_exec.get(h, np.nan)),
        "ma60_prev_at_hit": float(ma60_prev.get(h, np.nan)),
        "ge_ma60_prev_at_hit": bool(ok_prev.get(h, False)),
        "adx_at_trigger": float(adx.get(d, np.nan)) if pd.notna(d) else np.nan,
        "plusDI_at_trigger": float(adx_df["+DI"].get(d, np.nan)) if pd.notna(d) else np.nan,
        "minusDI_at_trigger": float(adx_df["-DI"].get(d, np.nan)) if pd.notna(d) else np.nan,
        "ath_filter_on": ATH_FILTER_ON,
        "ath_to_date": float(qqq_close.cummax().get(ath_ref_day, np.nan)),
        "close_on_ref": float(qqq_close.get(ath_ref_day, np.nan)),
        "pct_from_ath_on_ref": float(
            qqq_close.get(ath_ref_day, np.nan) / qqq_close.cummax().get(ath_ref_day, np.nan) - 1.0
        ) if pd.notna(qqq_close.cummax().get(ath_ref_day, np.nan)) else np.nan,
        "LB": LB, "PB": PB, "BGAP": BGAP, "COOLDOWN": COOLDOWN,
        "ADX_PERIOD": ADX_PERIOD, "ADX_WEAK_TH": ADX_WEAK_TH, "ADX_STRONG_TH": ADX_STRONG_TH,
        "USE_STRONG_SUPPRESS": USE_STRONG_SUPPRESS, "USE_ADX_WEAK": USE_ADX_WEAK,
        "USE_ADX_FALLING": USE_ADX_FALLING, "ADX_FALLING_N": ADX_FALLING_N,
        "EMA_EXEC": EMA_EXEC, "K": K,
        "MA_N": MA_N, "FILTER_ON": FILTER_ON,
        "ATH_PROX_PCT": ATH_PROX_PCT, "ATH_FILTER_ON": ATH_FILTER_ON,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

# Create with columns and sort only if non-empty
events_df = pd.DataFrame(rows, columns=events_cols)
if not events_df.empty:
    events_df = events_df.sort_values("hit_date")
events_df.to_csv(OUTPUT_EVENTS_CSV, index=False)
print(f"[Saved] {OUTPUT_EVENTS_CSV} -> {events_df.shape}")

# Full daily series (1/0)
series_df = pd.DataFrame({
    "date": qqq_close.index.strftime("%Y-%m-%d"),
    "signal": final_mask.astype(int).values
})
series_df.to_csv(OUTPUT_SERIES_CSV, index=False)
print(f"[Saved] {OUTPUT_SERIES_CSV} -> {series_df.shape}")

# ===================== 9) Standalone QQQ + PUT signal chart =====================
OUTPUT_PLOT_DIR = OUTPUT_BASE / "put_signals"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.plot(qqq_close.index, qqq_close.values, label="QQQ (Close)")

signal_on_idx = final_mask[final_mask].index
if len(signal_on_idx) > 0:
    ax.scatter(signal_on_idx, qqq_close.reindex(signal_on_idx).values,
               marker="o", s=64, label="PUT signal (generated)")

ax.set_title(f"QQQ with Generated PUT Signals (Only within ±{ATH_PROX_PCT*100:.0f}% of ATH on {ATH_FILTER_ON})")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

png_path = OUTPUT_PLOT_DIR / "qqq_put_signals.png"
pdf_path = OUTPUT_PLOT_DIR / "qqq_put_signals.pdf"
fig.tight_layout()
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"[Saved] {png_path} / {pdf_path}")

plt.show()


plt.tight_layout()

# ===================== 8) Save signals to CSV =====================
# Keep signal CSVs in root/data (not root/output), so other scripts can consume them
OUTPUT_EVENTS_CSV = DATA_DIR / "signals_events.csv"
OUTPUT_SERIES_CSV = DATA_DIR / "put_signals.csv"

rows = []
for h in sorted(signal_dates):
    d = hit_by_trigger.get(h, pd.NaT)
    ath_ref_day = d if ATH_FILTER_ON.lower() == "trigger" else h
    rows.append({
        "symbol": "QQQ",
        "signal": "EXIT",
        "hit_date": pd.Timestamp(h).strftime("%Y-%m-%d"),
        "trigger_date": (pd.Timestamp(d).strftime("%Y-%m-%d") if pd.notna(d) else None),
        "close_at_hit": float(qqq_close.get(h, np.nan)),
        "ema_exec_at_hit": float(ema_exec.get(h, np.nan)),
        "ma60_prev_at_hit": float(ma60_prev.get(h, np.nan)),
        "ge_ma60_prev_at_hit": bool(ok_prev.get(h, False)),
        "adx_at_trigger": float(adx.get(d, np.nan)) if pd.notna(d) else np.nan,
        "plusDI_at_trigger": float(adx_df["+DI"].get(d, np.nan)) if pd.notna(d) else np.nan,
        "minusDI_at_trigger": float(adx_df["-DI"].get(d, np.nan)) if pd.notna(d) else np.nan,
        # ATH proximity diagnostics
        "ath_filter_on": ATH_FILTER_ON,
        "ath_to_date": float(ath_to_date.get(ath_ref_day, np.nan)),
        "close_on_ref": float(qqq_close.get(ath_ref_day, np.nan)),
        "pct_from_ath_on_ref": float(qqq_close.get(ath_ref_day, np.nan) / ath_to_date.get(ath_ref_day, np.nan) - 1.0) if pd.notna(ath_to_date.get(ath_ref_day, np.nan)) else np.nan,
        # Params for provenance
        "LB": LB, "PB": PB, "BGAP": BGAP, "COOLDOWN": COOLDOWN,
        "ADX_PERIOD": ADX_PERIOD, "ADX_WEAK_TH": ADX_WEAK_TH, "ADX_STRONG_TH": ADX_STRONG_TH,
        "USE_STRONG_SUPPRESS": USE_STRONG_SUPPRESS, "USE_ADX_WEAK": USE_ADX_WEAK,
        "USE_ADX_FALLING": USE_ADX_FALLING, "ADX_FALLING_N": ADX_FALLING_N,
        "EMA_EXEC": EMA_EXEC, "K": K,
        "MA_N": MA_N, "FILTER_ON": FILTER_ON,
        "ATH_PROX_PCT": ATH_PROX_PCT, "ATH_FILTER_ON": ATH_FILTER_ON,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })

events_df = pd.DataFrame(rows).sort_values("hit_date")
events_df.to_csv(OUTPUT_EVENTS_CSV, index=False)
print(f"[Saved] {OUTPUT_EVENTS_CSV} -> {events_df.shape}")

series_df = pd.DataFrame({
    "date": qqq_close.index.strftime("%Y-%m-%d"),
    "signal": final_mask.astype(int).values
})
series_df.to_csv(OUTPUT_SERIES_CSV, index=False)
print(f"[Saved] {OUTPUT_SERIES_CSV} -> {series_df.shape}")

# ===================== 9) Standalone QQQ + PUT signal chart (figures -> data_gen/output/put_signals) =====================
OUTPUT_PLOT_DIR = OUTPUT_BASE / "put_signals"
OUTPUT_PLOT_DIR.mkdir(parents=True, exist_ok=True)

fig = plt.figure(figsize=(12, 6))
ax = plt.gca()
ax.plot(qqq_close.index, qqq_close.values, label="QQQ (Close)")

signal_on_idx = final_mask[final_mask].index
if len(signal_on_idx) > 0:
    ax.scatter(
        signal_on_idx,
        qqq_close.reindex(signal_on_idx).values,
        marker="o",
        s=64,
        label="PUT signal (generated)"
    )

ax.set_title(f"QQQ with Generated PUT Signals (Only within ±{ATH_PROX_PCT*100:.0f}% of ATH on {ATH_FILTER_ON})")
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.grid(alpha=0.3)
ax.legend(loc="upper left")

png_path = OUTPUT_PLOT_DIR / "qqq_put_signals.png"
pdf_path = OUTPUT_PLOT_DIR / "qqq_put_signals.pdf"
fig.tight_layout()
fig.savefig(png_path, dpi=150, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
print(f"[Saved] {png_path} / {pdf_path}")


