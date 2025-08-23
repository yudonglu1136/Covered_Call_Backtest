# Re-run a clean end-to-end job with the fixed overlap and produce outputs
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("DISPLAY","")=="" :
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path("data")
OUT_DIR = Path('output')/ "atm_call_cc_vs_stock_study"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Load QQQ ---
qqq = pd.read_csv(DATA_DIR/"QQQ_ohlcv_1d.csv")
qqq["date"] = pd.to_datetime(qqq["date"] if "date" in qqq.columns else qqq.iloc[:,0]).dt.normalize()
close_col = "close" if "close" in qqq.columns else qqq.columns[1]
qqq = qqq[["date", close_col]].rename(columns={close_col:"close"}).sort_values("date")
spot_map = qqq.set_index("date")["close"]
qqq_idx = qqq.set_index("date")

for n in [1,15,30]:
    qqq[f"ret_{n}d"] = qqq["close"].shift(-n) / qqq["close"] - 1.0

# --- Load options ---
opt = pd.read_csv(DATA_DIR/"options_with_iv_delta.csv")
opt["date"] = pd.to_datetime(opt["date"]).dt.normalize()
opt["expiration"] = pd.to_datetime(opt["expiration"]).dt.normalize()
opt["dte"] = (opt["expiration"] - opt["date"]).dt.days
opt["right"] = opt["type"].str.upper().str[0]  # 'C' or 'P'

def mid_price_row(r):
    v = np.nan
    if "h" in r and "l" in r and pd.notna(r["h"]) and pd.notna(r["l"]):
        v = (float(r["h"]) + float(r["l"])) / 2.0
    if (pd.isna(v)) and ("c" in r) and pd.notna(r["c"]): v = float(r["c"])
    if (pd.isna(v)) and ("o" in r) and pd.notna(r["o"]): v = float(r["o"])
    return v

opt["mid_price"] = opt.apply(mid_price_row, axis=1)
opt["spot"] = pd.to_numeric(opt.get("underlying_price", np.nan), errors="coerce")
opt.loc[opt["spot"].isna(), "spot"] = opt.loc[opt["spot"].isna(), "date"].map(spot_map)

# 仅保留 "看涨" 且价格/标的有效
opt = opt[(opt["right"]=="C") & opt["mid_price"].notna() & opt["spot"].notna()].copy()
opt["strike"] = pd.to_numeric(opt["strike"], errors="coerce"); opt = opt.dropna(subset=["strike"])

# DTE 桶不变
BUCKETS = {"DTE_1_3":(1,3), "DTE_15_18":(15,18), "DTE_28P":(28,None)}

records = []
for d, S0 in spot_map.items():
    day_slice = opt[opt["date"]==d]
    if day_slice.empty: 
        continue
    for name,(lo,hi) in BUCKETS.items():
        if hi is None:
            pool = day_slice[day_slice["dte"]>=lo]; tgt_dte = lo
        else:
            pool = day_slice[(day_slice["dte"]>=lo)&(day_slice["dte"]<=hi)]; tgt_dte = int(round((lo+hi)/2))
        if pool.empty: 
            continue
        pool = pool.copy()
        pool["abs_diff"] = (pool["strike"] - S0).abs()
        pool["dte_dist"] = (pool["dte"] - tgt_dte).abs()
        atm = pool.sort_values(["abs_diff","dte_dist"]).head(1).iloc[0]

        K = float(atm["strike"]); prem = float(atm["mid_price"]); dte = int(atm["dte"]); expiry = atm["expiration"]
        ST = qqq_idx["close"].get(expiry, np.nan)
        if pd.isna(ST): 
            continue

        # 覆盖式 Call：每股收益 = min(ST, K) - S0 + premium
        pnl_per_share = min(ST, K) - S0 + prem
        ret_cc = pnl_per_share / S0

        assigned_itm = ST > K
        below_breakeven = ret_cc < 0.0

        # map stock returns for the same start date（保持与原方法一致）
        r1  = qqq.loc[qqq["date"]==d, "ret_1d"].values[0]  if (qqq["date"]==d).any() else np.nan
        r15 = qqq.loc[qqq["date"]==d, "ret_15d"].values[0] if (qqq["date"]==d).any() else np.nan
        r30 = qqq.loc[qqq["date"]==d, "ret_30d"].values[0] if (qqq["date"]==d).any() else np.nan

        records.append({
            "bucket": name, "trade_date": d, "expiry": expiry, "dte": dte,
            "spot_trade": S0, "strike": K, "premium": prem, "spot_expiry": ST,
            "assigned_itm": bool(assigned_itm), "below_breakeven": bool(below_breakeven),
            "return_dec": ret_cc,
            "QQQ_ret_1d": r1, "QQQ_ret_15d": r15, "QQQ_ret_30d": r30
        })

results = pd.DataFrame(records).sort_values(["trade_date","bucket"]).reset_index(drop=True)

# summaries
summary_opt = (results.groupby("bucket")
               .agg(n=("return_dec","count"),
                    mean_return=("return_dec","mean"),
                    std_return=("return_dec","std"),
                    assigned_rate=("assigned_itm","mean"),
                    below_breakeven_rate=("below_breakeven","mean"))
               .reset_index().sort_values("bucket"))

# stock summaries（保持与原脚本一致：使用 options 有数据那些天的 1/15/30d 先行收益）
stock_stats = []
for label,col in [("QQQ_1d","QQQ_ret_1d"),("QQQ_15d","QQQ_ret_15d"),("QQQ_30d","QQQ_ret_30d")]:
    s = results[col].dropna()
    stock_stats.append({"horizon":label, "n":int(s.shape[0]), "mean_return":float(s.mean()), "std_return":float(s.std(ddof=1))})
summary_stock = pd.DataFrame(stock_stats)

# plots: two rows x three cols
fig, axes = plt.subplots(2,3, figsize=(18,10), sharey=False)
bucket_order = ["DTE_1_3","DTE_15_18","DTE_28P"]
stock_order = [("QQQ_1d","QQQ_ret_1d"),("QQQ_15d","QQQ_ret_15d"),("QQQ_30d","QQQ_ret_30d")]

# row1 options（ATM CALL covered）
for ax, b in zip(axes[0], bucket_order):
    sub = results[results["bucket"]==b]
    if sub.empty:
        ax.set_title(f"{b} (no data)"); ax.axis("off"); continue
    ax.hist(sub["return_dec"].values, bins=60, alpha=0.9)
    ax.set_title(f"Options (ATM CALL) {b}")
    ax.set_xlabel("Return (decimal)"); ax.set_ylabel("Freq"); ax.grid(True, linestyle="--", alpha=0.4)

# row2 stock
for ax, (label,col) in zip(axes[1], stock_order):
    s = results[col].dropna()
    if s.empty:
        ax.set_title(f"{label} (no data)"); ax.axis("off"); continue
    ax.hist(s.values, bins=60, alpha=0.9)
    ax.set_title(f"Stock {label}")
    ax.set_xlabel("Return (decimal)"); ax.set_ylabel("Freq"); ax.grid(True, linestyle="--", alpha=0.4)

plt.tight_layout()
fig_path = OUT_DIR / "histograms_options_vs_stock.png"
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
plt.show()

print("== OPTIONS (ATM CALL, covered) return summary ==")
for _, row in summary_opt.iterrows():
    print(f"{row['bucket']:10s} | n={int(row['n']):4d} | mean={row['mean_return']:.4f} | std={row['std_return']:.4f} | "
          f"assigned_rate={row['assigned_rate']:.3f} | below_breakeven_rate={row['below_breakeven_rate']:.3f}")

print("\n== STOCK (QQQ) forward return summary (aligned days) ==")
for _, row in summary_stock.iterrows():
    print(f"{row['horizon']:8s} | n={int(row['n']):4d} | mean={row['mean_return']:.4f} | std={row['std_return']:.4f}")

print(f"\nSaved:\n  {fig_path}")
