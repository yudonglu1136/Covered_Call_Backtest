import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
DATA_DIR = Path("data")

# === Load data ===
qqq = pd.read_csv(DATA_DIR / "QQQ_ohlcv_1d.csv", parse_dates=["date"]).sort_values("date")
options = pd.read_csv(DATA_DIR / "options_with_iv_delta.csv", parse_dates=["date","expiration"])

qqq["date"] = qqq["date"].dt.normalize()
options["date"] = options["date"].dt.normalize()
options["expiration"] = options["expiration"].dt.normalize()

# === Gap Up / Down ===
qqq["prev_close"] = qqq["close"].shift(1)
qqq["gap_label"] = None
qqq.loc[qqq["open"] > qqq["prev_close"], "gap_label"] = "Gap Up"
qqq.loc[qqq["open"] < qqq["prev_close"], "gap_label"] = "Gap Down"

# merge
opts = options.merge(qqq[["date","gap_label"]], on="date", how="left").dropna(subset=["gap_label"])

rows = []
for (dt, gl), g in opts.groupby(["date","gap_label"], sort=False):
    under = g["underlying_price"].iloc[0]
    days = (g["expiration"] - dt).dt.days
    calls = g[(g["type"]=="call") & (days.between(28,31))]

    if calls.empty: continue
    sel5 = calls["strike"] >= under*1.05
    sel7 = calls["strike"] >= under*1.07

    rows.append({
        "date": dt, "gap_label": gl,
        "call5": calls.loc[sel5,"o"].mean(),
        "call7": calls.loc[sel7,"o"].mean()
    })

summary = pd.DataFrame(rows)

# === Long format for plotting ===
long_prices = summary.melt(
    id_vars=["date","gap_label"],
    value_vars=["call5","call7"],
    var_name="option_type",
    value_name="price"
).dropna()

# === Plot: 3 subplots side by side ===
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# (1) KDE curves
sns.kdeplot(data=long_prices, x="price", hue="gap_label",
            multiple="layer", common_norm=False, alpha=0.6, linewidth=2, ax=axes[0])
axes[0].set_title("KDE of OTM Call Prices (28–31D)", fontsize=12)
axes[0].set_xlabel("Opening Price"); axes[0].set_ylabel("Density")

# (2) Histogram for 5% OTM
sns.histplot(
    data=long_prices[long_prices["option_type"]=="call5"],
    x="price", hue="gap_label", bins=50, kde=True,
    element="step", stat="density", common_norm=False, ax=axes[1]
)
axes[1].set_title("5% OTM Call Prices", fontsize=12)
axes[1].set_xlabel("Opening Price"); axes[1].set_ylabel("Density")

# (3) Histogram for 7% OTM
sns.histplot(
    data=long_prices[long_prices["option_type"]=="call7"],
    x="price", hue="gap_label", bins=50, kde=True,
    element="step", stat="density", common_norm=False, ax=axes[2]
)
axes[2].set_title("7% OTM Call Prices", fontsize=12)
axes[2].set_xlabel("Opening Price"); axes[2].set_ylabel("Density")

plt.tight_layout()
plt.show()
