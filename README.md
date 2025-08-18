<h1 align="center">Covered Options Backtests (QQQ)</h1>
<p align="center"><em>Layered strategies → clear comparisons → reproducible pipeline</em></p>

<p align="center">
  <a href="https://github.com/yudonglu1136/Covered_Call_Backtest/actions">
    <img alt="Backtest CI"
         src="https://img.shields.io/github/actions/workflow/status/yudonglu1136/Covered_Call_Backtest/run-covered-call.yml?label=Backtest%20CI&logo=githubactions">
  </a>
  <img alt="Python"
       src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg"></a>
</p>

<p align="center">
  <a href="#highlights">Highlights</a> •
  <a href="#strategy-ladder">Strategy ladder</a> •
  <a href="#results-preview">Results</a> •
  <a href="#setup">Setup</a> •
  <a href="#data-pipeline">Data pipeline</a> •
  <a href="#run-backtests">Run backtests</a>
</p>

---

## Highlights
- **Progressive design**: start from covered call and add one feature at a time (CSP → TQQQ overlay → hedge).
- **Deterministic rules**: IV→Δ targeting, 106% strike floor, CSP 1–3 DTE, hedge 28–31 DTE.
- **Reproducible**: a one-shot **pipeline** to fetch/update all datasets and generate signals.
- **Artifacts**: charts, CSVs, logs, and HTML reports saved under `output/**`.

> [!TIP]
> GitHub 的 README 

## Strategy ladder
| Level | Strategy | File | Adds on top of previous |
| :--: | :-- | :-- | :-- |
| 1 | **Covered Call** | `covered_call.py` | Sell CC (15–18 DTE) with IV→Δ target & **106% floor**; DCA & dividends. |
| 2 | **Covered Call Strangle** | `covered_call_strangle.py` | + Cash-secured PUT (1–3 DTE) to (re)build 100-lot; fully collateralized. |
| 3 | **CC Strangle + TQQQ** | `covered_call_strangle_with_TQQQ.py` | + **TQQQ overlay** when **F&G < 15**, integer-share buys, **+100%** TP per batch. |
| 4 | **Hedged Strategy** | `hedged_strategy.py` | + Protective **ATM PUT** (28–31 DTE) on `put_signal == 1`, track hedge P&L. |

> In our runs, **Level 3 (CC Strangle + TQQQ)** often shows the best risk/return trade-off (window-dependent).

## Results preview
<p align="center">

  <img src="output/covered_call_strangle_TQQQ/strategy_comparison_2025-08-15.png" width="820" alt="Strategy vs DCA">
</p>

<p align="center">
  <img src="output/covered_call_strangle_TQQQ/tqqq_trades_2025-08-15.png" width="820" alt="TQQQ trades">
</p>


## Setup

### 1) Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt  # pandas, numpy, yfinance, pandas_market_calendars, requests, scipy, matplotlib, python-dotenv, etc.
