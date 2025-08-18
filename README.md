Covered Options Backtests (QQQ)

This repository implements a progressive ladder of options strategies on QQQ. Each script adds one layer of logic so you can compare risk/return step-by-step.

Strategies

Covered Call — covered_call.py

Hold QQQ in 100-share lots.

Systematically sell covered calls (15–18 DTE).

Call selection uses an IV→Δ targeting rule with a 106% strike floor.

Includes quarterly DCA contributions and dividend handling.

Covered Call Strangle — covered_call_strangle.py

Everything from Covered Call plus cash-secured puts (1–3 DTE) to (re)build the long QQQ position.

Fully collateralized (no over-selling). Handles put assignment and then re-selling calls.

Covered Call Strangle + TQQQ overlay — covered_call_strangle_with_TQQQ.py

Adds an opportunistic TQQQ overlay: when the Fear & Greed index < 15, buy integer shares with idle cash; take profit at +100% per batch.

Realized gains can help restore 100-share QQQ lots to continue call selling.

Currently the best performer among the three non-hedged variants in our runs (results vary by window/data).

Hedged Strategy — hedged_strategy.py

Builds on (3) and buys 1× ATM protective put (28–31 DTE) when a divergence-based put_signal == 1.

Tracks hedge premiums, expiries (ITM/OTM settlement), and cumulative hedge P&L.

Complexity intentionally builds layer by layer: Covered Call → +CSP → +TQQQ overlay → +Hedge. Check each strategy’s output folder for concrete metrics.