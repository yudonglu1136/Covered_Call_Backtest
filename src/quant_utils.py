# src/quant_utils.py
# -*- coding: utf-8 -*-
"""
Quant utility functions shared across backtest scripts.

Functions:
- normalize_date_series: normalize datetime-like series to midnight (tz-naive)
- iv_to_delta: map IV to a target call delta via logistic curve
- price_on_or_before: get Open price on a timestamp, or nearest prior session
- shares_affordable_for_put: max CSP contracts affordable with given cash/strike
- sharpe_ratio: annualized Sharpe from an equity curve

All functions are dependency-light and safe to import from any script.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import pandas as pd


def normalize_date_series(s: pd.Series) -> pd.Series:
    """
    Return a Timestamp-normalized (00:00, tz-naive) date series.

    Parameters
    ----------
    s : pd.Series
        Any datetime-like series (str/np.datetime64/pd.Timestamp).

    Returns
    -------
    pd.Series
        tz-naive midnight timestamps (NaT for unparsable values).
    """
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def iv_to_delta(
    iv: float,
    steepness: float = 10.0,
    mid: float = 0.27,
    min_delta: float = 0.018,
    max_delta: float = 0.15,
) -> float:
    """
    Map implied volatility (IV) to a target call delta via a logistic curve.

    The curve is centered around `mid` with slope `steepness`, and output is
    clamped in [min_delta, max_delta].

    Notes
    -----
    - Designed to choose covered-call moneyness adaptively with IV.

    Returns
    -------
    float
        Target delta rounded to 4 decimals.
    """
    ivf = float(iv)
    norm = 1.0 / (1.0 + math.exp(-steepness * (mid - ivf)))
    delta_val = min_delta + (max_delta - min_delta) * norm
    return round(float(delta_val), 4)


def price_on_or_before(
    idx_price: pd.DataFrame,
    ts: pd.Timestamp,
    current_fallback: Union[int, float],
    price_col: str = "Open",
) -> float:
    """
    Get price on the exact timestamp if present; else the closest *prior*
    trading day's price; else fall back to `current_fallback`.

    Parameters
    ----------
    idx_price : pd.DataFrame
        Must be indexed by normalized pd.Timestamp and contain `price_col`.
    ts : pd.Timestamp
        Target timestamp (normalized to midnight preferred).
    current_fallback : float
        Price to return if `ts` is earlier than the first index value.
    price_col : str, default "Open"
        Column name to read price from.

    Returns
    -------
    float
    """
    if ts in idx_price.index:
        return float(idx_price.loc[ts, price_col])

    # all dates <= ts
    earlier = idx_price.index[idx_price.index <= ts]
    if len(earlier) > 0:
        return float(idx_price.loc[earier[-1], price_col])  # type: ignore[name-defined]

    return float(current_fallback)


def shares_affordable_for_put(cash_free: float, strike: float) -> int:
    """
    Max number of cash-secured PUT contracts affordable given free cash and strike.

    Parameters
    ----------
    cash_free : float
        Unencumbered cash available for collateral.
    strike : float
        Option strike.

    Returns
    -------
    int
        Floor(cash_free / (strike * 100)).
    """
    if strike <= 0:
        return 0
    return int(cash_free // (strike * 100.0))


def sharpe_ratio(
    equity_curve: pd.Series,
    rf_annual: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio from an equity curve sampled at a fixed frequency.

    Parameters
    ----------
    equity_curve : pd.Series
        Portfolio/equity time series (indexed by date). Must be positive.
    rf_annual : float, default 0.0
        Annualized risk-free rate (in decimal, e.g., 0.02 = 2%).
    periods_per_year : int, default 252
        Sampling periodicity (252 for trading days).

    Returns
    -------
    float
        Annualized Sharpe ratio. Returns 0.0 if insufficient data or zero std.
    """
    rets = equity_curve.pct_change().dropna()
    if rets.empty:
        return 0.0

    rf_per_period = rf_annual / float(periods_per_year)
    excess = rets - rf_per_period
    std = excess.std()

    if std == 0 or np.isnan(std):
        return 0.0

    return float(excess.mean() / std * np.sqrt(periods_per_year))
