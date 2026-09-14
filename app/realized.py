"""Realized (historical) volatility from daily closes.

The other half of the chart. Unlike option chains, price history is free and
deep for any ticker, so this side of "implied vs realized" is full-length from
day one even when the implied side has a single point.

Two windows are computed:

  trailing   the vol the stock *has* shown over the past N days. This is what
             the original IV.py plotted, and what a trader sees on screen.
  forward    the vol the stock *went on* to show over the next N days, aligned
             to the day the implied vol was quoted. This is the honest
             comparison: implied vol on day t is a forecast, and the forecast is
             scored against what happened after t, not before it. It is NaN for
             the most recent N days because that future has not happened yet.

Convention: log returns per trading day, sample std, annualized by sqrt(252).
The 252 is correct here because the observations are trading days -- the same
day-count question ivlib.bs settles for T, answered consistently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
WINDOW = 21   # trading days ~ 30 calendar days, matching the 30d implied series


def log_returns(close: pd.Series) -> pd.Series:
    close = close.astype(float).sort_index()
    return np.log(close / close.shift(1)).dropna()


def trailing(close: pd.Series, window: int = WINDOW) -> pd.Series:
    r = log_returns(close)
    out = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    out.name = f"rv{window}_trailing"
    return out


def forward(close: pd.Series, window: int = WINDOW) -> pd.Series:
    """Std of the *next* `window` returns, stamped on the day before they start."""
    r = log_returns(close)
    fwd = r[::-1].rolling(window).std(ddof=1)[::-1].shift(-1) * np.sqrt(TRADING_DAYS)
    fwd.name = f"rv{window}_forward"
    return fwd


def both(close: pd.Series, window: int = WINDOW) -> pd.DataFrame:
    return pd.concat([trailing(close, window), forward(close, window)], axis=1)
