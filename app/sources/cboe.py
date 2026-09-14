"""Cboe single-stock volatility indices, via FRED.

Cboe computes VIX-methodology 30-day implied vol indices for a handful of large
single names and FRED republishes the daily closes. For those names this is
sixteen years of history -- 2010-06-01 onward -- at zero cost and without
needing a single option chain.

It is *not* the same quantity as `ivlib.surface.constant_maturity(term, 30)`.
The Cboe number is a VIX-methodology variance-strip rate: a 1/K^2-weighted sum
over the whole OTM strip, which prices in the skew. Ours is at-the-money-forward
vol. Under a skewed smile the strip rate sits above ATM vol, so expect the Cboe
series to run higher. Measured on AAPL over 561 overlapping days: correlation
0.978, level ratio (ours / Cboe) median 0.90. The direction is the standard
result; the magnitude is consistent with theory but not documented anywhere as
"typical", so treat the 0.90 as an observation, not a target.

What makes it a useful second check on the engine is the correlation, not the
level: the two series must move together, and they do.

Coverage is exactly these five. There is no way to add a sixth.
"""

from __future__ import annotations

import io
import urllib.request

import pandas as pd

SERIES = {
    "AAPL": "VXAPLCLS",
    "AMZN": "VXAZNCLS",
    "GOOG": "VXGOGCLS",
    "GS": "VXGSCLS",
    "IBM": "VXIBMCLS",
}
_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def available(ticker: str) -> bool:
    return ticker.upper() in SERIES


def fetch(ticker: str, timeout: int = 30) -> pd.Series:
    """Daily 30-day implied vol as a decimal (0.27, not 27), indexed by date.

    Returns an empty Series for tickers Cboe does not cover.
    """
    sid = SERIES.get(ticker.upper())
    if sid is None:
        return pd.Series(dtype="float64", name="cboe_iv30")
    raw = urllib.request.urlopen(_URL.format(sid=sid), timeout=timeout).read().decode()
    return parse(raw)


def parse(csv_text: str) -> pd.Series:
    """FRED's CSV: `observation_date,<SERIES>` with '.' for missing values."""
    df = pd.read_csv(io.StringIO(csv_text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce") / 100.0
    s.index = pd.to_datetime(df.iloc[:, 0])
    s.index.name = "date"
    s.name = "cboe_iv30"
    return s.dropna()
