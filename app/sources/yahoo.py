"""Today's option chain from Yahoo Finance, via yfinance.

What this source is and is not
------------------------------
It is the only free source of *current* chains for any optionable US ticker. It
is not a source of history: Yahoo serves today's chain and nothing else, so the
only way to get a time series out of it is to call it every day and keep what
comes back. That is what app/snapshot.py does.

Two things about the data that shape everything downstream:

1. Outside regular trading hours Yahoo returns bid = ask = 0 on every contract.
   A chain captured at 3 AM is not "slightly stale", it is empty of the only
   fields the engine uses. `fetch` records `marketState` so the caller can refuse
   to store it, and the snapshot job does exactly that.

2. Yahoo's own `impliedVolatility` column is not usable. Sampled pre-market on
   AAPL it read 0.00001 for every ITM call and stepped through 0.0156, 0.0313,
   0.0625, 0.125 for OTM ones -- placeholder values, not a computation. The
   engine takes bid/ask only and derives its own vol. That column is dropped.

yfinance itself has broken repeatedly as Yahoo changes its site (v0.2.33, which
this repo's venv shipped with, fails outright in 2026). Pin a recent version and
expect to bump it.
"""

from __future__ import annotations

import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app import schema

ET = ZoneInfo("America/New_York")


class ChainUnavailable(RuntimeError):
    """Yahoo returned nothing usable for this ticker."""


def _ticker(symbol):
    import yfinance as yf
    return yf.Ticker(symbol)


def fetch(symbol: str, max_expiries: int | None = None) -> pd.DataFrame:
    """Fetch every listed expiry for `symbol` and return one schema-shaped table.

    Raises ChainUnavailable if the ticker has no listed options or Yahoo returns
    no expiries (delisted, wrong symbol, or the API is having a bad day).
    """
    symbol = symbol.upper().strip()
    tk = _ticker(symbol)

    try:
        expiries = list(tk.options)
    except Exception as e:  # yfinance raises a grab-bag of types here
        raise ChainUnavailable(f"{symbol}: could not list expiries ({e})") from e
    if not expiries:
        raise ChainUnavailable(f"{symbol}: no listed options")
    if max_expiries:
        expiries = expiries[:max_expiries]

    frames, spot, state, captured = [], None, None, int(time.time())
    for exp in expiries:
        try:
            oc = tk.option_chain(exp)
        except Exception:
            continue
        if spot is None:
            u = getattr(oc, "underlying", {}) or {}
            spot = u.get("regularMarketPrice")
            state = u.get("marketState", "UNKNOWN")
        frames.append(_merge_sides(oc.calls, oc.puts, exp))

    if not frames or spot is None or not np.isfinite(spot):
        raise ChainUnavailable(f"{symbol}: chain returned but no usable underlying price")

    chain = pd.concat(frames, ignore_index=True)
    today = datetime.now(ET).date()
    chain["QUOTE_DATE"] = today.isoformat()
    chain["DTE"] = [(date.fromisoformat(e) - today).days for e in chain["EXPIRE_DATE"]]
    chain["UNDERLYING_LAST"] = float(spot)
    chain["TICKER"] = symbol
    chain["SOURCE"] = "yahoo"
    chain["MARKET_STATE"] = state
    chain["QUOTE_UNIXTIME"] = captured
    return schema.validate(chain)


def _merge_sides(calls: pd.DataFrame, puts: pd.DataFrame, expiry: str) -> pd.DataFrame:
    """Put the call and put for each strike on one row, CSV-style."""
    c = calls[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].rename(columns={
        "strike": "STRIKE", "bid": "C_BID", "ask": "C_ASK", "lastPrice": "C_LAST",
        "volume": "C_VOLUME", "openInterest": "C_OI"})
    p = puts[["strike", "bid", "ask", "lastPrice", "volume", "openInterest"]].rename(columns={
        "strike": "STRIKE", "bid": "P_BID", "ask": "P_ASK", "lastPrice": "P_LAST",
        "volume": "P_VOLUME", "openInterest": "P_OI"})
    m = c.merge(p, on="STRIKE", how="outer").sort_values("STRIKE")
    m["EXPIRE_DATE"] = expiry
    return m


def is_live(chain: pd.DataFrame) -> bool:
    """True if this chain was captured during regular hours and has real quotes."""
    state = str(chain["MARKET_STATE"].iloc[0]) if len(chain) else ""
    two_sided = ((chain["C_BID"] > 0) & (chain["C_ASK"] > 0)).mean() if len(chain) else 0.0
    return state == "REGULAR" and two_sided > 0.2


def price_history(symbol: str, period: str = "5y") -> pd.Series:
    """Daily closes for realized-vol calculation. Unlike chains, this is historical."""
    h = _ticker(symbol).history(period=period, auto_adjust=True)
    if h.empty:
        raise ChainUnavailable(f"{symbol}: no price history")
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = "close"
    return s
