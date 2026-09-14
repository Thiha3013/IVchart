"""The canonical option-chain table every source must produce.

Column names deliberately match the vendor CSV (`aapl_2021_2023.csv`) that the engine
was validated against, so a chain from any source drops straight into
`ivlib.surface.build_iv_table` with no adapter in between. If a new source needs a
different shape, it is the source that adapts, never the engine.

One row per (expiry, strike), with the call and put side by side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Required by ivlib. Types are what the engine expects after loading.
CORE = {
    "QUOTE_DATE": "string",        # YYYY-MM-DD, exchange-local calendar date
    "EXPIRE_DATE": "string",       # YYYY-MM-DD
    "DTE": "float64",              # calendar days to expiry (see ivlib.bs day count)
    "UNDERLYING_LAST": "float64",
    "STRIKE": "float64",
    "C_BID": "float64", "C_ASK": "float64",
    "P_BID": "float64", "P_ASK": "float64",
}

# Kept for provenance and for re-running the engine later; not read by ivlib.
EXTRA = {
    "C_LAST": "float64", "P_LAST": "float64",
    "C_VOLUME": "float64", "P_VOLUME": "float64",
    "C_OI": "float64", "P_OI": "float64",
    "TICKER": "string",
    "SOURCE": "string",            # "yahoo", "alphavantage", ...
    "MARKET_STATE": "string",      # "REGULAR", "PRE", "POST", "CLOSED"
    "QUOTE_UNIXTIME": "int64",     # when the quote was captured
}

COLUMNS = list(CORE) + list(EXTRA)


def empty() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in {**CORE, **EXTRA}.items()})


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the schema and fail loudly on anything the engine cannot use."""
    missing = [c for c in CORE if c not in df.columns]
    if missing:
        raise ValueError(f"chain is missing required columns: {missing}")

    out = df.copy()
    for c, t in {**CORE, **EXTRA}.items():
        if c not in out.columns:
            fill = pd.NA if t == "string" else (0 if t == "int64" else np.nan)
            out[c] = pd.Series([fill] * len(out), dtype=t)
        elif t == "string":
            out[c] = out[c].astype("string")
        elif t == "int64":
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int64")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    if (out["DTE"] < 0).any():
        raise ValueError("chain has negative DTE -- expiry before quote date")
    if not np.isfinite(out["UNDERLYING_LAST"]).all():
        raise ValueError("chain has a non-finite underlying price")
    return out[COLUMNS].reset_index(drop=True)


def compact(df: pd.DataFrame, moneyness=0.30, max_dte=400) -> pd.DataFrame:
    """Trim a chain to the region the engine can actually use, for storage.

    Strikes more than 30% from spot and expiries past ~13 months contribute
    nothing to a 30-day constant-maturity series or a skew measure, and they
    are the bulk of a listed chain. Dropping them and downcasting prices to
    float32 takes a snapshot from ~150 KB to ~15 KB with no loss the engine
    would notice.
    """
    spot = df["UNDERLYING_LAST"]
    keep = (
        (df["STRIKE"] >= spot * (1 - moneyness))
        & (df["STRIKE"] <= spot * (1 + moneyness))
        & (df["DTE"] <= max_dte)
    )
    out = df[keep].copy()
    for c in ("C_BID", "C_ASK", "P_BID", "P_ASK", "C_LAST", "P_LAST",
              "C_VOLUME", "P_VOLUME", "C_OI", "P_OI"):
        out[c] = out[c].astype("float32")
    return out.reset_index(drop=True)
