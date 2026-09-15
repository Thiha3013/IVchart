"""Chain schema and on-disk layout.

Columns match the vendor CSV so any source drops into ivlib unchanged.

    data/chains/<TICKER>/<YYYY-MM-DD>.parquet   one compacted chain per day (committed)
    data/metrics/<TICKER>.parquet               derived series, a cache (ignored)
    data/vendor/aapl_2021_2023.parquet          AAPL 2021-23 vendor chains (committed, 9 MB)
    data/vendor/aapl_2021_2023_implied.parquet  the engine's daily series over those chains,
                                                precomputed so the API never loads 548k rows
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
CHAINS, METRICS = ROOT / "chains", ROOT / "metrics"
VENDOR = ROOT / "vendor" / "aapl_2021_2023.parquet"
VENDOR_IMPLIED = ROOT / "vendor" / "aapl_2021_2023_implied.parquet"

CORE = {
    "QUOTE_DATE": "string", "EXPIRE_DATE": "string", "DTE": "float64",
    "UNDERLYING_LAST": "float64", "STRIKE": "float64",
    "C_BID": "float64", "C_ASK": "float64", "P_BID": "float64", "P_ASK": "float64",
}
EXTRA = {
    "C_LAST": "float64", "P_LAST": "float64", "C_VOLUME": "float64", "P_VOLUME": "float64",
    "C_OI": "float64", "P_OI": "float64",
    "TICKER": "string", "SOURCE": "string", "MARKET_STATE": "string", "QUOTE_UNIXTIME": "int64",
}
COLUMNS = list(CORE) + list(EXTRA)
_PRICE_COLS = ("C_BID", "C_ASK", "P_BID", "P_ASK", "C_LAST", "P_LAST", "C_VOLUME", "P_VOLUME", "C_OI", "P_OI")


# ---------------------------------------------------------------- schema

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to the schema; fail loudly on what the engine can't use."""
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
    """Trim to +/-30% moneyness, <=400 DTE, float32 prices: ~150 KB -> ~15 KB, lossless for the engine."""
    spot = df["UNDERLYING_LAST"]
    keep = (df["STRIKE"] >= spot * (1 - moneyness)) & (df["STRIKE"] <= spot * (1 + moneyness)) & (df["DTE"] <= max_dte)
    out = df[keep].copy()
    for c in _PRICE_COLS:
        out[c] = out[c].astype("float32")
    return out.reset_index(drop=True)


def _widen(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


# ---------------------------------------------------------------- chains

def chain_path(ticker: str, day: str) -> Path:
    return CHAINS / ticker.upper() / f"{day}.parquet"


def write_chain(chain: pd.DataFrame) -> Path:
    p = chain_path(str(chain["TICKER"].iloc[0]), str(chain["QUOTE_DATE"].iloc[0]))
    p.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(p, compression="zstd", index=False)
    return p


def has_chain(ticker: str, day: str) -> bool:
    return chain_path(ticker, day).exists()


def chain_days(ticker: str) -> list[str]:
    d = CHAINS / ticker.upper()
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def read_chains(ticker: str) -> pd.DataFrame:
    """All stored days for a ticker, float64."""
    d = CHAINS / ticker.upper()
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    return _widen(pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)) if files else pd.DataFrame()


def latest_chain(ticker: str) -> pd.DataFrame:
    days = chain_days(ticker)
    return _widen(pd.read_parquet(chain_path(ticker, days[-1]))) if days else pd.DataFrame()


def tickers() -> list[str]:
    return sorted(p.name for p in CHAINS.iterdir() if p.is_dir()) if CHAINS.exists() else []


def vendor_history(ticker: str) -> pd.DataFrame:
    """AAPL 2021-23 vendor chains. Same schema, older dates. 548k rows -- avoid in the API."""
    if ticker.upper() != "AAPL" or not VENDOR.exists():
        return pd.DataFrame()
    return _widen(pd.read_parquet(VENDOR))


def vendor_last_day(ticker: str) -> pd.DataFrame:
    """Only the vendor dataset's final day, read with a parquet filter (a few hundred rows)."""
    if ticker.upper() != "AAPL" or not VENDOR.exists():
        return pd.DataFrame()
    import pyarrow.parquet as pq
    last = pq.read_table(VENDOR, columns=["QUOTE_DATE"]).column(0).to_pylist()
    day = max(last)
    return _widen(pd.read_parquet(VENDOR, filters=[("QUOTE_DATE", "==", day)]))


def vendor_implied(ticker: str) -> pd.DataFrame:
    """Precomputed implied series over the vendor chains (see pipeline.vendor)."""
    if ticker.upper() != "AAPL" or not VENDOR_IMPLIED.exists():
        return pd.DataFrame()
    return pd.read_parquet(VENDOR_IMPLIED)


# ---------------------------------------------------------------- metrics cache

def metrics_path(ticker: str) -> Path:
    return METRICS / f"{ticker.upper()}.parquet"


def write_metrics(ticker: str, df: pd.DataFrame) -> Path:
    p = metrics_path(ticker)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, compression="zstd")
    return p


def read_metrics(ticker: str) -> pd.DataFrame:
    p = metrics_path(ticker)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
