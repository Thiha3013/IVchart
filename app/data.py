"""Chain schema and on-disk layout.

Columns match the vendor CSV so any source drops into ivlib unchanged.

    data/chains/<TICKER>/<YYYY-MM-DD>.parquet   one compacted chain per day (committed)
    data/implied.parquet                        engine's daily series, one row per ticker per snapshot
                                                (committed with the chains, so metrics never re-solve history)
    data/metrics/<TICKER>.parquet               derived series, a cache (ignored)
    data/vendor/aapl_2021_2023.parquet          AAPL 2021-23 vendor chains (committed, 9 MB)
    data/vendor/aapl_2021_2023_implied.parquet  the engine's daily series over those chains,
                                                precomputed so the API never loads 548k rows
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
CHAINS, METRICS = ROOT / "chains", ROOT / "metrics"
IMPLIED = ROOT / "implied.parquet"
VENDOR = ROOT / "vendor" / "aapl_2021_2023.parquet"
VENDOR_IMPLIED = ROOT / "vendor" / "aapl_2021_2023_implied.parquet"
VENDOR_LASTDAY = ROOT / "vendor" / "aapl_2021_2023_lastday.parquet"

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


def widen(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


# ---------------------------------------------------------------- files

def to_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    """Parquet bytes, so a file can be committed before it exists locally."""
    buf = io.BytesIO()
    df.to_parquet(buf, compression="zstd", index=index)
    return buf.getvalue()


def write_bytes(path: Path, blob: bytes) -> Path:
    """Atomic: a crash mid-write never leaves a torn parquet behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(blob)
    tmp.replace(path)
    return path


# ---------------------------------------------------------------- chains

def chain_path(ticker: str, day: str) -> Path:
    return CHAINS / ticker.upper() / f"{day}.parquet"


def write_chain(chain: pd.DataFrame) -> Path:
    p = chain_path(str(chain["TICKER"].iloc[0]), str(chain["QUOTE_DATE"].iloc[0]))
    return write_bytes(p, to_bytes(chain))


def has_chain(ticker: str, day: str) -> bool:
    return chain_path(ticker, day).exists()


def chain_days(ticker: str) -> list[str]:
    d = CHAINS / ticker.upper()
    return sorted(p.stem for p in d.glob("*.parquet")) if d.exists() else []


def read_chains(ticker: str, days: list[str] | None = None) -> pd.DataFrame:
    """Stored days for a ticker (all, or just `days`), float64."""
    d = CHAINS / ticker.upper()
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if days is not None:
        files = [f for f in files if f.stem in set(days)]
    return widen(pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)) if files else pd.DataFrame()


def latest_chain(ticker: str) -> pd.DataFrame:
    days = chain_days(ticker)
    return widen(pd.read_parquet(chain_path(ticker, days[-1]))) if days else pd.DataFrame()


def tickers() -> list[str]:
    return sorted(p.name for p in CHAINS.iterdir() if p.is_dir()) if CHAINS.exists() else []


# ---------------------------------------------------------------- implied series

def read_implied_all() -> pd.DataFrame:
    return pd.read_parquet(IMPLIED) if IMPLIED.exists() else pd.DataFrame()


def read_implied(ticker: str) -> pd.DataFrame:
    """One ticker's daily implied rows, indexed by date."""
    df = read_implied_all()
    if df.empty:
        return df
    out = df[df["ticker"] == ticker.upper()].drop(columns="ticker").set_index("date")
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


# ---------------------------------------------------------------- vendor

def vendor_history(ticker: str) -> pd.DataFrame:
    """AAPL 2021-23 vendor chains. Same schema, older dates. 548k rows -- avoid in the API."""
    if ticker.upper() != "AAPL" or not VENDOR.exists():
        return pd.DataFrame()
    return widen(pd.read_parquet(VENDOR))


def vendor_last_day(ticker: str) -> pd.DataFrame:
    """The vendor dataset's final day (a few hundred rows), precomputed by `pipeline vendor`."""
    if ticker.upper() != "AAPL" or not VENDOR_LASTDAY.exists():
        return pd.DataFrame()
    return widen(pd.read_parquet(VENDOR_LASTDAY))


def vendor_implied(ticker: str) -> pd.DataFrame:
    """Precomputed implied series over the vendor chains (see pipeline.vendor)."""
    if ticker.upper() != "AAPL" or not VENDOR_IMPLIED.exists():
        return pd.DataFrame()
    return pd.read_parquet(VENDOR_IMPLIED)


# ---------------------------------------------------------------- metrics cache

METRICS_MAX = 200   # files; the endpoint takes any ticker, so bound the disk


def metrics_path(ticker: str) -> Path:
    return METRICS / f"{ticker.upper()}.parquet"


def metrics_age(ticker: str) -> float | None:
    """Seconds since the cached metrics were built; None if never."""
    try:
        return time.time() - metrics_path(ticker).stat().st_mtime
    except FileNotFoundError:
        return None


def expire_metrics(ticker: str) -> None:
    """Stale, not deleted: a failed rebuild can still serve it."""
    p = metrics_path(ticker)
    if p.exists():
        os.utime(p, (0, 0))


def write_metrics(ticker: str, df: pd.DataFrame) -> Path:
    p = write_bytes(metrics_path(ticker), to_bytes(df, index=True))
    for f in sorted(METRICS.glob("*.parquet"), key=lambda f: f.stat().st_mtime)[:-METRICS_MAX]:
        f.unlink(missing_ok=True)   # oldest first
    return p


def read_metrics(ticker: str) -> pd.DataFrame:
    p = metrics_path(ticker)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
