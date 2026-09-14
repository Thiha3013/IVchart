"""Where snapshots live and how they are read back.

Layout:

    data/chains/<TICKER>/<YYYY-MM-DD>.parquet     one compacted chain per day
    data/metrics/<TICKER>.parquet                 derived daily series (rebuilt)

The chains are the asset. They are what the original project only ever had for
AAPL, and every derived number can be recomputed from them when the engine
changes. Metrics are a cache.

`data/` is committed to the repo, on purpose. At ~20 KB per ticker-day, twenty
tickers cost ~100 MB per year -- fine for git for a few years, and it means a
plain clone carries the whole dataset with nothing to configure. If it outgrows
that, moving to object storage is a change to this file only.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data"
CHAINS = ROOT / "chains"
METRICS = ROOT / "metrics"


def chain_path(ticker: str, day: str) -> Path:
    return CHAINS / ticker.upper() / f"{day}.parquet"


def write_chain(chain: pd.DataFrame) -> Path:
    ticker = str(chain["TICKER"].iloc[0])
    day = str(chain["QUOTE_DATE"].iloc[0])
    p = chain_path(ticker, day)
    p.parent.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(p, compression="zstd", index=False)
    return p


def has_chain(ticker: str, day: str) -> bool:
    return chain_path(ticker, day).exists()


def chain_days(ticker: str) -> list[str]:
    d = CHAINS / ticker.upper()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.parquet"))


def read_chains(ticker: str) -> pd.DataFrame:
    """Every stored day for one ticker, concatenated, float64 for the engine."""
    d = CHAINS / ticker.upper()
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    f32 = df.select_dtypes("float32").columns
    return df.astype({c: "float64" for c in f32})


def tickers() -> list[str]:
    if not CHAINS.exists():
        return []
    return sorted(p.name for p in CHAINS.iterdir() if p.is_dir())


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
