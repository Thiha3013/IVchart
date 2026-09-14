"""Vendor CSV -> Parquet. 195 MB text -> 9 MB typed columns (7x compression; the rest is float32 + dropped columns).

    python -m bench.ingest [src.csv] [dst.parquet]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent / "data" / "vendor"
NUMERIC = ["UNDERLYING_LAST", "STRIKE", "DTE", "C_BID", "C_ASK", "C_LAST", "C_IV", "C_VOLUME",
           "P_BID", "P_ASK", "P_LAST", "P_IV", "P_VOLUME", "STRIKE_DISTANCE"]
KEEP_STR = ["QUOTE_DATE", "EXPIRE_DATE"]


def tidy(df):
    """Strip the bracket/space noise from headers and string cells."""
    df.columns = [c.strip().strip("[]") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def main(src=ROOT / "aapl_2021_2023.csv", dst=ROOT / "aapl_2021_2023.parquet"):
    src, dst = Path(src), Path(dst)
    t0 = time.perf_counter()
    df = tidy(pd.read_csv(src, low_memory=False))
    read_s = time.perf_counter() - t0
    df = df[[c for c in KEEP_STR + NUMERIC if c in df.columns]].copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    t1 = time.perf_counter()
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)
    src_mb, dst_mb = src.stat().st_size / 1e6, dst.stat().st_size / 1e6
    print(f"rows          {len(df):,}")
    print(f"csv           {src_mb:8.1f} MB   read {read_s:6.2f}s")
    print(f"parquet       {dst_mb:8.1f} MB   write {time.perf_counter() - t1:6.2f}s")
    print(f"compression   {src_mb / dst_mb:8.2f}x")


if __name__ == "__main__":
    main(*sys.argv[1:])
