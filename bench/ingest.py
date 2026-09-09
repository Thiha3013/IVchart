"""Convert the raw option CSV to Parquet.

The CSV is 195 MB of text with bracketed, space-padded column names and numeric
columns that pandas types as `object` because of embedded blanks. Every run pays
the parsing cost. Parquet stores typed columns, so later runs read only what they
need.

Usage:  python bench/ingest.py [src.csv] [dst.parquet]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

NUMERIC = [
    "UNDERLYING_LAST", "STRIKE", "DTE",
    "C_BID", "C_ASK", "C_LAST", "C_IV", "C_VOLUME",
    "P_BID", "P_ASK", "P_LAST", "P_IV", "P_VOLUME",
    "STRIKE_DISTANCE",
]
KEEP_STR = ["QUOTE_DATE", "EXPIRE_DATE"]


def tidy(df):
    """Strip the bracket/space noise from headers and string cells."""
    df.columns = [c.strip().strip("[]") for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def main(src="aapl_2021_2023.csv", dst="aapl_2021_2023.parquet"):
    t0 = time.perf_counter()
    df = tidy(pd.read_csv(src, low_memory=False))
    read_s = time.perf_counter() - t0

    cols = [c for c in KEEP_STR + NUMERIC if c in df.columns]
    df = df[cols].copy()
    for c in NUMERIC:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")

    t1 = time.perf_counter()
    df.to_parquet(dst, engine="pyarrow", compression="zstd", index=False)
    write_s = time.perf_counter() - t1

    src_mb = Path(src).stat().st_size / 1e6
    dst_mb = Path(dst).stat().st_size / 1e6
    print(f"rows          {len(df):,}")
    print(f"csv           {src_mb:8.1f} MB   read {read_s:6.2f}s")
    print(f"parquet       {dst_mb:8.1f} MB   write {write_s:6.2f}s")
    print(f"compression   {src_mb / dst_mb:8.2f}x")

    t2 = time.perf_counter()
    pd.read_parquet(dst, columns=["QUOTE_DATE", "STRIKE"])
    print(f"2-col read    {time.perf_counter() - t2:8.3f}s")
    t3 = time.perf_counter()
    pd.read_parquet(dst)
    print(f"full read     {time.perf_counter() - t3:8.3f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
