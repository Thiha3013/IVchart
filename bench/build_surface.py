"""Build the surface-derived series that bench/plot_surface.py draws from.

Produces four Parquet files in the working directory:

    iv_table.parquet   one row per usable strike -- the full surface
    term.parquet       ATM-forward vol per (date, expiry)
    skew.parquet       IV(-10%) - IV(+10%) per (date, expiry)
    iv_hv.parquet      30-day constant-maturity implied vol beside trailing
                       30-day realized vol, one row per day

Usage:  python -m bench.build_surface [parquet]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ivlib import surface
from bench.validate import load

OUTPUTS = ("iv_table.parquet", "term.parquet", "skew.parquet", "iv_hv.parquet")


def realized_vol(path, window=21):
    """Trailing annualized realized vol from the underlying's daily closes.

    21 trading days is the customary '30 calendar day' window, and the sqrt(252)
    annualization is correct here because the returns are per *trading* day --
    the same convention question as the option's T, answered consistently.
    """
    raw = pd.read_parquet(path, columns=["QUOTE_DATE", "UNDERLYING_LAST"])
    spot = raw.groupby("QUOTE_DATE")["UNDERLYING_LAST"].first().astype(float).sort_index()
    lr = np.log(spot / spot.shift(1))
    return (lr.rolling(window).std(ddof=1) * np.sqrt(252)).rename(f"hv{30}")


def main(path="aapl_2021_2023.parquet"):
    t0 = time.perf_counter()
    df = load(path)
    tab = surface.build_iv_table(df)
    print(f"iv table   {len(tab):>9,} rows   {tab['quote_date'].nunique()} days   "
          f"[{time.perf_counter()-t0:.2f}s]")

    term = surface.atm_term_structure(tab)
    sk = surface.skew(tab, wing=0.10)
    print(f"term       {len(term):>9,} (date, expiry) pairs")
    print(f"skew       {len(sk):>9,} (date, expiry) pairs")

    cm30 = surface.constant_maturity(term, days=30)
    hv = realized_vol(path)
    ivhv = cm30.set_index("quote_date").join(hv).dropna()
    prem = ivhv["atm_iv_30d"] - ivhv["hv30"]
    print(f"iv vs hv   {len(ivhv):>9,} days   implied above realized on "
          f"{100*(prem > 0).mean():.1f}% of days, mean gap {prem.mean():+.4f}")

    tab.to_parquet("iv_table.parquet", index=False)
    term.to_parquet("term.parquet", index=False)
    sk.to_parquet("skew.parquet", index=False)
    ivhv.to_parquet("iv_hv.parquet")
    print(f"wrote {', '.join(OUTPUTS)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
