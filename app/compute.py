"""Turn stored chains into the daily series the chart draws.

For one ticker:

    stored chains  ->  ivlib.surface  ->  iv30, iv90, skew, coverage   (implied)
    price history  ->  app.realized   ->  rv21 trailing, rv21 forward  (realized)
    Cboe index     ->  cboe_iv30                                       (check)

joined on date into one table, written to data/metrics/<TICKER>.parquet.

Everything here is a pure function of the stored chains plus two free feeds, so
it is safe to delete data/metrics/ and rebuild at any time -- and that is what
you do after improving the engine.

AAPL gets the 2021-2023 vendor CSV prepended to its snapshots, which is what
makes its chart three years deep on day one. Any other ticker's implied series
starts on its first snapshot.

Usage:
    python -m app.compute              # every ticker with stored chains
    python -m app.compute AAPL GME     # just these
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app import realized, store
from app.sources import cboe, yahoo
from ivlib import surface

VENDOR_CSV_PARQUET = Path(__file__).resolve().parent.parent / "aapl_2021_2023.parquet"


def _vendor_history(ticker: str) -> pd.DataFrame:
    """The 2021-2023 AAPL CSV, if it has been ingested. Same schema, older dates."""
    if ticker.upper() != "AAPL" or not VENDOR_CSV_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(VENDOR_CSV_PARQUET)
    f32 = df.select_dtypes("float32").columns
    return df.astype({c: "float64" for c in f32})


def implied_series(chains: pd.DataFrame) -> pd.DataFrame:
    """ivlib over every stored day -> one row per date."""
    if chains.empty:
        return pd.DataFrame()
    table = surface.build_iv_table(chains)
    if table.empty:
        return pd.DataFrame()

    term = surface.atm_term_structure(table)
    iv30 = surface.constant_maturity(term, days=30).set_index("quote_date")
    iv90 = surface.constant_maturity(term, days=90).set_index("quote_date")

    sk = surface.skew(table, wing=0.10)
    sk30 = (sk[(sk["dte"] >= 20) & (sk["dte"] <= 45)]
            .groupby("quote_date")["skew"].mean().rename("skew30"))

    per_day = table.groupby("quote_date").agg(
        n_solved=("iv", "size"),
        n_expiries=("expire_date", "nunique"),
        spot=("forward", "median"),
    )
    n_quoted = chains.groupby("QUOTE_DATE").size().rename("n_quoted")

    out = pd.concat([iv30, iv90, sk30, per_day, n_quoted], axis=1)
    out["coverage"] = out["n_solved"] / out["n_quoted"]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.sort_index()


def build(ticker: str, price_period: str = "5y") -> pd.DataFrame:
    ticker = ticker.upper()
    chains = pd.concat([_vendor_history(ticker), store.read_chains(ticker)], ignore_index=True)
    imp = implied_series(chains)

    close = yahoo.price_history(ticker, period=price_period)
    rv = realized.both(close)

    parts = [imp, rv, close.rename("close")]
    if cboe.available(ticker):
        parts.append(cboe.fetch(ticker))

    m = pd.concat(parts, axis=1).sort_index()
    m.index.name = "date"
    # Keep every date that has *some* signal; drop the all-NaN padding pandas adds.
    return m.dropna(how="all")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    tickers = [a.upper() for a in argv] or store.tickers()
    if "AAPL" not in tickers and VENDOR_CSV_PARQUET.exists():
        tickers.append("AAPL")
    if not tickers:
        print("no stored chains yet -- run app.snapshot first")
        return 2

    t0 = time.perf_counter()
    for tk in sorted(set(tickers)):
        t = time.perf_counter()
        try:
            m = build(tk)
        except Exception as e:
            print(f"[failed ] {tk}: {type(e).__name__}: {e}")
            continue
        p = store.write_metrics(tk, m)
        n_iv = int(m["atm_iv_30d"].notna().sum()) if "atm_iv_30d" in m else 0
        print(f"[ok     ] {tk}: {len(m)} dates, {n_iv} with implied vol "
              f"-> {p.relative_to(store.ROOT.parent)}  [{time.perf_counter()-t:.1f}s]")
    print(f"\ndone [{time.perf_counter()-t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
