"""HTTP API over the engine, for the web frontend.

Thin by design: every endpoint is a few lines that call into app/ and ivlib/
and shape the result as JSON. Nothing is computed here that is not already
computed elsewhere, so the API can be replaced (serverless functions, a
different framework) without touching the engine.

Run locally:   uvicorn app.api:app --reload --port 8000
Docs:          http://localhost:8000/docs
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app import compute, schema, snapshot, store
from app.sources import cboe, yahoo
from ivlib import filter as qf, surface

app = FastAPI(title="IVchart", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # the frontend is a static site on another origin
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_LIVE_TTL = 300   # seconds a fetched chain is reused before going back to Yahoo
_live_cache: dict[str, tuple[float, pd.DataFrame]] = {}


# ---------------------------------------------------------------- helpers

def _clean(v):
    """JSON cannot carry NaN; send null instead."""
    if v is None:
        return None
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    cols = [c for c in cols if c in df.columns]
    out = []
    for idx, row in df[cols].iterrows():
        rec = {"date": idx.strftime("%Y-%m-%d")}
        rec.update({c: _clean(row[c]) for c in cols})
        out.append(rec)
    return out


def _live_chain(ticker: str) -> pd.DataFrame:
    now = time.time()
    hit = _live_cache.get(ticker)
    if hit and now - hit[0] < _LIVE_TTL:
        return hit[1]
    chain = yahoo.fetch(ticker)
    _live_cache[ticker] = (now, chain)
    return chain


def _stored_chain(ticker: str) -> pd.DataFrame:
    """Most recent stored chain; for AAPL, the vendor CSV's last day as a fallback."""
    days = store.chain_days(ticker)
    if days:
        df = pd.read_parquet(store.chain_path(ticker, days[-1]))
    else:
        vendor = compute._vendor_history(ticker)
        if vendor.empty:
            return pd.DataFrame()
        df = schema.validate(vendor[vendor["QUOTE_DATE"] == vendor["QUOTE_DATE"].max()])
        df["TICKER"], df["SOURCE"], df["MARKET_STATE"] = ticker, "vendor", "REGULAR"
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


# ---------------------------------------------------------------- endpoints

@app.get("/api/tickers")
def tickers():
    """Watchlist plus anything with stored chains, with what history each has."""
    wl = snapshot.load_watchlist()
    names = sorted(set(wl) | set(store.tickers()) | {"AAPL"})
    return [{
        "ticker": t,
        "watched": t in wl,
        "days_stored": len(store.chain_days(t)),
        "cboe_index": cboe.available(t),
        "vendor_history": t == "AAPL" and compute.VENDOR_CSV_PARQUET.exists(),
    } for t in names]


@app.get("/api/metrics/{ticker}")
def metrics(ticker: str, rebuild: bool = False):
    """Daily implied / realized / skew series, plus summary tiles."""
    ticker = ticker.upper()
    m = pd.DataFrame() if rebuild else store.read_metrics(ticker)
    if m.empty:
        try:
            m = compute.build(ticker)
        except yahoo.ChainUnavailable as e:
            raise HTTPException(404, str(e))
        store.write_metrics(ticker, m)

    cols = ["atm_iv_30d", "atm_iv_90d", "rv21_trailing", "rv21_forward",
            "skew30", "cboe_iv30", "coverage", "close"]
    series = _records(m, cols)

    def last(col):
        s = m[col].dropna() if col in m else pd.Series(dtype=float)
        return (None, None) if s.empty else (float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d"))

    iv, iv_d = last("atm_iv_30d")
    rv, rv_d = last("rv21_trailing")
    sk, _ = last("skew30")

    # The gap must compare the two on the SAME day. The latest implied point may
    # be years older than the latest realized point (AAPL: 2023 vs today), and
    # differencing those would be a number with no meaning.
    gap, gap_d = None, None
    if "atm_iv_30d" in m:
        both = m[["atm_iv_30d", "rv21_trailing"]].dropna()
        if not both.empty:
            gap = float(both["atm_iv_30d"].iloc[-1] - both["rv21_trailing"].iloc[-1])
            gap_d = both.index[-1].strftime("%Y-%m-%d")

    return {
        "ticker": ticker,
        "summary": {
            "iv30": iv, "iv30_date": iv_d,
            "rv21": rv, "rv21_date": rv_d,
            "gap": gap, "gap_date": gap_d,
            "skew30": sk,
            "days_implied": int(m["atm_iv_30d"].notna().sum()) if "atm_iv_30d" in m else 0,
            "days_total": int(len(m)),
            "cboe_index": cboe.available(ticker),
        },
        "series": series,
    }


@app.get("/api/smile/{ticker}")
def smile(ticker: str, expiries: int = Query(4, ge=1, le=8)):
    """Today's smile if the market is open, otherwise the last stored chain."""
    ticker = ticker.upper()
    source, chain = "live", pd.DataFrame()
    try:
        chain = _live_chain(ticker)
        if not yahoo.is_live(chain):
            stored = _stored_chain(ticker)
            if stored.empty:
                return {"ticker": ticker, "available": False,
                        "reason": f"market is {chain['MARKET_STATE'].iloc[0]} and no chain is stored yet",
                        "market_state": str(chain["MARKET_STATE"].iloc[0])}
            source = "stored" if store.chain_days(ticker) else "vendor"
            chain = stored
    except yahoo.ChainUnavailable as e:
        raise HTTPException(404, str(e))

    cm, crep = qf.filter_quotes(chain["C_BID"], chain["C_ASK"], dte=chain["DTE"])
    table = surface.build_iv_table(chain)
    day = str(chain["QUOTE_DATE"].iloc[0])

    curves = []
    if not table.empty:
        d = table[table["quote_date"] == day]
        picks = d.groupby("expire_date")["dte"].first().sort_values()
        wants = (14, 35, 90, 200, 7, 60, 120, 300)[:expiries]
        chosen = []
        for w in wants:
            dte = picks.iloc[(picks - w).abs().argmin()]
            if dte not in chosen:
                chosen.append(dte)
        for dte in sorted(chosen):
            g = d[d["dte"] == dte].sort_values("log_moneyness")
            g = g[(g["log_moneyness"] > -0.30) & (g["log_moneyness"] < 0.25)]
            if len(g) < 4:
                continue
            curves.append({
                "dte": int(dte),
                "expiry": str(g["expire_date"].iloc[0]),
                "points": [{"k": _clean(k), "iv": _clean(v), "strike": _clean(s)}
                           for k, v, s in zip(g["log_moneyness"], g["iv"], g["strike"])],
            })

    return {
        "ticker": ticker, "available": True, "source": source, "date": day,
        "spot": _clean(chain["UNDERLYING_LAST"].iloc[0]),
        "market_state": str(chain["MARKET_STATE"].iloc[0]),
        "n_solved": int(len(table)), "n_expiries": int(table["expire_date"].nunique()) if len(table) else 0,
        "quality": {"total": crep["total"], "kept": crep["kept"], "kept_pct": crep["kept_pct"],
                    "rejected_by": {k: v for k, v in crep["rejected_by"].items() if v}},
        "curves": curves,
    }


@app.post("/api/snapshot/{ticker}")
def take_snapshot(ticker: str, force: bool = False):
    """Store today's chain. Refuses outside market hours unless force."""
    status, detail = snapshot.snapshot_one(ticker.upper(), force=force)
    if status == "stored":
        _live_cache.pop(ticker.upper(), None)
    return {"status": status, "detail": detail}


@app.get("/api/watchlist")
def watchlist():
    return snapshot.load_watchlist()


@app.post("/api/watchlist/{ticker}")
def track(ticker: str):
    """Start tracking a ticker. Locally this appends to app/watchlist.txt; a
    deployed API must write somewhere the daily job reads from (the repo)."""
    added, detail = snapshot.add_to_watchlist(ticker)
    if not added and "not a ticker" in detail:
        raise HTTPException(400, detail)
    return {"added": added, "detail": detail, "watchlist": snapshot.load_watchlist()}


@app.get("/api/health")
def health():
    return {"ok": True, "tickers_stored": len(store.tickers())}
