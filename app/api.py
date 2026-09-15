"""HTTP API for the web frontend. Thin: every endpoint calls app/ and ivlib/, shapes JSON.

    uvicorn app.api:app --reload --port 8000        docs at /docs
"""

from __future__ import annotations

import collections
import contextlib
import math
import os
import re
import threading
import time

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import data, pipeline, sources
from ivlib import market as mk, solver, surface


def _warm():
    """JIT numba and pre-build AAPL so a cold start costs the boot, not the first request."""
    try:
        solver.implied_vol_fast(np.array([3.0]), 100.0, np.array([100.0]), 0.1, 1.0, True)
        _metrics("AAPL")
    except Exception:
        pass   # warm-up is best-effort; requests build on demand anyway


@contextlib.asynccontextmanager
async def _lifespan(_app):
    threading.Thread(target=_warm, daemon=True).start()   # don't block the health check
    yield


app = FastAPI(title="IVchart", version="0.2.0", lifespan=_lifespan)

# Browser access only from the deployed frontend, its Vercel previews, and local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ivchart.vercel.app", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https://ivchart-[a-z0-9-]+-thiha3013\.vercel\.app",
    allow_methods=["GET", "POST"], allow_headers=["*"],
)

# Per-IP rate limit. Every endpoint can reach Yahoo, and one abusive client
# getting Render's IP throttled would take the site down for everyone.
_RATE = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_hits_lock = threading.Lock()


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    ip = (request.headers.get("x-forwarded-for") or request.client.host or "?").split(",")[0].strip()
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= _RATE:
            return JSONResponse({"detail": "rate limit: 60 requests/min"}, status_code=429)
        q.append(now)
        if len(_hits) > 10_000:   # bound memory under a flood of distinct IPs
            _hits.clear()
    return await call_next(request)


_TICKER = re.compile(r"^[A-Z0-9.^-]{1,8}$")


def _ticker(raw: str) -> str:
    """Uppercase, and only characters a listed symbol can contain. Rejects everything else."""
    t = raw.upper().strip()
    if not _TICKER.match(t):
        raise HTTPException(400, "not a ticker symbol")
    return t

_LIVE_TTL = 300
_live_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_build_lock = threading.Lock()   # one metrics build at a time: 512 MB instance


def _clean(v):
    """JSON has no NaN."""
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else float(v)
    return int(v) if isinstance(v, np.integer) else v


def _records(df, cols):
    """Rows as JSON-safe dicts. Vectorized: iterrows() cost ~2 s per 4k rows on Render's shared CPU."""
    cols = [c for c in cols if c in df.columns]
    out = df[cols].astype(object).where(df[cols].notna(), None)
    out.insert(0, "date", df.index.strftime("%Y-%m-%d"))
    return out.to_dict("records")


def _live_chain(ticker):
    now, hit = time.time(), _live_cache.get(ticker)
    if hit and now - hit[0] < _LIVE_TTL:
        return hit[1]
    chain = sources.fetch_chain(ticker)
    _live_cache[ticker] = (now, chain)
    return chain


def _fallback_chain(ticker):
    """Last stored chain; for AAPL, the vendor dataset's last day."""
    stored = data.latest_chain(ticker)
    if not stored.empty:
        return stored, "stored"
    vendor = data.vendor_last_day(ticker)
    if vendor.empty:
        return pd.DataFrame(), None
    df = data.validate(vendor)
    df["TICKER"], df["SOURCE"], df["MARKET_STATE"] = ticker, "vendor", "REGULAR"
    return df, "vendor"


def _metrics(ticker, rebuild=False):
    m = pd.DataFrame() if rebuild else data.read_metrics(ticker)
    if m.empty:
        with _build_lock:
            m = data.read_metrics(ticker)          # another thread may have just built it
            if m.empty:
                m = pipeline.build_metrics(ticker)
                data.write_metrics(ticker, m)
    return m


@app.get("/api/tickers")
def tickers():
    wl = pipeline.load_watchlist()
    return [{"ticker": t, "watched": t in wl, "days_stored": len(data.chain_days(t)),
             "cboe_index": sources.cboe_available(t), "vendor_history": t == "AAPL" and data.VENDOR.exists()}
            for t in sorted(set(wl) | set(data.tickers()) | {"AAPL"})]


@app.get("/api/metrics/{ticker}")
def metrics(ticker: str, rebuild: bool = False):
    ticker = _ticker(ticker)
    try:
        m = _metrics(ticker, rebuild)
    except sources.ChainUnavailable as e:
        raise HTTPException(404, str(e))

    def last(col):
        s = m[col].dropna() if col in m else pd.Series(dtype=float)
        return (None, None) if s.empty else (float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d"))

    iv, iv_d = last("atm_iv_30d")
    rv, rv_d = last("rv21_trailing")
    sk, _ = last("skew30")
    gap, gap_d = None, None   # must be same-day: latest implied may be years older than latest realized
    if "atm_iv_30d" in m:
        both = m[["atm_iv_30d", "rv21_trailing"]].dropna()
        if not both.empty:
            gap = float(both["atm_iv_30d"].iloc[-1] - both["rv21_trailing"].iloc[-1])
            gap_d = both.index[-1].strftime("%Y-%m-%d")

    return {
        "ticker": ticker,
        "summary": {"iv30": iv, "iv30_date": iv_d, "rv21": rv, "rv21_date": rv_d, "gap": gap, "gap_date": gap_d,
                    "skew30": sk, "days_implied": int(m["atm_iv_30d"].notna().sum()) if "atm_iv_30d" in m else 0,
                    "days_total": int(len(m)), "cboe_index": sources.cboe_available(ticker)},
        "series": _records(m, ["atm_iv_30d", "atm_iv_90d", "rv21_trailing", "rv21_forward", "skew30", "cboe_iv30", "coverage", "close"]),
    }


@app.get("/api/smile/{ticker}")
def smile(ticker: str, expiries: int = Query(4, ge=1, le=8)):
    """Today's smile if the market is open, else the last stored chain."""
    ticker = _ticker(ticker)
    try:
        chain, source = _live_chain(ticker), "live"
        if not sources.is_live(chain):
            state = str(chain["MARKET_STATE"].iloc[0])
            chain, source = _fallback_chain(ticker)
            if chain.empty:
                return {"ticker": ticker, "available": False, "market_state": state,
                        "reason": f"market is {state} and no chain is stored yet"}
    except sources.ChainUnavailable as e:
        raise HTTPException(404, str(e))

    _, crep = mk.filter_quotes(chain["C_BID"], chain["C_ASK"], dte=chain["DTE"])
    table = surface.build_iv_table(chain)
    day = str(chain["QUOTE_DATE"].iloc[0])

    curves = []
    if not table.empty:
        d = table[table["quote_date"] == day]
        picks = d.groupby("expire_date")["dte"].first().sort_values()
        chosen = []
        for w in (14, 35, 90, 200, 7, 60, 120, 300)[:expiries]:
            dte = picks.iloc[(picks - w).abs().argmin()]
            if dte not in chosen:
                chosen.append(dte)
        for dte in sorted(chosen):
            g = d[d["dte"] == dte].sort_values("log_moneyness")
            g = g[(g["log_moneyness"] > -0.30) & (g["log_moneyness"] < 0.25)]
            if len(g) >= 4:
                curves.append({"dte": int(dte), "expiry": str(g["expire_date"].iloc[0]),
                               "points": [{"k": _clean(k), "iv": _clean(v), "strike": _clean(s)}
                                          for k, v, s in zip(g["log_moneyness"], g["iv"], g["strike"])]})

    return {"ticker": ticker, "available": True, "source": source, "date": day,
            "spot": _clean(chain["UNDERLYING_LAST"].iloc[0]), "market_state": str(chain["MARKET_STATE"].iloc[0]),
            "n_solved": int(len(table)), "n_expiries": int(table["expire_date"].nunique()) if len(table) else 0,
            "quality": {"total": crep["total"], "kept": crep["kept"], "kept_pct": crep["kept_pct"],
                        "rejected_by": {k: v for k, v in crep["rejected_by"].items() if v}},
            "curves": curves}


@app.get("/api/watchlist")
def watchlist():
    return pipeline.load_watchlist()


@app.post("/api/watchlist/{ticker}")
def track(ticker: str):
    """Local: append to app/watchlist.txt. Deployed (GITHUB_TOKEN set): commit to the repo."""
    t = _ticker(ticker)
    if not t.isalnum() or len(t) > 6:
        raise HTTPException(400, "not a ticker symbol")
    if sources.github_configured():
        try:
            sources.fetch_chain(t, max_expiries=1)
        except sources.ChainUnavailable as e:
            return {"added": False, "detail": str(e), "watchlist": pipeline.load_watchlist()}
        added, detail = sources.github_append_ticker(t)
    else:
        added, detail = pipeline.add_to_watchlist(t)
    wl = pipeline.load_watchlist()
    if added and t not in wl:   # github path: local file lags the commit until redeploy
        wl.append(t)
    return {"added": added, "detail": detail, "watchlist": wl,
            "via": "github" if sources.github_configured() else "local"}


@app.post("/api/snapshot/{ticker}")
def take_snapshot(ticker: str):
    ticker = _ticker(ticker)
    status, detail = pipeline.snapshot_one(ticker, force=False)   # force is local-only; never over HTTP
    if status == "stored":
        _live_cache.pop(ticker, None)
    return {"status": status, "detail": detail}


@app.get("/api/health")
def health():
    return {"ok": True, "tickers_stored": len(data.tickers())}
