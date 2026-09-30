"""HTTP API for the web frontend. Thin: every endpoint calls app/ and ivlib/, shapes JSON.

    uvicorn app.api:app --reload --port 8000        docs at /docs
"""

from __future__ import annotations

import collections
import contextlib
import math
import os
import platform
import re
import threading
import time
from datetime import date
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import data, pipeline, sources
from ivlib import market as mk, solver, surface


# ---------------------------------------------------------------- background threads

_started = time.time()
_clock_log: collections.deque = collections.deque(maxlen=50)   # in-memory; /api/health shows it
_threads: dict[str, threading.Thread] = {}
_beat = {"clock": time.time()}   # last clock loop; a stuck pass stops it moving


def _log(event, **kw):
    now = pipeline.datetime.now(sources.ET).strftime("%Y-%m-%d %H:%M:%S ET")
    _clock_log.append({"t": now, "event": event, **kw})


def _warm():
    """JIT numba and pre-build AAPL so a cold start costs the boot, not the first request."""
    try:
        solver.implied_vol_fast(np.array([3.0]), 100.0, np.array([100.0]), 0.1, 1.0, True)
        _metrics("AAPL")
    except Exception:
        pass   # warm-up is best-effort; requests build on demand anyway


def _clock():
    """Every 5 min: refresh the GitHub check; in the window with work pending, snapshot + publish."""
    _log("clock started")
    while True:
        _beat["clock"] = time.time()
        try:
            sources.github_status()   # self-caches 6 h; kept off the request path
            if pipeline.in_snapshot_window() and pipeline.pending():
                _log("pass started")
                _log("pass finished", **pipeline.snapshot_and_publish())
        except Exception as e:   # never let the clock die
            _log("pass error", error=f"{type(e).__name__}: {e}"[:300])
        time.sleep(300)


def _self_ping():
    """Hit our own public URL every 5 min so Render's idle timer never fires.

    Render sets RENDER_EXTERNAL_URL. Defense in depth next to the external pinger,
    which lapsed on 2026-09-26 and let the API sleep through the snapshot window.
    Unverified whether Render counts self-traffic as inbound; if not, harmless.
    """
    import urllib.request
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    while True:
        time.sleep(300)
        try:
            urllib.request.urlopen(f"{url}/api/health", timeout=30).read()
        except Exception:
            pass


@contextlib.asynccontextmanager
async def _lifespan(_app):
    for target in (_warm, _clock, _self_ping):
        t = threading.Thread(target=target, name=target.__name__, daemon=True)   # none may block the health check
        t.start()
        _threads[target.__name__] = t
    yield


app = FastAPI(title="IVchart", version="0.3.0", lifespan=_lifespan)

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


def _client_ip(request: Request) -> str:
    """Edge-set headers first: a client can forge XFF's leftmost entry."""
    h = request.headers
    ip = h.get("cf-connecting-ip") or h.get("true-client-ip") or (h.get("x-forwarded-for") or "").split(",")[0].strip()
    return ip or (request.client.host if request.client else "?")


@app.middleware("http")
async def _rate_limit(request: Request, call_next):
    if request.url.path == "/api/health":   # Render's health check and the pingers; cheap, never throttled
        return await call_next(request)
    ip, now = _client_ip(request), time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= _RATE:
            return JSONResponse({"detail": f"rate limit: {_RATE} requests/min"}, status_code=429)
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


# ---------------------------------------------------------------- caches

_LIVE_TTL = 300
_live_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_yahoo = threading.BoundedSemaphore(3)   # live chain fetches in flight: memory, on a 512 MB box
_METRICS_TTL = 6 * 3600                  # realized vol moves once a day; a new snapshot expires it early
_build_lock = threading.Lock()           # one metrics build at a time


def _clean(v):
    """JSON has no NaN."""
    if isinstance(v, (float, np.floating)):
        return None if not math.isfinite(v) else float(v)
    return int(v) if isinstance(v, np.integer) else v


def _records(df, cols):
    """Rows as JSON-safe dicts. Vectorized: iterrows() cost ~2 s per 4k rows on Render's shared CPU."""
    cols = [c for c in cols if c in df.columns]
    out = df[cols].astype(object).where(df[cols].notna(), None)
    out.insert(0, "date", pd.to_datetime(df.index).strftime("%Y-%m-%d"))
    return out.to_dict("records")


def _live_chain(ticker):
    now, hit = time.time(), _live_cache.get(ticker)
    if hit and now - hit[0] < _LIVE_TTL:
        return hit[1]
    if not _yahoo.acquire(timeout=30):
        raise HTTPException(503, "busy -- try again in a moment")
    try:
        chain = sources.fetch_chain(ticker)
    finally:
        _yahoo.release()
    for k in [k for k, (t, _) in _live_cache.items() if now - t > _LIVE_TTL]:   # evict expired
        _live_cache.pop(k, None)
    if len(_live_cache) >= 32:
        _live_cache.pop(next(iter(_live_cache)))
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


def _metrics(ticker):
    """Cached, rebuilt past the TTL. A failed rebuild serves the stale copy rather than an error."""
    def fresh():
        age = data.metrics_age(ticker)
        return age is not None and age < _METRICS_TTL

    m = data.read_metrics(ticker)
    if not m.empty and fresh():
        return m
    if not _build_lock.acquire(timeout=60):
        if not m.empty:
            return m
        raise HTTPException(503, "busy building another ticker -- try again in a moment")
    try:
        m = data.read_metrics(ticker)          # another request may have just built it
        if not m.empty and fresh():
            return m
        try:
            new = pipeline.build_metrics(ticker)
        except Exception:
            if m.empty:
                raise
            return m
        data.write_metrics(ticker, new)
        return new
    finally:
        _build_lock.release()


# ---------------------------------------------------------------- endpoints

@app.get("/api/tickers")
def tickers():
    wl = pipeline.load_watchlist()
    return [{"ticker": t, "watched": t in wl, "days_stored": len(data.chain_days(t)),
             "cboe_index": sources.cboe_available(t), "vendor_history": t == "AAPL" and data.VENDOR.exists()}
            for t in sorted(set(wl) | set(data.tickers()) | {"AAPL"})]


@app.get("/api/metrics/{ticker}")
def metrics(ticker: str):
    ticker = _ticker(ticker)
    try:
        m = _metrics(ticker)
    except sources.ChainUnavailable as e:
        raise HTTPException(404, str(e))
    except HTTPException:
        raise
    except Exception as e:   # Yahoo/FRED hiccup with nothing cached: retryable, not a 500
        raise HTTPException(503, f"could not build {ticker} right now ({type(e).__name__}) -- try again shortly")

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
    """Today's smile if the market is open, else (or if Yahoo fails) the last stored chain."""
    ticker = _ticker(ticker)
    state, reason = None, None
    try:
        chain, source = _live_chain(ticker), "live"
        state = str(chain["MARKET_STATE"].iloc[0])
        if not sources.is_live(chain):
            reason = f"market is {state} and no chain is stored yet"
    except sources.ChainUnavailable as e:
        reason = str(e)
    except Exception as e:   # Yahoo down, rate-limited, or we're busy: a stored chain beats an error
        reason = f"live quotes unavailable ({getattr(e, 'detail', None) or type(e).__name__}) and no chain is stored yet"
    if reason:
        chain, source = _fallback_chain(ticker)
        if chain.empty:
            return {"ticker": ticker, "available": False, "market_state": state, "reason": reason}

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


_TRACK_PER_HOUR = 5   # site-wide: each add is a commit, and each commit a Render redeploy (build minutes)
_tracked: collections.deque = collections.deque()


@app.post("/api/watchlist/{ticker}")
def track(ticker: str):
    """Local: append to app/watchlist.txt. Deployed (GITHUB_TOKEN set): commit to the repo."""
    t = _ticker(ticker)
    if not t.isalnum() or len(t) > 6:
        raise HTTPException(400, "not a ticker symbol")
    now = time.time()
    with _hits_lock:
        while _tracked and now - _tracked[0] > 3600:
            _tracked.popleft()
        if len(_tracked) >= _TRACK_PER_HOUR:
            raise HTTPException(429, f"{_TRACK_PER_HOUR} tickers were added in the last hour -- try again later")
    if sources.github_configured():
        try:
            sources.fetch_chain(t, max_expiries=1)
        except sources.ChainUnavailable as e:
            return {"added": False, "detail": str(e), "watchlist": pipeline.load_watchlist()}
        added, detail = sources.github_append_ticker(t)
    else:
        added, detail = pipeline.add_to_watchlist(t)
    wl = pipeline.load_watchlist()
    if added:
        _tracked.append(now)
        if t not in wl:   # github path: local file lags the commit until redeploy
            wl.append(t)
    return {"added": added, "detail": detail, "watchlist": wl,
            "via": "github" if sources.github_configured() else "local"}


# ---------------------------------------------------------------- health

def _installed(pkg):
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


_PINS = dict(re.findall(r"^([\w.-]+)==(\S+)", (Path(__file__).parent / "constraints.txt").read_text(), re.M))
_VERSIONS = {"python": platform.python_version(), **{p: _installed(p) for p in [*_PINS, "yfinance"]},
             "numba_threads": numba.config.NUMBA_NUM_THREADS}
_UNPINNED = sorted(p for p, v in _PINS.items() if _VERSIONS[p] != v)   # [] on Render = the tested set


@app.get("/api/health")
def health(strict: bool = False):
    """Always 200 for Render's health check and the keep-alive. ?strict=1 is 503 while `problems` is
    non-empty: point a daily monitor there."""
    days = [d for t in data.tickers() for d in data.chain_days(t)]
    last = max(days) if days else None
    gh = sources.github_status(max_age=float("inf"))   # read-only; the clock refreshes it
    problems = []
    if n := pipeline.missed_sessions(last):
        problems.append(f"{n} trading day(s) since {last} without a snapshot")
    if os.environ.get("RENDER"):   # deployed: snapshots must be able to commit
        if gh["ok"] is False or not sources.github_configured():
            problems.append(gh["error"])
        elif gh["expires"] and (date.fromisoformat(gh["expires"]) - date.today()).days <= 14:
            problems.append(f"GitHub token expires {gh['expires']}: regenerate it, update GITHUB_TOKEN on Render")
    clock = _threads.get("_clock")
    if clock is not None and (not clock.is_alive() or time.time() - _beat["clock"] > 7200):
        problems.append("snapshot clock is not running")
    if pipeline.unpublished:
        problems.append(f"{len(pipeline.unpublished)} chains fetched but not committed (GitHub failing)")
    wl = pipeline.load_watchlist()
    body = {"ok": not problems, "problems": problems, "version": app.version, "uptime_s": int(time.time() - _started),
            "last_snapshot": last, "missing_last": [t for t in wl if last and not data.has_chain(t, last)],
            "tickers_stored": len(data.tickers()), "in_window": pipeline.in_snapshot_window(),
            "github": {k: gh[k] for k in ("ok", "expires", "error")},
            "versions": _VERSIONS, "unpinned": _UNPINNED, "clock_log": list(_clock_log)}
    return JSONResponse(body, status_code=503 if strict and problems else 200)
