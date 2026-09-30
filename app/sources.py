"""External data: Yahoo (today's chain, price history), FRED (Cboe vol indices), GitHub (watchlist writes).

Yahoo: current chain only, any optionable ticker. Outside regular hours bid=ask=0
everywhere, and its `impliedVolatility` column is a placeholder -- bid/ask in, our own
vol out. Cboe indices: 30d VIX-methodology IV for 5 names back to 2010, free, no key.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from app import data

ET = ZoneInfo("America/New_York")


class ChainUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------- yahoo

def _ticker(symbol):
    import yfinance as yf
    return yf.Ticker(symbol)


def _retry(fn, tries=3, base=1.5):
    """Call fn, backing off 1.5 s, 3 s on failure. Yahoo 429s bursts of requests."""
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(base * 2 ** i)


def fetch_chain(symbol: str, max_expiries: int | None = None) -> pd.DataFrame:
    """Every listed expiry for `symbol`, as one schema-shaped table."""
    symbol = symbol.upper().strip()
    tk = _ticker(symbol)
    try:
        expiries = list(_retry(lambda: tk.options))
    except Exception as e:
        raise ChainUnavailable(f"{symbol}: could not list expiries ({e})") from e
    if not expiries:
        raise ChainUnavailable(f"{symbol}: no listed options")
    if max_expiries:
        expiries = expiries[:max_expiries]

    frames, spot, state, captured, failed = [], None, None, int(time.time()), 0
    for exp in expiries:
        try:
            oc = _retry(lambda: tk.option_chain(exp))
        except Exception:
            failed += 1
            continue
        if spot is None:
            u = getattr(oc, "underlying", {}) or {}
            spot, state = u.get("regularMarketPrice"), u.get("marketState", "UNKNOWN")
        frames.append(_merge_sides(oc.calls, oc.puts, exp))

    # A chain missing expiries is not today's chain. Yahoo rate-limits bursts; better
    # to fail the ticker and let the clock retry than to store a partial record.
    if failed > max(1, len(expiries) // 10):
        raise ChainUnavailable(f"{symbol}: {failed} of {len(expiries)} expiries failed after retries")
    if not frames or spot is None or not np.isfinite(spot):
        raise ChainUnavailable(f"{symbol}: chain returned but no usable underlying price")

    chain = pd.concat(frames, ignore_index=True)
    today = datetime.now(ET).date()
    chain["QUOTE_DATE"] = today.isoformat()
    chain["DTE"] = [(date.fromisoformat(e) - today).days for e in chain["EXPIRE_DATE"]]
    chain["UNDERLYING_LAST"] = float(spot)
    chain["TICKER"], chain["SOURCE"], chain["MARKET_STATE"] = symbol, "yahoo", state
    chain["QUOTE_UNIXTIME"] = captured
    return data.validate(chain)


def _merge_sides(calls, puts, expiry):
    ren = {"strike": "STRIKE", "bid": "BID", "ask": "ASK", "lastPrice": "LAST", "volume": "VOLUME", "openInterest": "OI"}
    cols = list(ren)
    c = calls[cols].rename(columns={k: ("STRIKE" if v == "STRIKE" else f"C_{v}") for k, v in ren.items()})
    p = puts[cols].rename(columns={k: ("STRIKE" if v == "STRIKE" else f"P_{v}") for k, v in ren.items()})
    m = c.merge(p, on="STRIKE", how="outer").sort_values("STRIKE")
    m["EXPIRE_DATE"] = expiry
    return m


def is_live(chain: pd.DataFrame) -> bool:
    """Regular hours and >20% two-sided quotes."""
    if not len(chain):
        return False
    two_sided = ((chain["C_BID"] > 0) & (chain["C_ASK"] > 0)).mean()
    return str(chain["MARKET_STATE"].iloc[0]) == "REGULAR" and two_sided > 0.2


def price_history(symbol: str, period: str = "5y") -> pd.Series:
    h = _ticker(symbol).history(period=period, auto_adjust=True)
    if h.empty:
        raise ChainUnavailable(f"{symbol}: no price history")
    s = h["Close"].copy()
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s.name = "close"
    return s


# ---------------------------------------------------------------- cboe via FRED
# Variance-strip rate, not ATM vol: prices in the skew, runs ~10% above ours on
# AAPL (corr 0.978). The correlation is the check; the level gap is expected.

CBOE_SERIES = {"AAPL": "VXAPLCLS", "AMZN": "VXAZNCLS", "GOOG": "VXGOGCLS", "GS": "VXGSCLS", "IBM": "VXIBMCLS"}
_FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"


def cboe_available(ticker: str) -> bool:
    return ticker.upper() in CBOE_SERIES


def cboe_index(ticker: str, timeout: int = 30) -> pd.Series:
    """Daily 30d IV as a decimal; empty for uncovered names."""
    sid = CBOE_SERIES.get(ticker.upper())
    if sid is None:
        return pd.Series(dtype="float64", name="cboe_iv30")
    raw = urllib.request.urlopen(_FRED.format(sid=sid), timeout=timeout).read().decode()
    return parse_fred(raw)


def parse_fred(csv_text: str) -> pd.Series:
    """`observation_date,<SERIES>` with '.' for missing."""
    df = pd.read_csv(io.StringIO(csv_text))
    s = pd.to_numeric(df.iloc[:, 1], errors="coerce") / 100.0
    s.index = pd.to_datetime(df.iloc[:, 0])
    s.index.name, s.name = "date", "cboe_iv30"
    return s.dropna()


# ---------------------------------------------------------------- github (the deployed store)
# Render's disk is ephemeral; the repo is the store. Track commits the watchlist (Contents
# API), snapshots commit chains (Git Data API), and each commit redeploys Render, so the
# instance's disk mirrors the repo. Env: GITHUB_REPO, GITHUB_TOKEN (fine-grained, Contents
# r/w on this repo), GITHUB_BRANCH (default main).

WATCHLIST_PATH = "app/watchlist.txt"
WATCHLIST_MAX = int(os.environ.get("WATCHLIST_MAX", "40"))   # endpoint is public; bound the blast radius


def github_configured() -> bool:
    return bool(os.environ.get("GITHUB_TOKEN") and os.environ.get("GITHUB_REPO"))


def _gh(method: str, url: str, body: dict | None = None, with_headers: bool = False):
    req = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None,
                                 method=method, headers={
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/json", "User-Agent": "ivchart-api"})
    with urllib.request.urlopen(req, timeout=20) as r:
        out = json.loads(r.read().decode())
        return (out, r.headers) if with_headers else out


_status: dict = {"ok": None, "expires": None, "error": "not checked yet", "checked": 0.0}


def github_status(max_age: float = 6 * 3600) -> dict:
    """Can the token still reach the repo, and when does it expire? Cached; the clock refreshes it.

    401/403/404 -> ok False (expired, revoked, no access). Network or GitHub 5xx -> unknown, not cached.
    Expiry comes from the GitHub-Authentication-Token-Expiration header; absent for tokens that never expire.
    """
    global _status
    if not github_configured():
        return {"ok": None, "expires": None, "error": "GITHUB_TOKEN/GITHUB_REPO not set", "checked": 0.0}
    if time.time() - _status["checked"] < max_age:
        return _status
    try:
        _, h = _gh("GET", f"https://api.github.com/repos/{os.environ['GITHUB_REPO']}", with_headers=True)
        exp = h.get("github-authentication-token-expiration")
        _status = {"ok": True, "expires": exp[:10] if exp else None, "error": None, "checked": time.time()}
    except urllib.error.HTTPError as e:
        if e.code not in (401, 403, 404):
            return {**_status, "error": f"GitHub HTTP {e.code}"}
        _status = {"ok": False, "expires": None, "error": f"GitHub HTTP {e.code}: token expired, revoked, or lacks repo access",
                   "checked": time.time()}
    except Exception as e:
        return {**_status, "error": f"{type(e).__name__}: {e}"}
    return _status


def _contents_url() -> str:
    return f"https://api.github.com/repos/{os.environ['GITHUB_REPO']}/contents/{WATCHLIST_PATH}"


def github_commit_files(files: dict[str, bytes], message: str) -> str | None:
    """Commit several files in ONE commit via the Git Data API. Returns the sha; None if nothing changed.

    A branch that moved underneath (a Track commit landing mid-publish) is retried on the new head.
    """
    repo, branch = os.environ["GITHUB_REPO"], os.environ.get("GITHUB_BRANCH", "main")
    base = f"https://api.github.com/repos/{repo}"
    tree = [{"path": path, "mode": "100644", "type": "blob",
             "sha": _gh("POST", f"{base}/git/blobs",
                        {"content": base64.b64encode(blob).decode(), "encoding": "base64"})["sha"]}
            for path, blob in files.items()]
    for attempt in range(3):
        head = _gh("GET", f"{base}/git/ref/heads/{branch}")["object"]["sha"]
        base_tree = _gh("GET", f"{base}/git/commits/{head}")["tree"]["sha"]
        tree_sha = _gh("POST", f"{base}/git/trees", {"base_tree": base_tree, "tree": tree})["sha"]
        if tree_sha == base_tree:
            return None   # repo already has these bytes: no empty commit, no redeploy
        commit = _gh("POST", f"{base}/git/commits", {"message": message, "tree": tree_sha, "parents": [head]})["sha"]
        try:
            _gh("PATCH", f"{base}/git/refs/heads/{branch}", {"sha": commit, "force": False})
            return commit
        except urllib.error.HTTPError as e:
            if e.code != 422 or attempt == 2:   # 422 = not a fast-forward
                raise


def github_append_ticker(ticker: str) -> tuple[bool, str]:
    """Append to the repo watchlist in one commit. Idempotent; stale sha -> reported, not retried."""
    ticker = ticker.upper().strip()
    branch = os.environ.get("GITHUB_BRANCH", "main")
    try:
        j = _gh("GET", f"{_contents_url()}?ref={branch}")
    except urllib.error.HTTPError as e:
        return False, f"could not read watchlist from GitHub ({e.code})"
    text, sha = base64.b64decode(j["content"]).decode(), j["sha"]

    present = {ln.split("#", 1)[0].strip().upper() for ln in text.splitlines()} - {""}
    if ticker in present:
        return False, f"{ticker} is already on the watchlist"
    if len(present) >= WATCHLIST_MAX:
        return False, f"watchlist is full ({WATCHLIST_MAX} tickers) -- remove one in the repo to add another"

    lines = text.splitlines() + [ticker]
    body = {"message": f"watchlist: track {ticker}", "sha": sha, "branch": branch,
            "content": base64.b64encode("".join(ln + chr(10) for ln in lines).encode()).decode()}
    try:
        _gh("PUT", _contents_url(), body)
    except urllib.error.HTTPError as e:
        return False, "watchlist changed underneath us -- try again" if e.code == 409 else f"GitHub rejected the update ({e.code})"
    return True, f"{ticker} added to the repo watchlist -- history starts with the next scheduled snapshot"
