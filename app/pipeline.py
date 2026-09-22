"""The jobs: snapshot today's chains, compute the daily series.

    python -m app.pipeline snapshot [TICKERS...] [--force]   store today's chain per watched ticker
    python -m app.pipeline compute  [TICKERS...]             chains -> iv30/skew/coverage + realized vol
    python -m app.pipeline vendor                            precompute the implied series over the vendor chains

Snapshot refuses chains captured outside regular hours (bid=ask=0 then) and stores
one per ticker per day. History for a ticker starts the day it's first snapshotted.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from app import data, sources
from ivlib import surface

WATCHLIST = Path(__file__).resolve().parent / "watchlist.txt"
EARLIEST_ET_HOUR = 14   # store only late-session chains, so snapshot time is consistent day to day
CLOSE_ET_HOUR = 16
WATCHLIST_MAX = sources.WATCHLIST_MAX
TRADING_DAYS = 252
RV_WINDOW = 21   # trading days ~ 30 calendar, matching the 30d implied series


# ---------------------------------------------------------------- watchlist

def load_watchlist(path: Path = WATCHLIST) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip().upper()
        if s:
            out.append(s)
    return out


def add_to_watchlist(ticker: str, path: Path = WATCHLIST) -> tuple[bool, str]:
    """Local append after validating the symbol has options. Returns (added, detail)."""
    ticker = ticker.upper().strip()
    if not ticker.isalnum() or len(ticker) > 6:
        return False, f"{ticker!r} is not a ticker symbol"
    current = load_watchlist(path)
    if ticker in current:
        return False, f"{ticker} is already on the watchlist"
    if len(current) >= WATCHLIST_MAX:
        return False, f"watchlist is full ({WATCHLIST_MAX} tickers)"
    try:
        sources.fetch_chain(ticker, max_expiries=1)
    except sources.ChainUnavailable as e:
        return False, str(e)
    except Exception as e:
        return False, f"{ticker}: could not verify ({type(e).__name__})"
    lines = path.read_text().splitlines() if path.exists() else []
    path.write_text("".join(ln + chr(10) for ln in lines + [ticker]))
    return True, f"{ticker} added -- history starts with the next snapshot"


# ---------------------------------------------------------------- snapshot

def snapshot_one(ticker: str, force: bool = False) -> tuple[str, str]:
    """Returns (status, detail); status in stored / skipped / failed."""
    try:
        chain = sources.fetch_chain(ticker)
    except sources.ChainUnavailable as e:
        return "failed", str(e)
    except Exception as e:
        return "failed", f"{ticker}: {type(e).__name__}: {e}"

    day, state = str(chain["QUOTE_DATE"].iloc[0]), str(chain["MARKET_STATE"].iloc[0])
    if not force and data.has_chain(ticker, day):
        return "skipped", f"{ticker}: already have {day}"
    if not force and datetime.now(sources.ET).hour < EARLIEST_ET_HOUR:
        return "skipped", f"{ticker}: before {EARLIEST_ET_HOUR}:00 ET -- not storing yet"
    if not force and not sources.is_live(chain):
        two_sided = float(((chain["C_BID"] > 0) & (chain["C_ASK"] > 0)).mean())
        return "skipped", f"{ticker}: market state {state}, {two_sided:.0%} two-sided quotes -- not storing"

    compact = data.compact(chain)
    p = data.write_chain(compact)
    return "stored", f"{ticker}: {day} {len(compact)} rows -> {p.relative_to(data.ROOT.parent)}"


def in_snapshot_window(now: datetime | None = None) -> bool:
    """Weekday, 14:00-16:00 ET. snapshot_one's market-state check still covers holidays."""
    now = now or datetime.now(sources.ET)
    return now.weekday() < 5 and EARLIEST_ET_HOUR <= now.hour < CLOSE_ET_HOUR


def snapshot_and_publish(tickers: list[str] | None = None) -> dict:
    """Snapshot the watchlist, then commit new chain files to the repo in one commit.

    The API's own daily job: GitHub's cron fired hours late or not at all, so the
    always-on server keeps the clock and publishes with the token it already has.
    """
    tickers = tickers or load_watchlist()
    today = datetime.now(sources.ET).date().isoformat()
    new_files, counts = {}, {"stored": 0, "skipped": 0, "failed": 0}
    for tk in tickers:
        status, _ = snapshot_one(tk)
        counts[status] += 1
        time.sleep(1.0)   # spread ~300 Yahoo requests out; a burst gets rate-limited
        if status == "stored":   # repo path is a contract: data/chains/<T>/<day>.parquet
            new_files[f"data/chains/{tk.upper()}/{today}.parquet"] = data.chain_path(tk, today).read_bytes()
    committed = None
    if new_files and sources.github_configured():
        committed = sources.github_commit_files(new_files, f"snapshot {today} ({len(new_files)} tickers)")
    return {**counts, "committed": committed}


def snapshot(tickers: list[str] | None = None, force: bool = False) -> int:
    tickers = tickers or load_watchlist()
    if not tickers:
        print("nothing to do: no tickers given and watchlist is empty")
        return 2
    t0 = time.perf_counter()
    counts = {"stored": 0, "skipped": 0, "failed": 0}
    for tk in tickers:
        status, detail = snapshot_one(tk, force=force)
        counts[status] += 1
        print(f"[{status:7}] {detail}")
    print(f"\n{counts['stored']} stored, {counts['skipped']} skipped, {counts['failed']} failed  [{time.perf_counter()-t0:.1f}s]")
    return 1 if counts["failed"] else 0


# ---------------------------------------------------------------- realized vol
# Log returns per trading day, sample std, sqrt(252). Trailing = what the stock has
# shown; forward = what it went on to show (the honest score for an implied vol).

def realized_vol(close: pd.Series, window: int = RV_WINDOW) -> pd.DataFrame:
    r = np.log(close.astype(float).sort_index() / close.astype(float).sort_index().shift(1)).dropna()
    trailing = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    forward = r[::-1].rolling(window).std(ddof=1)[::-1].shift(-1) * np.sqrt(TRADING_DAYS)
    return pd.DataFrame({f"rv{window}_trailing": trailing, f"rv{window}_forward": forward})


# ---------------------------------------------------------------- compute

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
    sk30 = sk[(sk["dte"] >= 20) & (sk["dte"] <= 45)].groupby("quote_date")["skew"].mean().rename("skew30")
    per_day = table.groupby("quote_date").agg(n_solved=("iv", "size"), n_expiries=("expire_date", "nunique"),
                                              spot=("forward", "median"))
    n_quoted = chains.groupby("QUOTE_DATE").size().rename("n_quoted")
    out = pd.concat([iv30, iv90, sk30, per_day, n_quoted], axis=1)
    out["coverage"] = out["n_solved"] / out["n_quoted"]
    out.index = pd.to_datetime(out.index)
    out.index.name = "date"
    return out.sort_index()


def build_metrics(ticker: str, price_period: str = "5y") -> pd.DataFrame:
    """Implied + realized + Cboe, joined on date. Vendor history comes precomputed (see `vendor`)."""
    ticker = ticker.upper()
    implied = pd.concat([data.vendor_implied(ticker), implied_series(data.read_chains(ticker))])
    parts = [implied[~implied.index.duplicated(keep="last")]] if len(implied) else []   # an empty frame poisons the index dtype on pandas 3
    close = sources.price_history(ticker, period=price_period)
    parts += [realized_vol(close), close.rename("close")]
    if sources.cboe_available(ticker):
        parts.append(sources.cboe_index(ticker))
    m = pd.concat(parts, axis=1)
    m.index = pd.to_datetime(m.index)
    m.index.name = "date"
    return m.sort_index().dropna(how="all")


def compute(tickers: list[str] | None = None) -> int:
    tickers = [t.upper() for t in (tickers or data.tickers())]
    if "AAPL" not in tickers and data.VENDOR.exists():
        tickers.append("AAPL")
    if not tickers:
        print("no stored chains yet -- run snapshot first")
        return 2
    t0 = time.perf_counter()
    for tk in sorted(set(tickers)):
        t = time.perf_counter()
        try:
            m = build_metrics(tk)
        except Exception as e:
            print(f"[failed ] {tk}: {type(e).__name__}: {e}")
            continue
        p = data.write_metrics(tk, m)
        n_iv = int(m["atm_iv_30d"].notna().sum()) if "atm_iv_30d" in m else 0
        print(f"[ok     ] {tk}: {len(m)} dates, {n_iv} with implied vol -> {p.relative_to(data.ROOT.parent)}  [{time.perf_counter()-t:.1f}s]")
    print(f"\ndone [{time.perf_counter()-t0:.1f}s]")
    return 0


def vendor() -> int:
    """One-off: run the engine over the 548k-row vendor dataset and store the daily series."""
    if not data.VENDOR.exists():
        print("no vendor dataset"); return 2
    t0 = time.perf_counter()
    chains = data.vendor_history("AAPL")
    s = implied_series(chains)
    s.to_parquet(data.VENDOR_IMPLIED, compression="zstd")
    last = chains[chains["QUOTE_DATE"] == chains["QUOTE_DATE"].max()]
    last.to_parquet(data.VENDOR_LASTDAY, compression="zstd", index=False)
    print(f"{len(s)} days -> {data.VENDOR_IMPLIED.relative_to(data.ROOT.parent)}; "
          f"{len(last)} rows -> {data.VENDOR_LASTDAY.relative_to(data.ROOT.parent)}  [{time.perf_counter()-t0:.1f}s]")
    return 0


# ---------------------------------------------------------------- cli

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("snapshot", "compute", "vendor"):
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "vendor":
        return vendor()
    force = "--force" in rest
    tickers = [a for a in rest if not a.startswith("--")]
    return snapshot(tickers, force) if cmd == "snapshot" else compute(tickers)


if __name__ == "__main__":
    sys.exit(main())
