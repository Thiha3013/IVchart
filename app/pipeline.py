"""The jobs: snapshot today's chains, compute the daily series.

    python -m app.pipeline snapshot [TICKERS...] [--force]   store today's chain per watched ticker
    python -m app.pipeline compute  [TICKERS...]             chains -> iv30/skew/coverage + realized vol
    python -m app.pipeline implied                           rebuild data/implied.parquet from every stored chain
    python -m app.pipeline vendor                            precompute the implied series over the vendor chains

Snapshot refuses chains captured outside regular hours (bid=ask=0 then) and stores
one per ticker per day. History for a ticker starts the day it's first snapshotted.
"""

from __future__ import annotations

import functools
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import TH
from pandas.tseries.holiday import (AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
                                    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday)
from pandas.tseries.offsets import DateOffset, Day

from app import data, sources
from ivlib import surface

WATCHLIST = Path(__file__).resolve().parent / "watchlist.txt"
EARLIEST_ET_HOUR = 14   # store only late-session chains, so snapshot time is consistent day to day
CLOSE_ET_HOUR = 16
TICKER_TIMEOUT = 90     # s per ticker; a hung Yahoo call must not stall the whole pass
WATCHLIST_MAX = sources.WATCHLIST_MAX
TRADING_DAYS = 252
RV_WINDOW = 21   # trading days ~ 30 calendar, matching the 30d implied series
GAP_FILL_MAX = 30   # stored days solved per call when data/implied.parquet lacks them; bounds memory


def today_et() -> str:
    return datetime.now(sources.ET).date().isoformat()


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

def _fetch_compact(ticker: str, force: bool = False):
    """(status, detail, compact chain | None, market state | None). Never writes.

    Cheap checks come first: a ticker already stored today, or a pass before the
    window, costs no Yahoo call.
    """
    today = today_et()
    if not force and data.has_chain(ticker, today):
        return "skipped", f"{ticker}: already have {today}", None, None
    if not force and datetime.now(sources.ET).hour < EARLIEST_ET_HOUR:
        return "skipped", f"{ticker}: before {EARLIEST_ET_HOUR}:00 ET -- not storing yet", None, None
    try:
        chain = sources.fetch_chain(ticker)
    except sources.ChainUnavailable as e:
        return "failed", str(e), None, None
    except Exception as e:
        return "failed", f"{ticker}: {type(e).__name__}: {e}", None, None

    day, state = str(chain["QUOTE_DATE"].iloc[0]), str(chain["MARKET_STATE"].iloc[0])
    if not force and data.has_chain(ticker, day):
        return "skipped", f"{ticker}: already have {day}", None, state
    if not force and not sources.is_live(chain):
        two_sided = float(((chain["C_BID"] > 0) & (chain["C_ASK"] > 0)).mean())
        return "skipped", f"{ticker}: market state {state}, {two_sided:.0%} two-sided quotes -- not storing", None, state
    compact = data.compact(chain)
    return "stored", f"{ticker}: {day} {len(compact)} rows", compact, state


def snapshot_one(ticker: str, force: bool = False) -> tuple[str, str]:
    """CLI path: fetch and write locally. Returns (status, detail)."""
    status, detail, compact, _ = _fetch_compact(ticker, force)
    if status == "stored":
        detail += f" -> {data.write_chain(compact).relative_to(data.ROOT.parent)}"
    return status, detail


class _NYSE(AbstractHolidayCalendar):
    """NYSE closures + 13:00 early closes (before our 14:00 window, so closed for us). Rules: no yearly upkeep.
    Unscheduled closures (mourning days) aren't here; the market-state check catches those."""
    rules = [Holiday("New Year", month=1, day=1, observance=sunday_to_monday), USMartinLutherKingJr,
             USPresidentsDay, GoodFriday, USMemorialDay,
             Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01", observance=nearest_workday),
             Holiday("July 4", month=7, day=4, observance=nearest_workday), USLaborDay, USThanksgivingDay,
             Holiday("Christmas", month=12, day=25, observance=nearest_workday),
             Holiday("July 3, early", month=7, day=3), Holiday("Christmas Eve, early", month=12, day=24),
             Holiday("Black Friday, early", month=11, day=1, offset=[DateOffset(weekday=TH(4)), Day(1)])]


@functools.lru_cache(maxsize=8)
def _closed(year: int) -> frozenset:
    return frozenset(t.date() for t in _NYSE().holidays(f"{year}-01-01", f"{year}-12-31"))


def trading_day(d: date) -> bool:
    """A session still open at 14:00 ET."""
    return d.weekday() < 5 and d not in _closed(d.year)


def in_snapshot_window(now: datetime | None = None) -> bool:
    """Trading day, 14:00-16:00 ET. No Yahoo calls on holidays."""
    now = now or datetime.now(sources.ET)
    return trading_day(now.date()) and EARLIEST_ET_HOUR <= now.hour < CLOSE_ET_HOUR


def missed_sessions(last: str | None, now: datetime | None = None) -> int:
    """Trading days after `last` whose window has closed (16:15 ET) with no snapshot."""
    if not last:
        return 0
    now = now or datetime.now(sources.ET)
    end = now.date() if (now.hour, now.minute) >= (CLOSE_ET_HOUR, 15) else now.date() - timedelta(days=1)
    d, n = date.fromisoformat(last) + timedelta(days=1), 0
    while d <= end:
        n += trading_day(d)
        d += timedelta(days=1)
    return n


unpublished: dict[tuple[str, str], pd.DataFrame] = {}   # (ticker, day) -> chain whose commit failed


def pending(tickers: list[str] | None = None) -> bool:
    """Work for the clock: an unpublished chain, or a watched ticker without today's."""
    today = today_et()
    return bool(unpublished) or any(not data.has_chain(t, today) for t in (tickers or load_watchlist()))


def _with_implied_rows(frames: dict) -> pd.DataFrame | None:
    """data/implied.parquet + rows for new chains + stored days it lacks, so it converges even after a race."""
    old = data.read_implied_all()
    added = []
    for tk in sorted(set(data.tickers()) | {t for t, _ in frames}):
        have = pd.DatetimeIndex(old.loc[old["ticker"] == tk, "date"]) if len(old) else pd.DatetimeIndex([])
        new = [implied_series(data.widen(df)) for (t, _), df in frames.items() if t == tk] + [_implied_gaps(tk, have)]
        added += [s.reset_index().assign(ticker=tk) for s in new if not s.empty]
    if not added:
        return None
    out = pd.concat([old, *added] if len(old) else added, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)


def _publish() -> str | None:
    """Commit unpublished chains + implied rows in ONE commit, then write them locally. Raises -> kept for next pass."""
    frames = dict(unpublished)
    files = {f"data/chains/{tk}/{day}.parquet": (data.chain_path(tk, day), data.to_bytes(df))
             for (tk, day), df in frames.items()}   # repo path -> (local path, bytes)
    if (implied := _with_implied_rows(frames)) is not None:
        files["data/implied.parquet"] = (data.IMPLIED, data.to_bytes(implied))
    days = ", ".join(sorted({day for _, day in frames}))
    sha = (sources.github_commit_files({k: b for k, (_, b) in files.items()}, f"snapshot {days} ({len(frames)} tickers)")
           if sources.github_configured() else None)
    for path, blob in files.values():
        data.write_bytes(path, blob)
    for tk, day in frames:
        unpublished.pop((tk, day), None)
        data.expire_metrics(tk)
    return sha


def snapshot_and_publish(tickers: list[str] | None = None) -> dict:
    """Snapshot the watchlist; commit chains + implied rows in one commit; only then write locally.

    Local disk mirrors the repo, so a failed commit never looks like a stored day. Its chains
    wait in `unpublished` and the commit is retried first thing next pass -- before any new
    fetch, so a dead token costs one round of Yahoo calls, not one per pass. A market that
    isn't open (unscheduled closure) ends the pass at the first ticker.
    """
    out = {"stored": 0, "skipped": 0, "failed": 0, "committed": None}
    if unpublished:
        out["committed"] = _publish()               # raises again -> pass ends, nothing fetched
    today, failures = today_et(), []
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")
    try:
        for tk in tickers or load_watchlist():
            if data.has_chain(tk, today):          # committed earlier today: no Yahoo call
                out["skipped"] += 1
                continue
            try:
                status, detail, compact, state = pool.submit(_fetch_compact, tk).result(timeout=TICKER_TIMEOUT)
            except FutureTimeout:
                status, detail, compact, state = "failed", f"{tk}: no response in {TICKER_TIMEOUT}s", None, None
                pool.shutdown(wait=False)
                pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")   # abandon the stuck worker
            out[status] += 1
            if status == "failed":
                failures.append(detail)
            elif status == "stored":
                unpublished[(tk.upper(), str(compact["QUOTE_DATE"].iloc[0]))] = compact
            elif state and state != "REGULAR":
                out["market"] = state
                break
            time.sleep(1.0)   # spread ~300 Yahoo requests out; a burst gets rate-limited
    finally:
        pool.shutdown(wait=False)
    if failures:
        out["failures"] = failures[:5]
    if unpublished:
        out["committed"] = _publish()
    return out


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
    """ivlib over every given day -> one row per date."""
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


def _implied_gaps(ticker: str, have: pd.Index) -> pd.DataFrame:
    """Solve stored days missing from `have` (a race, a CLI snapshot): most recent first, capped."""
    missing = [d for d in data.chain_days(ticker) if pd.Timestamp(d) not in have][-GAP_FILL_MAX:]
    return implied_series(data.read_chains(ticker, missing)) if missing else pd.DataFrame()


def implied_history(ticker: str) -> pd.DataFrame:
    """Vendor series + data/implied.parquet + any stored day it lacks. Work stays bounded as history grows."""
    ticker = ticker.upper()
    have = data.read_implied(ticker)
    parts = [p for p in (data.vendor_implied(ticker), have, _implied_gaps(ticker, have.index)) if not p.empty]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="last")].sort_index()


def build_metrics(ticker: str, price_period: str = "5y") -> pd.DataFrame:
    """Implied + realized + Cboe, joined on date. Cboe is optional: FRED being down must not blank the page."""
    ticker = ticker.upper()
    implied = implied_history(ticker)
    parts = [implied] if len(implied) else []   # an empty frame poisons the index dtype on pandas 3
    close = sources.price_history(ticker, period=price_period)
    parts += [realized_vol(close), close.rename("close")]
    if sources.cboe_available(ticker):
        try:
            parts.append(sources.cboe_index(ticker))
        except Exception:
            pass
    m = pd.concat(parts, axis=1, sort=True)
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


def backfill_implied() -> int:
    """Rebuild data/implied.parquet from every stored chain (one-off, or after an engine change)."""
    rows = [s.reset_index().assign(ticker=tk) for tk in data.tickers()
            if not (s := implied_series(data.read_chains(tk))).empty]
    if not rows:
        print("no stored chains"); return 2
    df = pd.concat(rows, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    data.write_bytes(data.IMPLIED, data.to_bytes(df))
    print(f"{len(df)} rows, {df['ticker'].nunique()} tickers -> {data.IMPLIED.relative_to(data.ROOT.parent)}")
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
    cmds = {"snapshot", "compute", "implied", "vendor"}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "vendor":
        return vendor()
    if cmd == "implied":
        return backfill_implied()
    force = "--force" in rest
    tickers = [a for a in rest if not a.startswith("--")]
    return snapshot(tickers, force) if cmd == "snapshot" else compute(tickers)


if __name__ == "__main__":
    sys.exit(main())
