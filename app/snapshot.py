"""The daily job. Fetch today's chain for every watched ticker and store it.

This is the whole "continuous" part of the project: history for any ticker
exists only from the day this first runs for it. Run it once per trading day,
during regular hours, and the dataset builds itself.

Rules it enforces:

  * Refuses to store a chain captured outside regular hours (see the Yahoo
    source for why: bid/ask are zero, so the chain carries nothing). Pass
    --force to override, which is only useful for testing the plumbing.
  * One snapshot per ticker per calendar day. A second run the same day is a
    no-op unless --force.
  * Never lets one bad ticker stop the others. Failures are reported at the end
    and the exit code reflects them, so a scheduler can see something went wrong.

Usage:
    python -m app.snapshot                      # everything in app/watchlist.txt
    python -m app.snapshot AAPL MSFT            # just these
    python -m app.snapshot --force              # ignore the market-hours guard
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from app import schema, store
from app.sources import yahoo

WATCHLIST = Path(__file__).resolve().parent / "watchlist.txt"


def load_watchlist(path: Path = WATCHLIST) -> list[str]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip().upper()
        if s:
            out.append(s)
    return out


def snapshot_one(ticker: str, force: bool = False) -> tuple[str, str]:
    """Returns (status, detail). status in {'stored', 'skipped', 'failed'}."""
    try:
        chain = yahoo.fetch(ticker)
    except yahoo.ChainUnavailable as e:
        return "failed", str(e)
    except Exception as e:  # network, parsing -- anything yfinance throws
        return "failed", f"{ticker}: {type(e).__name__}: {e}"

    day = str(chain["QUOTE_DATE"].iloc[0])
    state = str(chain["MARKET_STATE"].iloc[0])

    if not force and store.has_chain(ticker, day):
        return "skipped", f"{ticker}: already have {day}"
    if not force and not yahoo.is_live(chain):
        two_sided = float(((chain["C_BID"] > 0) & (chain["C_ASK"] > 0)).mean())
        return "skipped", (f"{ticker}: market state {state}, "
                           f"{two_sided:.0%} two-sided quotes -- not storing")

    compact = schema.compact(chain)
    p = store.write_chain(compact)
    return "stored", f"{ticker}: {day} {len(compact)} rows -> {p.relative_to(store.ROOT.parent)}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    force = "--force" in argv
    tickers = [a for a in argv if not a.startswith("--")] or load_watchlist()
    if not tickers:
        print("nothing to do: no tickers given and watchlist is empty")
        return 2

    t0 = time.perf_counter()
    counts = {"stored": 0, "skipped": 0, "failed": 0}
    for tk in tickers:
        status, detail = snapshot_one(tk, force=force)
        counts[status] += 1
        print(f"[{status:7}] {detail}")

    print(f"\n{counts['stored']} stored, {counts['skipped']} skipped, "
          f"{counts['failed']} failed  [{time.perf_counter()-t0:.1f}s]")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
