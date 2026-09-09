"""Quote quality filtering, with per-reason accounting.

Why this is its own module
--------------------------
On a single real AAPL expiry, 114 strikes were listed but only 53 had two-sided
markets. The other 61 were zero-bid: strikes nobody was willing to buy at any
price. The original code fed rows like these straight into a Newton solver and
stored whatever came back.

Discarding bad quotes is easy. The part that matters is *reporting* what was
discarded and why, so that a coverage number can be defended rather than
hand-waved. Every function here returns a mask plus a breakdown.

The checks, in order of severity
--------------------------------
crossed      bid > ask -- the quote is nonsense, not merely wide.
zero_bid     bid <= 0 -- nobody bids; the "price" carries no information.
wide_spread  spread as a fraction of mid exceeds a threshold. A quote with a
             50%-wide market does not pin down a price, so it cannot pin down a
             volatility either.
stale        last trade far from the current mid (optional; needs last price).
dte          outside the usable maturity range. Very short-dated options carry
             almost no vega, so their implied vol is numerically meaningless.

Note the distinction from solver.feasible: that checks whether an implied vol can
exist *mathematically* (no-arbitrage bounds). This module checks whether the quote
is *trustworthy enough to bother*. Both are needed, and they reject different rows.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_SPREAD_PCT = 0.20   # spread must be under 20% of mid
DEFAULT_MIN_MID = 0.01          # sub-penny mids are noise
DEFAULT_MIN_DTE = 5             # under a week, vega is too small to be useful
DEFAULT_MAX_DTE = 400


def mid(bid, ask):
    """Bid-ask midpoint -- the market's current opinion.

    Preferred over the last traded price, which may be hours stale on an
    illiquid strike. Measured on real quotes, using last-trade instead of mid
    shifted median implied vol by ~7%.
    """
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    return 0.5 * (bid + ask)


def spread_pct(bid, ask):
    """Bid-ask spread as a fraction of the midpoint."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    m = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, (ask - bid) / m, np.inf)


def quote_checks(
    bid, ask, dte=None, last=None,
    max_spread_pct=DEFAULT_MAX_SPREAD_PCT,
    min_mid=DEFAULT_MIN_MID,
    min_dte=DEFAULT_MIN_DTE,
    max_dte=DEFAULT_MAX_DTE,
    max_stale_pct=None,
):
    """Return an ordered dict of check name -> boolean pass mask."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    m = mid(bid, ask)

    checks = {
        "finite": np.isfinite(bid) & np.isfinite(ask),
        "crossed": ~(bid > ask),
        "zero_bid": bid > 0,
        "min_mid": m >= min_mid,
        "wide_spread": spread_pct(bid, ask) <= max_spread_pct,
    }
    if dte is not None:
        d = np.asarray(dte, dtype=float)
        checks["dte_range"] = (d >= min_dte) & (d <= max_dte)
    if last is not None and max_stale_pct is not None:
        lastv = np.asarray(last, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            drift = np.where(m > 0, np.abs(lastv - m) / m, np.inf)
        checks["stale"] = ~np.isfinite(lastv) | (drift <= max_stale_pct)
    return checks


def filter_quotes(bid, ask, **kwargs):
    """Apply all checks and report a waterfall of rejections.

    Returns (mask, report). The report attributes each rejected quote to the
    *first* check it failed, so the per-reason counts sum to the total rejected
    and can be read as a funnel rather than as overlapping tallies.
    """
    checks = quote_checks(bid, ask, **kwargs)
    n = len(np.asarray(bid, dtype=float))

    remaining = np.ones(n, dtype=bool)
    waterfall = {}
    for name, passed in checks.items():
        failed_here = remaining & ~passed
        waterfall[name] = int(failed_here.sum())
        remaining = remaining & passed

    report = {
        "total": int(n),
        "kept": int(remaining.sum()),
        "kept_pct": round(100.0 * remaining.sum() / n, 2) if n else 0.0,
        "rejected_by": waterfall,
    }
    return remaining, report


def paired_mask(call_mask, put_mask):
    """Strikes where BOTH the call and the put are usable.

    Required by the parity fit, which needs C and P at the same strike. A strike
    with a good call and a dead put is unusable for recovering the forward even
    though the call itself is fine.
    """
    return np.asarray(call_mask, dtype=bool) & np.asarray(put_mask, dtype=bool)


def format_report(report, title="quote filter"):
    """Human-readable funnel, for logs and notebooks."""
    lines = [f"{title}: {report['kept']}/{report['total']} kept ({report['kept_pct']}%)"]
    for name, cnt in report["rejected_by"].items():
        if cnt:
            pct = 100.0 * cnt / report["total"]
            lines.append(f"    -{cnt:>7,d}  ({pct:5.2f}%)  {name}")
    return "\n".join(lines)
