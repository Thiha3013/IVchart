"""Reading the market: quote filtering, and the forward/discount implied by parity.

Filtering rejects quotes that carry no information (zero bid, crossed, wide spread)
and reports why, so coverage is a number rather than an assumption.

Parity: C - P = df*(F - K) is a line in K. Per expiry, slope = -df and
F = intercept / df. Absorbs rate, dividend, and borrow -- nothing is downloaded.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MAX_SPREAD_PCT = 0.20
DEFAULT_MIN_MID = 0.01
DEFAULT_MIN_DTE = 5           # under a week, vega is too small to mean anything
DEFAULT_MAX_DTE = 400

MIN_STRIKES = 3
MAX_DISCOUNT = 1.02
MIN_DISCOUNT = 0.70


# ---------------------------------------------------------------- quote filtering

def mid(bid, ask):
    """Bid-ask mid. Last trade is stale on illiquid strikes (~7% IV error on real data)."""
    return 0.5 * (np.asarray(bid, dtype=float) + np.asarray(ask, dtype=float))


def spread_pct(bid, ask):
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
    m = 0.5 * (bid + ask)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(m > 0, (ask - bid) / m, np.inf)


def quote_checks(bid, ask, dte=None, last=None, max_spread_pct=DEFAULT_MAX_SPREAD_PCT,
                 min_mid=DEFAULT_MIN_MID, min_dte=DEFAULT_MIN_DTE, max_dte=DEFAULT_MAX_DTE,
                 max_stale_pct=None):
    """Ordered dict of check name -> pass mask."""
    bid, ask = np.asarray(bid, dtype=float), np.asarray(ask, dtype=float)
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
    """Returns (mask, report). Each rejection attributed to its first failing check, so counts sum."""
    checks = quote_checks(bid, ask, **kwargs)
    n = len(np.asarray(bid, dtype=float))
    remaining = np.ones(n, bool)
    waterfall = {}
    for name, passed in checks.items():
        waterfall[name] = int((remaining & ~passed).sum())
        remaining &= passed
    return remaining, {
        "total": int(n), "kept": int(remaining.sum()),
        "kept_pct": round(100.0 * remaining.sum() / n, 2) if n else 0.0,
        "rejected_by": waterfall,
    }


def paired_mask(call_mask, put_mask):
    """Strikes where both legs are usable -- parity needs C and P at the same K."""
    return np.asarray(call_mask, bool) & np.asarray(put_mask, bool)


def format_report(report, title="quote filter"):
    lines = [f"{title}: {report['kept']}/{report['total']} kept ({report['kept_pct']}%)"]
    for name, cnt in report["rejected_by"].items():
        if cnt:
            lines.append(f"    -{cnt:>7,d}  ({100.0 * cnt / report['total']:5.2f}%)  {name}")
    return "\n".join(lines)


# ---------------------------------------------------------------- implied forward

def _grouped_wls(x, y, w, g, ngroups):
    """Weighted least squares per group via bincount. Returns (slope, intercept, n, r2)."""
    sw = np.bincount(g, w, ngroups)
    swx, swy = np.bincount(g, w * x, ngroups), np.bincount(g, w * y, ngroups)
    swxx, swxy, swyy = (np.bincount(g, w * x * x, ngroups), np.bincount(g, w * x * y, ngroups),
                        np.bincount(g, w * y * y, ngroups))
    n = np.bincount(g, None, ngroups)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_x = swxx - swx * swx / sw
        cov_xy = swxy - swx * swy / sw
        var_y = swyy - swy * swy / sw
        slope = cov_xy / var_x
        intercept = (swy - slope * swx) / sw
        r2 = np.where(var_y > 0, (cov_xy * cov_xy) / (var_x * var_y), np.nan)
    return slope, intercept, n, r2


def implied_forward(K, call_mid, put_mid, group=None, weights=None, trim=True):
    """Fit F and df per group. Returns dict of per-group arrays: forward, discount, rate_x_T, n, r2, ok."""
    K = np.asarray(K, dtype=float)
    y = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)
    group = np.zeros(K.shape, np.intp) if group is None else np.asarray(group, np.intp)
    ngroups = int(group.max()) + 1 if group.size else 0
    w = np.ones(K.shape) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(K) & np.isfinite(y) & np.isfinite(w) & (w > 0), w, 0.0)

    slope, intercept, n, r2 = _grouped_wls(K, y, w, group, ngroups)

    if trim:  # second pass without points > 3 robust deviations off the first fit
        with np.errstate(invalid="ignore"):
            resid = np.abs(y - (intercept[group] + slope[group] * K))
        scale = np.bincount(group, w * resid, ngroups) / np.maximum(np.bincount(group, w, ngroups), 1e-12)
        w2 = np.where(resid <= 3.0 * np.maximum(scale[group], 1e-9), w, 0.0)
        n_keep = np.bincount(group, (w2 > 0).astype(float), ngroups)
        s2, i2, _, r22 = _grouped_wls(K, y, w2, group, ngroups)
        refit = n_keep >= MIN_STRIKES
        slope, intercept = np.where(refit, s2, slope), np.where(refit, i2, intercept)
        r2, n = np.where(refit, r22, r2), np.where(refit, n_keep, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        discount = -slope
        forward = intercept / discount      # intercept is df*F, not F
        rate_x_T = -np.log(discount)

    ok = ((n >= MIN_STRIKES) & np.isfinite(forward) & (forward > 0)
          & np.isfinite(discount) & (discount > MIN_DISCOUNT) & (discount < MAX_DISCOUNT))
    return {"forward": forward, "discount": discount, "rate_x_T": rate_x_T, "n": n, "r2": r2, "ok": ok}


def implied_rate(discount, T):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(np.asarray(discount, dtype=float)) / np.asarray(T, dtype=float)


def group_codes(*keys):
    """Dense int codes for (key1, key2, ...) tuples. Factorizes per key: stacking breaks on object dtype."""
    if not keys:
        raise ValueError("group_codes requires at least one key array")
    combined = None
    for k in keys:
        _, code = np.unique(np.asarray(k), return_inverse=True)
        code = code.ravel().astype(np.int64)
        combined = code if combined is None else combined * (int(code.max()) + 1) + code
    _, dense = np.unique(combined, return_inverse=True)
    return dense.ravel().astype(np.intp)
