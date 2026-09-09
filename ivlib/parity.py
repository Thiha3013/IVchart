"""Recover the forward price and discount factor from option quotes themselves.

The idea
--------
Put-call parity is an arbitrage identity, not a model. For a call and put on the
same underlying with the same strike K and expiry:

    C - K = df * (F - K)                                                  (1)

where df is the discount factor to expiry and F is the forward price. Rearranged:

    (C - P) = (df * F) - (df) * K
     \\_____/    \\______/   \\__/
        y      intercept   slope * K

That is a straight line in K. So if we take every strike on one expiry, plot
(C - P) against K, and fit a line:

    slope     = -df          ->  df = -slope
    intercept =  df * F      ->  F  = intercept / df

Both unknowns fall out of the quotes. Note carefully that F is *not* the
intercept -- the intercept is the discounted forward, and dividing by df is
required. (Getting this backwards is an easy and quiet error.)

Why bother
----------
The original code downloaded DGS10 -- a 10-year Treasury yield -- and used it for
30-day options, while ignoring dividends entirely. Both problems disappear here:
the forward already contains the market's own view of the financing rate, the
dividend, and the borrow cost, because those are exactly what market makers price
into the call-put spread. Nothing is downloaded, and the numbers are guaranteed
consistent with the very quotes being inverted.

Relation to market practice
---------------------------
Cboe's VIX methodology also derives its forward from put-call parity, but uses a
*single* strike -- the one with the smallest |C - P| -- and does separately source
a Treasury curve. The regression used here is a variant: it uses every strike, so
it is less sensitive to one bad quote, and it needs no external rate at all.
"""

from __future__ import annotations

import numpy as np

MIN_STRIKES = 3          # a line through 2 points has no residual to check
MAX_DISCOUNT = 1.02      # df above this implies a large negative rate
MIN_DISCOUNT = 0.70      # df below this implies an implausible one


def _grouped_wls(x, y, w, g, ngroups):
    """Weighted least squares of y on x, computed for every group at once.

    Uses bincount to accumulate the five sums a closed-form linear fit needs,
    which keeps this a handful of vectorized passes regardless of how many
    expiries are present. Returns (slope, intercept, n, r2) per group.
    """
    sw = np.bincount(g, w, ngroups)
    swx = np.bincount(g, w * x, ngroups)
    swy = np.bincount(g, w * y, ngroups)
    swxx = np.bincount(g, w * x * x, ngroups)
    swxy = np.bincount(g, w * x * y, ngroups)
    swyy = np.bincount(g, w * y * y, ngroups)
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
    """Fit F and df per expiry group from paired call/put quotes.

    Parameters
    ----------
    K, call_mid, put_mid : arrays of equal length, one entry per strike.
    group : integer group codes (one per expiry). None means a single group.
    weights : per-quote weights. Near-the-money quotes are the reliable ones,
        so weighting by them is usually better than an unweighted fit.
    trim : run a second pass that drops points more than 3 robust deviations
        from the first fit. Stale far-wing quotes are the usual culprits and a
        single bad one can visibly tilt the line.

    Returns a dict of per-group arrays: forward, discount, rate_x_T, n, r2, ok.
    `ok` flags groups that produced a usable, plausible fit.
    """
    K = np.asarray(K, dtype=float)
    y = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)

    if group is None:
        group = np.zeros(K.shape, dtype=np.intp)
    group = np.asarray(group, dtype=np.intp)
    ngroups = int(group.max()) + 1 if group.size else 0

    w = np.ones(K.shape) if weights is None else np.asarray(weights, dtype=float)
    valid = np.isfinite(K) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    w = np.where(valid, w, 0.0)

    slope, intercept, n, r2 = _grouped_wls(K, y, w, group, ngroups)

    if trim:
        pred = intercept[group] + slope[group] * K
        resid = np.abs(y - pred)
        # Robust scale per group: median absolute residual, via a sort-free
        # approximation using the weighted mean of |resid|, scaled up.
        scale = np.bincount(group, w * resid, ngroups) / np.maximum(
            np.bincount(group, w, ngroups), 1e-12
        )
        keep = resid <= 3.0 * np.maximum(scale[group], 1e-9)
        w2 = np.where(keep, w, 0.0)
        # Only re-fit groups that still have enough surviving points.
        n_keep = np.bincount(group, (w2 > 0).astype(float), ngroups)
        s2, i2, _, r22 = _grouped_wls(K, y, w2, group, ngroups)
        refit = n_keep >= MIN_STRIKES
        slope = np.where(refit, s2, slope)
        intercept = np.where(refit, i2, intercept)
        r2 = np.where(refit, r22, r2)
        n = np.where(refit, n_keep, n)

    with np.errstate(divide="ignore", invalid="ignore"):
        discount = -slope
        forward = intercept / discount          # NOT the intercept itself
        rate_x_T = -np.log(discount)            # r*T; divide by T for the rate

    ok = (
        (n >= MIN_STRIKES)
        & np.isfinite(forward)
        & (forward > 0)
        & np.isfinite(discount)
        & (discount > MIN_DISCOUNT)
        & (discount < MAX_DISCOUNT)
    )
    return {
        "forward": forward,
        "discount": discount,
        "rate_x_T": rate_x_T,
        "n": n,
        "r2": r2,
        "ok": ok,
    }


def implied_rate(discount, T):
    """Continuously-compounded rate implied by a discount factor: r = -ln(df)/T."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.log(np.asarray(discount, dtype=float)) / np.asarray(T, dtype=float)


def group_codes(*keys):
    """Map one or more parallel key arrays to dense integer group codes.

    Typical use: group_codes(quote_date, expire_date) to get one group per
    (date, expiry) surface slice.

    Each key is factorized independently and the results are combined, rather
    than stacking the keys into a 2-D array first. Stacking looks tidier but
    breaks on object-dtype input -- which is exactly what pandas hands back for
    string columns, so it fails on real data while passing on synthetic arrays
    built from numpy string types.
    """
    if not keys:
        raise ValueError("group_codes requires at least one key array")

    combined = None
    for k in keys:
        _, code = np.unique(np.asarray(k), return_inverse=True)
        code = code.ravel().astype(np.int64)
        if combined is None:
            combined = code
        else:
            combined = combined * (int(code.max()) + 1) + code
    _, dense = np.unique(combined, return_inverse=True)
    return dense.ravel().astype(np.intp)
