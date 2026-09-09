"""Numba-compiled implied volatility kernel.

Same algorithm as solver.implied_vol -- bracketed Newton with bisection
fallback -- but fused into a single per-element loop.

Why this is faster than the NumPy version, which is already vectorized:

  * The NumPy solver makes many passes over memory. Each call to bs.price
    allocates temporaries for d1, d2, two CDF evaluations, and the combination;
    at 260k elements those arrays do not fit in cache, so every iteration pays
    main-memory bandwidth several times over.
  * The fused kernel keeps one option's intermediates in registers for the whole
    solve, touching memory once to read inputs and once to write the answer.
  * Every element converges independently. In the vectorized version the whole
    array iterates until the slowest element finishes (masked, but still looping);
    here an easy option exits after 3 iterations and its lane moves on.
  * prange spreads the elements across cores.

The tradeoff is a compile on first call (a few seconds), which is why the
benchmark harness warms the kernel before timing it.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

SIGMA_MIN = 1e-6
SIGMA_MAX = 5.0
SQRT_2PI = 2.5066282746310002
INV_SQRT_2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327
INV_PI = 0.3183098861837907


@njit(cache=True, fastmath=False, inline="always")
def _cdf(x):
    return 0.5 * (1.0 + math.erf(x * INV_SQRT_2))


@njit(cache=True, fastmath=False, inline="always")
def _pdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True, inline="always")
def _price_vega(F, K, sigma, T, df, is_call):
    """Black-76 price and vega in one pass -- they share d1."""
    vst = sigma * math.sqrt(T)
    if vst <= 0.0:
        intr = (F - K) if is_call else (K - F)
        return (df * intr if intr > 0.0 else 0.0), 0.0
    d1 = (math.log(F / K) + 0.5 * vst * vst) / vst
    d2 = d1 - vst
    if is_call:
        px = df * (F * _cdf(d1) - K * _cdf(d2))
    else:
        px = df * (K * _cdf(-d2) - F * _cdf(-d1))
    vega = df * F * _pdf(d1) * math.sqrt(T)
    return px, vega


@njit(cache=True, inline="always")
def _seed(target, F, K, T, df, is_call):
    """Corrado-Miller with an ATM fallback -- matches ivlib.seed."""
    c = target / df
    if not is_call:
        c = c + (F - K)
    m = F - K
    a = c - 0.5 * m
    disc = a * a - m * m * INV_PI
    if disc > 0.0:
        s = (SQRT_2PI / (F + K) * (a + math.sqrt(disc))) / math.sqrt(T)
        if s > 1e-3 and s < 4.9:
            return s
    s = math.sqrt(2.0 * math.pi / T) * (c / F)
    if s < 0.05:
        return 0.05
    if s > 1.5:
        return 1.5
    return s


@njit(parallel=True, cache=True)
def implied_vol_kernel(target, F, K, T, df, is_call, tol, max_iter):
    n = target.shape[0]
    out = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int32)

    for i in prange(n):
        t = target[i]; f = F[i]; k = K[i]; tt = T[i]; d = df[i]; c = is_call[i]

        if not (np.isfinite(t) and tt > 0.0 and f > 0.0 and k > 0.0):
            continue
        # No-arbitrage feasibility.
        intr = (f - k) if c else (k - f)
        lo_b = d * (intr if intr > 0.0 else 0.0)
        hi_b = d * (f if c else k)
        if t <= lo_b + 1e-12 or t >= hi_b - 1e-12:
            continue

        s = _seed(t, f, k, tt, d, c)
        if s < SIGMA_MIN:
            s = SIGMA_MIN
        elif s > SIGMA_MAX:
            s = SIGMA_MAX
        lo = SIGMA_MIN
        hi = SIGMA_MAX

        for it in range(1, max_iter + 1):
            px, v = _price_vega(f, k, s, tt, d, c)
            diff = px - t
            if diff > 0.0:
                hi = s
            else:
                lo = s

            if v > 1e-300:
                step = s - diff / v
            else:
                step = 0.5 * (lo + hi)
            if not (step > lo and step < hi) or not np.isfinite(step):
                step = 0.5 * (lo + hi)

            moved = step - s
            if moved < 0.0:
                moved = -moved
            s = step
            iters[i] = it
            if moved < tol or (hi - lo) < tol:
                break

        out[i] = s
    return out, iters


def implied_vol(target, F, K, T, df=1.0, is_call=True, tol=1e-12, max_iter=60,
                return_info=False):
    """Numba-backed drop-in for solver.implied_vol."""
    target, F, K, T, df = (np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), np.shape(np.asarray(target))))
                           for a in (target, F, K, T, df))
    is_call = np.ascontiguousarray(np.broadcast_to(np.asarray(is_call, dtype=np.bool_), target.shape))
    out, iters = implied_vol_kernel(target, F, K, T, df, is_call, tol, max_iter)
    if return_info:
        conv = np.isfinite(out)
        return out, {
            "feasible": conv, "converged": conv,
            "iterations": int(iters.max()),
            "mean_iterations": float(iters[conv].mean()) if conv.any() else 0.0,
            "max_iterations": int(iters.max()),
            "iters_per_element": iters,
        }
    return out
