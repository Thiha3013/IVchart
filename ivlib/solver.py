"""Implied vol inversion: bracketed Newton with bisection fallback.

`implied_vol`      vectorized NumPy, the reference implementation
`implied_vol_fast` same algorithm as a fused Numba kernel (~10x, all cores)

Price is monotonic in sigma, so the root is unique when the quote is inside the
no-arb bounds. A Newton step is taken only if it lands inside the maintained
bracket; otherwise bisect. Cannot freeze or diverge. Infeasible quotes -> NaN.
Convergence is on sigma, not price (absolute price tolerance fails in the wings).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ivlib import pricing as bs

SIGMA_MIN = 1e-6
SIGMA_MAX = 5.0
DEFAULT_TOL = 1e-12      # on sigma
DEFAULT_MAX_ITER = 60


def feasible(target, F, K, T, df=1.0, is_call=True):
    """Quotes strictly inside the no-arb bounds; nothing else has an implied vol."""
    target = np.asarray(target, dtype=float)
    lo = bs.intrinsic(F, K, df, is_call)
    hi = bs.upper_bound(F, K, df, is_call)
    return np.isfinite(target) & (np.asarray(T) > 0) & (target > lo + 1e-12) & (target < hi - 1e-12)


def implied_vol(target, F, K, T, df=1.0, is_call=True, tol=DEFAULT_TOL,
                max_iter=DEFAULT_MAX_ITER, return_info=False, seed_fn=None):
    target, F, K, T, df, is_call = np.broadcast_arrays(
        np.asarray(target, dtype=float), np.asarray(F, dtype=float),
        np.asarray(K, dtype=float), np.asarray(T, dtype=float),
        np.asarray(df, dtype=float), np.asarray(is_call, dtype=bool))

    ok = feasible(target, F, K, T, df, is_call)
    sigma = np.full(target.shape, np.nan)
    if not ok.any():
        info = {"feasible": ok, "converged": np.zeros(target.shape, bool), "iterations": 0}
        return (sigma, info) if return_info else sigma

    t, f, k, tt, d, c = (a[ok] for a in (target, F, K, T, df, is_call))
    if seed_fn is None:
        s = bs.brenner_subrahmanyam(t, f, k, tt, d)
    elif seed_fn is bs.corrado_miller:
        s = seed_fn(t, f, k, tt, d, c)
    else:
        s = seed_fn(t, f, k, tt, d)
    s = np.clip(s, SIGMA_MIN, SIGMA_MAX)
    lo, hi = np.full(s.shape, SIGMA_MIN), np.full(s.shape, SIGMA_MAX)
    converged = np.zeros(s.shape, bool)
    n_iter = np.zeros(s.shape, np.int32)

    used = 0
    for used in range(1, max_iter + 1):
        active = ~converged
        if not active.any():
            used -= 1
            break
        i = np.flatnonzero(active)
        n_iter[i] += 1
        si = s[i]
        diff = bs.price(f[i], k[i], si, tt[i], d[i], c[i]) - t[i]
        v = bs.vega(f[i], k[i], si, tt[i], d[i])

        too_high = diff > 0
        hi[i] = np.where(too_high, si, hi[i])
        lo[i] = np.where(too_high, lo[i], si)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = si - diff / v
        usable = np.isfinite(newton) & (v > 1e-300) & (newton > lo[i]) & (newton < hi[i])
        step = np.where(usable, newton, 0.5 * (lo[i] + hi[i]))
        s[i] = step
        converged[i[(np.abs(step - si) < tol) | (hi[i] - lo[i] < tol)]] = True

    sigma[ok] = s
    if not return_info:
        return sigma
    conv_full = np.zeros(target.shape, bool); conv_full[ok] = converged
    iters_full = np.zeros(target.shape, np.int32); iters_full[ok] = n_iter
    return sigma, {
        "feasible": ok, "converged": conv_full,
        "iterations": used, "mean_iterations": float(n_iter.mean()),
        "max_iterations": int(n_iter.max()), "iters_per_element": iters_full,
    }


def coverage(info, n=None):
    n = n or info["feasible"].size
    return {
        "total": int(n),
        "feasible": int(info["feasible"].sum()),
        "solved": int(info["converged"].sum()),
        "coverage_pct": round(100.0 * info["converged"].sum() / n, 3),
        "rejected_no_arb_pct": round(100.0 * (~info["feasible"]).sum() / n, 3),
        "mean_iterations": round(info.get("mean_iterations", float("nan")), 2),
        "max_iterations": info.get("max_iterations"),
    }


# ---------------------------------------------------------------- numba kernel
# Fused per-element loop: one option's intermediates stay in registers, easy
# options exit early, prange over cores. First call compiles (~2 s), then cached.

_INV_SQRT_2 = 0.7071067811865476
_INV_SQRT_2PI = 0.3989422804014327


@njit(cache=True, inline="always")
def _cdf(x):
    return 0.5 * (1.0 + math.erf(x * _INV_SQRT_2))


@njit(cache=True, inline="always")
def _pdf(x):
    return _INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(cache=True, inline="always")
def _price_vega(F, K, sigma, T, df, is_call):
    vst = sigma * math.sqrt(T)
    if vst <= 0.0:
        intr = (F - K) if is_call else (K - F)
        return (df * intr if intr > 0.0 else 0.0), 0.0
    d1 = (math.log(F / K) + 0.5 * vst * vst) / vst
    d2 = d1 - vst
    px = df * (F * _cdf(d1) - K * _cdf(d2)) if is_call else df * (K * _cdf(-d2) - F * _cdf(-d1))
    return px, df * F * _pdf(d1) * math.sqrt(T)


@njit(cache=True, inline="always")
def _seed(target, F, K, T, df, is_call):
    """Corrado-Miller with ATM fallback; mirrors pricing.corrado_miller."""
    c = target / df
    if not is_call:
        c = c + (F - K)
    m = F - K
    a = c - 0.5 * m
    disc = a * a - m * m * bs.INV_PI
    if disc > 0.0:
        s = (bs.SQRT_2PI / (F + K) * (a + math.sqrt(disc))) / math.sqrt(T)
        if 1e-3 < s < 4.9:
            return s
    s = math.sqrt(2.0 * math.pi / T) * (c / F)
    return 0.05 if s < 0.05 else (1.5 if s > 1.5 else s)


@njit(parallel=True, cache=True)
def _kernel(target, F, K, T, df, is_call, tol, max_iter):
    n = target.shape[0]
    out = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int32)
    for i in prange(n):
        t, f, k, tt, d, c = target[i], F[i], K[i], T[i], df[i], is_call[i]
        if not (np.isfinite(t) and tt > 0.0 and f > 0.0 and k > 0.0):
            continue
        intr = (f - k) if c else (k - f)
        lo_b = d * (intr if intr > 0.0 else 0.0)
        hi_b = d * (f if c else k)
        if t <= lo_b + 1e-12 or t >= hi_b - 1e-12:
            continue

        s = min(max(_seed(t, f, k, tt, d, c), SIGMA_MIN), SIGMA_MAX)
        lo, hi = SIGMA_MIN, SIGMA_MAX
        for it in range(1, max_iter + 1):
            px, v = _price_vega(f, k, s, tt, d, c)
            diff = px - t
            if diff > 0.0:
                hi = s
            else:
                lo = s
            step = s - diff / v if v > 1e-300 else 0.5 * (lo + hi)
            if not (lo < step < hi) or not np.isfinite(step):
                step = 0.5 * (lo + hi)
            moved = abs(step - s)
            s = step
            iters[i] = it
            if moved < tol or (hi - lo) < tol:
                break
        out[i] = s
    return out, iters


def implied_vol_fast(target, F, K, T, df=1.0, is_call=True, tol=DEFAULT_TOL,
                     max_iter=DEFAULT_MAX_ITER, return_info=False):
    """Drop-in for implied_vol. Same answers to ~1e-11."""
    shape = np.shape(np.asarray(target))
    target, F, K, T, df = (np.ascontiguousarray(np.broadcast_to(np.asarray(a, dtype=np.float64), shape))
                           for a in (target, F, K, T, df))
    is_call = np.ascontiguousarray(np.broadcast_to(np.asarray(is_call, dtype=np.bool_), shape))
    out, iters = _kernel(target, F, K, T, df, is_call, tol, max_iter)
    if not return_info:
        return out
    conv = np.isfinite(out)
    return out, {
        "feasible": conv, "converged": conv,
        "iterations": int(iters.max()), "max_iterations": int(iters.max()),
        "mean_iterations": float(iters[conv].mean()) if conv.any() else 0.0,
        "iters_per_element": iters,
    }
