"""Vectorized implied volatility inversion.

The problem: given an observed option price, find the sigma that reproduces it.
Black-76 price is strictly increasing in sigma, so when a solution exists it is
unique. That is what makes the inversion tractable.

Why the original Newton loop failed
-----------------------------------
    sigma <- sigma - (price(sigma) - target) / vega(sigma)

Three independent problems, none of which the original code checked for:

1. Vega collapses toward zero for deep ITM/OTM options and as T -> 0. Dividing
   by ~0 produces an enormous step, sigma goes negative, log/sqrt of a negative
   yields NaN, and the NaN then propagates silently through all 100 iterations.

2. There was no convergence test. Every option got exactly 100 iterations --
   wasteful when it converged in 4, and undetected when it never converged.

3. There was no feasibility check. For a quote outside the no-arbitrage bounds,
   *no* sigma reproduces it. Newton was run anyway and returned noise.

The fix here is a hybrid: vectorized Newton for speed, with a per-element
bisection fallback for whatever fails to converge. Bisection cannot diverge once
a root is bracketed -- but note it also cannot manufacture a bracket that does
not exist, which is why infeasible quotes must be filtered out *first*.
"""

from __future__ import annotations

import numpy as np

from ivlib import bs, seed as _seed

SIGMA_MIN = 1e-6
SIGMA_MAX = 5.0        # 500% vol -- far beyond any real listed equity option
# Convergence is tested on *sigma*, not on price.
#
# An absolute price tolerance is scale-dependent and fails badly in the wings: a
# 7-day 5%-OTM call is worth ~1.3e-6 with vega ~3.5e-4, so |price error| < 1e-8
# is satisfied while sigma is still wrong by 3e-5. Testing the step size in sigma
# is scale-free and converges to machine precision everywhere.
DEFAULT_TOL = 1e-12    # on sigma
DEFAULT_MAX_ITER = 60


def feasible(target, F, K, T, df=1.0, is_call=True):
    """Boolean mask of quotes for which an implied vol can exist at all.

    A price must sit strictly inside the no-arbitrage bounds:
        df*max(F-K, 0)  <  C  <  df*F        (call)
        df*max(K-F, 0)  <  P  <  df*K        (put)

    Quotes outside these are stale prints, crossed markets, or zero-bid strikes
    nobody traded. They are not solvable and must not be handed to a solver.
    """
    target = np.asarray(target, dtype=float)
    lo = bs.intrinsic(F, K, df, is_call)
    hi = bs.upper_bound(F, K, df, is_call)
    return (
        np.isfinite(target)
        & (np.asarray(T) > 0)
        & (target > lo + 1e-12)
        & (target < hi - 1e-12)
    )


def initial_guess(target, F, K, T, df=1.0):
    """Brenner-Subrahmanyam ATM seed, sigma ~ sqrt(2*pi/T) * C/F.

    Exact at the money, decent nearby, mediocre in the wings -- but any sane
    starting point beats the original code's use of 30-day realized vol, which
    could be NaN for the first 30 rows of the sample and dragged the solver
    through far more iterations than necessary.

    This is the rung the ladder later improves on: a better seed means fewer
    iterations, which is an algorithmic speedup rather than a hardware one.
    """
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    seed = np.sqrt(2.0 * np.pi / T) * (np.asarray(target) / (df * F))
    return np.clip(seed, 0.05, 1.5)


def implied_vol(
    target, F, K, T, df=1.0, is_call=True,
    tol=DEFAULT_TOL, max_iter=DEFAULT_MAX_ITER, return_info=False,
    seed_fn=None,
):
    """Invert Black-76 for sigma. Fully vectorized, safeguarded Newton.

    Algorithm: Newton, but with a maintained bracket [lo, hi] that is guaranteed
    to contain the root. Each iteration narrows the bracket using the sign of the
    residual (valid because price is monotonic in sigma). A Newton step is taken
    only if it lands *inside* the current bracket and vega is usable; otherwise a
    bisection step is taken instead.

    This is what makes the solver unconditionally convergent. Plain Newton can
    overshoot into a region where vega ~ 0, at which point the step size collapses
    to zero and the element freezes -- neither converged nor progressing. That is
    a real failure mode, not a hypothetical: with a naive ATM seed applied across
    all strikes, ~45% of a 500-strike grid froze permanently that way.

    Worst case here is bisection's linear rate; typical case is Newton's quadratic
    one. Infeasible quotes return NaN explicitly.
    """
    arrs = np.broadcast_arrays(
        np.asarray(target, dtype=float), np.asarray(F, dtype=float),
        np.asarray(K, dtype=float), np.asarray(T, dtype=float),
        np.asarray(df, dtype=float), np.asarray(is_call, dtype=bool),
    )
    target, F, K, T, df, is_call = arrs

    ok = feasible(target, F, K, T, df, is_call)
    sigma = np.full(target.shape, np.nan, dtype=float)
    empty = {"feasible": ok, "converged": np.zeros(target.shape, bool), "iterations": 0}
    if not ok.any():
        return (sigma, empty) if return_info else sigma

    t, f, k, tt, d, c = (a[ok] for a in (target, F, K, T, df, is_call))
    if seed_fn is None:
        s = initial_guess(t, f, k, tt, d)
    elif seed_fn is _seed.corrado_miller:
        s = seed_fn(t, f, k, tt, d, c)
    else:
        s = seed_fn(t, f, k, tt, d)
    s = np.clip(s, SIGMA_MIN, SIGMA_MAX)
    lo = np.full(s.shape, SIGMA_MIN)
    hi = np.full(s.shape, SIGMA_MAX)
    converged = np.zeros(s.shape, dtype=bool)
    # Per-element iteration count. Mean iterations is the number a better initial
    # guess reduces, so it is the metric the seed-improvement work is judged on.
    n_iter = np.zeros(s.shape, dtype=np.int32)

    used = 0
    for used in range(1, max_iter + 1):
        active = ~converged
        if not active.any():
            used -= 1  # nothing was done on this pass
            break

        i = np.flatnonzero(active)
        n_iter[i] += 1
        si = s[i]
        diff = bs.price(f[i], k[i], si, tt[i], d[i], c[i]) - t[i]
        v = bs.vega(f[i], k[i], si, tt[i], d[i])

        # Narrow the bracket. Price is increasing in sigma, so a positive
        # residual means the current sigma sits above the root.
        too_high = diff > 0
        hi[i] = np.where(too_high, si, hi[i])
        lo[i] = np.where(too_high, lo[i], si)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            newton = si - diff / v
        # Accept Newton only when it is finite, vega is usable, and it stays
        # strictly inside the bracket. Otherwise bisect.
        usable = np.isfinite(newton) & (v > 1e-300) & (newton > lo[i]) & (newton < hi[i])
        step = np.where(usable, newton, 0.5 * (lo[i] + hi[i]))

        # Always take the step -- it is the refined estimate -- then declare
        # convergence when sigma has stopped moving or the bracket has collapsed.
        s[i] = step
        converged[i[(np.abs(step - si) < tol) | (hi[i] - lo[i] < tol)]] = True

    sigma[ok] = s
    if return_info:
        conv_full = np.zeros(target.shape, dtype=bool)
        conv_full[ok] = converged
        iters_full = np.zeros(target.shape, dtype=np.int32)
        iters_full[ok] = n_iter
        return sigma, {
            "feasible": ok,
            "converged": conv_full,
            "iterations": used,                       # loop passes (worst element)
            "mean_iterations": float(n_iter.mean()),  # per-element average
            "max_iterations": int(n_iter.max()),
            "iters_per_element": iters_full,
        }
    return sigma


def coverage(info, n=None):
    """Solve-coverage stats -- a quality metric, not a side effect."""
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
