"""Initial guesses for the implied volatility solver.

Why this is worth its own module
--------------------------------
Vectorizing the solver is a hardware win: the same arithmetic, executed better.
Improving the *starting point* is an algorithmic win -- it removes arithmetic
that no longer has to happen.

Every Newton iteration costs two normal CDF evaluations plus a PDF, so cutting
the iteration count removes real work.

Measured on 262,796 real AAPL quotes, though, the gain is modest: the mean falls
from 6.89 iterations to 6.26 (-9%), worth about 1.05x end to end. Far less than a
naive reading suggests, and worth understanding why:

  * The convergence test needs |dsigma| < 1e-12. Even from a perfect seed,
    Newton needs ~3-4 iterations to walk down to that, so there is a hard floor
    the seed cannot cross.
  * The mean is dragged up by a tail, not the typical case. Per-element counts on
    real data run p10=3, p50=6, p90=9, p99=18, max=50. The expensive elements are
    ones where a *bracketing* step is needed regardless of the starting point,
    and no seed helps those.

The lesson is worth keeping: the algorithmic rung was the one expected to pay
most and paid least, while swapping scipy.stats.norm for a direct erf call --
pure constant-factor work, no cleverness -- paid ~58x. Measure the rungs.

Two seeds are provided:

brenner_subrahmanyam
    sigma ~ sqrt(2*pi/T) * C / F. Exact at the money, poor in the wings. Cheap.

corrado_miller
    A closed-form approximation that stays usable away from the money by
    accounting for intrinsic value explicitly. Same arithmetic cost, better
    starting point where its discriminant is positive; falls back to the ATM
    seed where it is not.
"""

from __future__ import annotations

import numpy as np

SQRT_2PI = 2.5066282746310002
INV_PI = 0.3183098861837907


def brenner_subrahmanyam(target, F, K, T, df=1.0):
    """ATM approximation. Correct at F == K, degrades away from it."""
    F = np.asarray(F, dtype=float)
    T = np.asarray(T, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.sqrt(2.0 * np.pi / T) * (np.asarray(target) / (df * F))
    return np.clip(s, 0.05, 1.5)


def corrado_miller(target, F, K, T, df=1.0, is_call=True):
    """Corrado-Miller closed-form implied volatility approximation.

    Working in forward space, let c be the undiscounted option price and use
    put-call parity to convert puts to the equivalent call, so one formula
    covers both. With m = (F - K):

        sigma*sqrt(T) ~ sqrt(2*pi)/(F + K) * [ (c - m/2) + sqrt((c - m/2)^2 - m^2/pi) ]

    The discriminant goes negative for prices near the no-arbitrage boundary,
    where the approximation has no real root; those elements fall back to the
    ATM seed rather than producing NaN. The result is clamped into the solver's
    own sigma bounds, since a seed outside the bracket is worse than useless.
    """
    target = np.asarray(target, dtype=float)
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    df = np.asarray(df, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        c = target / df
        # Convert puts to the equivalent call: c_call = c_put + (F - K).
        c = np.where(np.asarray(is_call), c, c + (F - K))

        m = F - K
        a = c - 0.5 * m
        disc = a * a - m * m * INV_PI
        root = np.sqrt(np.maximum(disc, 0.0))
        sigma_sqrt_t = SQRT_2PI / (F + K) * (a + root)
        s_cm = sigma_sqrt_t / np.sqrt(T)

        # Where the discriminant is negative the approximation has no real root.
        # That happens near the no-arbitrage boundary, where volatility is barely
        # determined by price anyway, so fall back to the ATM seed -- clipped to
        # the same sane band, since an unclipped fallback is what makes this seed
        # look worse than the naive one rather than better.
        fallback = np.clip(np.sqrt(2.0 * np.pi / T) * (c / F), 0.05, 1.5)
        valid = np.isfinite(s_cm) & (disc > 0.0) & (s_cm > 0.0)
        s = np.where(valid, np.clip(s_cm, 1e-3, 4.9), fallback)

    return np.nan_to_num(s, nan=0.3, posinf=1.5, neginf=0.05)
