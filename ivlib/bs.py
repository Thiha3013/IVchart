"""Vectorized Black-76 pricing in forward space.

Black-76 is Black-Scholes reparameterized in terms of the forward price F rather
than spot S. It is the *same model*: substituting F = S*exp((r-q)*T) recovers
Black-Scholes exactly. The reason to prefer it here is practical, not theoretical:
the forward can be read directly out of the option market via put-call parity
(see parity.py), which means the interest rate and the dividend yield never have
to be sourced separately. They are already inside F.

Time convention
---------------
T is a year fraction, calendar-based: T = calendar_days / 365.

The original code used T = DTE / 252 where DTE was a *calendar* day count. That
mixes a calendar numerator with a trading-day denominator and overstates T by
~45%, biasing recovered implied vol low by a factor of 0.8309. Both self-consistent
alternatives -- 30/365 = 0.08219 and 21/252 = 0.08333 -- agree to within 1.4%.
See DAY_COUNT in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr

DAYS_PER_YEAR = 365.0
DAY_COUNT = "ACT/365 (calendar days over 365)"

# Standard normal CDF via scipy's ndtr, which is a direct C implementation and
# the fastest vectorized option available (~13.7ms for 1e6 values on this machine).
#
# Note this is NOT the same as scipy.stats.norm.cdf, which wraps ndtr in a large
# amount of distribution-object machinery. In a scalar Python loop that wrapper
# dominates runtime -- it is the single reason the original iterrows() code took
# 352s rather than 73s. Same math, ~5x the cost per call.
_INV_SQRT_2PI = 0.3989422804014327


def norm_cdf(x):
    """Standard normal CDF. Vectorized."""
    return ndtr(x)


def norm_pdf(x):
    """Standard normal PDF."""
    return _INV_SQRT_2PI * np.exp(-0.5 * np.square(x))


def year_fraction(calendar_days):
    """Convert a calendar-day count to a year fraction.

    This is the single place the day-count convention is defined. Nothing else
    in the library is allowed to divide by a day count.
    """
    return np.asarray(calendar_days, dtype=float) / DAYS_PER_YEAR


def d1_d2(F, K, sigma, T):
    """The two Black-76 arguments.

    Returns (d1, d2). Both are +/-inf in the degenerate limits sigma->0 or T->0,
    which is correct: the option collapses to its intrinsic value there.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    T = np.asarray(T, dtype=float)

    vol_sqrt_t = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * np.square(vol_sqrt_t)) / vol_sqrt_t
        d2 = d1 - vol_sqrt_t
    return d1, d2


def call_price(F, K, sigma, T, df=1.0):
    """Undiscounted-forward Black-76 call, scaled by discount factor df."""
    d1, d2 = d1_d2(F, K, sigma, T)
    return df * (np.asarray(F) * norm_cdf(d1) - np.asarray(K) * norm_cdf(d2))


def put_price(F, K, sigma, T, df=1.0):
    """Black-76 put."""
    d1, d2 = d1_d2(F, K, sigma, T)
    return df * (np.asarray(K) * norm_cdf(-d2) - np.asarray(F) * norm_cdf(-d1))


def vega(F, K, sigma, T, df=1.0):
    """dPrice/dSigma. Identical for calls and puts.

    Strictly positive for T>0, sigma>0, but decays toward 0 for deep ITM/OTM
    options and as T->0. That collapse is the principal failure mode of a naive
    Newton solver -- see solver.py.
    """
    d1, _ = d1_d2(F, K, sigma, T)
    return df * np.asarray(F) * norm_pdf(d1) * np.sqrt(np.asarray(T))


def price(F, K, sigma, T, df=1.0, is_call=True):
    """Dispatch to call/put by boolean mask. `is_call` may be an array."""
    is_call = np.asarray(is_call)
    if is_call.ndim == 0:
        return call_price(F, K, sigma, T, df) if is_call else put_price(F, K, sigma, T, df)
    return np.where(
        is_call,
        call_price(F, K, sigma, T, df),
        put_price(F, K, sigma, T, df),
    )


def intrinsic(F, K, df=1.0, is_call=True):
    """Discounted intrinsic value -- the no-arbitrage *lower* bound on price."""
    F, K = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    return df * np.maximum(np.where(np.asarray(is_call), F - K, K - F), 0.0)


def upper_bound(F, K, df=1.0, is_call=True):
    """No-arbitrage *upper* bound: df*F for a call, df*K for a put."""
    F, K = np.asarray(F, dtype=float), np.asarray(K, dtype=float)
    return df * np.where(np.asarray(is_call), F, K)
