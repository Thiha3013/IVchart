"""Black-76 pricing, day count, initial guesses."""

import numpy as np
import pytest

from ivlib import pricing as bs, solver

T30 = bs.year_fraction(30)


# --------------------------------------------------------------------------
# Day count -- the original bug
# --------------------------------------------------------------------------

def test_year_fraction_is_calendar_over_365():
    assert bs.year_fraction(365) == pytest.approx(1.0)
    assert bs.year_fraction(30) == pytest.approx(0.0821917, abs=1e-6)


def test_the_two_valid_conventions_agree_to_within_2_percent():
    """30 calendar/365 vs 21 trading/252 -- both self-consistent, ~1.4% apart."""
    calendar = 30 / 365
    trading = 21 / 252
    assert abs(trading / calendar - 1.0) < 0.02


def test_mixed_convention_biases_implied_vol_low_by_0_8309():
    """Reproduces the original bug and pins its magnitude.

    Price an option at the correct T, then re-imply it at the buggy T=30/252.
    The recovered vol should come back low by sqrt(365/252) -- and the bias is
    strike-independent, because it is a pure rescaling of total variance.
    """
    from ivlib.solver import implied_vol

    true_sigma, F, df = 0.30, 100.0, 1.0
    for K in (90.0, 100.0, 110.0):
        px = bs.call_price(F, K, true_sigma, 30 / 365, df)
        recovered = implied_vol(px, F, K, 30 / 252, df, is_call=True)
        assert float(recovered) / true_sigma == pytest.approx(0.8309, abs=1e-3)


# --------------------------------------------------------------------------
# Model identities
# --------------------------------------------------------------------------

def test_put_call_parity_holds():
    """C - P = df*(F - K). The defining identity of forward-space pricing."""
    F, K, sigma, df = 100.0, np.array([80.0, 100.0, 120.0]), 0.25, 0.99
    c = bs.call_price(F, K, sigma, T30, df)
    p = bs.put_price(F, K, sigma, T30, df)
    np.testing.assert_allclose(c - p, df * (F - K), rtol=1e-12)


def test_atm_forward_call_equals_put():
    """A corollary of parity: at F == K the two are identical."""
    c = bs.call_price(100.0, 100.0, 0.3, T30)
    p = bs.put_price(100.0, 100.0, 0.3, T30)
    assert c == pytest.approx(p)


def test_price_is_strictly_increasing_in_sigma():
    """Why the implied-vol root is unique when it exists."""
    sigmas = np.linspace(0.01, 3.0, 300)
    px = bs.call_price(100.0, 110.0, sigmas, T30)
    assert np.all(np.diff(px) > 0)


def test_vega_is_positive_but_collapses_in_the_wings():
    """Vega > 0 always, yet small enough far OTM to break naive Newton."""
    atm = bs.vega(100.0, 100.0, 0.3, T30)
    wing = bs.vega(100.0, 400.0, 0.3, T30)
    assert atm > 0 and wing > 0
    assert wing < atm * 1e-6


def test_prices_respect_no_arbitrage_bounds():
    K = np.array([70.0, 100.0, 130.0])
    for is_call in (True, False):
        px = bs.price(100.0, K, 0.3, T30, 0.99, is_call)
        lo = bs.intrinsic(100.0, K, 0.99, is_call)
        hi = bs.upper_bound(100.0, K, 0.99, is_call)
        assert np.all(px >= lo - 1e-12)
        assert np.all(px <= hi + 1e-12)


def test_vectorizes_over_arrays():
    K = np.linspace(50, 150, 1000)
    px = bs.call_price(100.0, K, 0.3, T30)
    assert px.shape == (1000,)
    assert np.all(np.diff(px) < 0)  # calls cheapen as strike rises


def test_scipy_stats_norm_is_not_used_in_the_hot_path():
    """The normal CDF is scipy.special.ndtr, not scipy.stats.norm.cdf.

    These compute the same function, but scipy.stats.norm.cdf wraps ndtr in
    distribution-object machinery. In a scalar loop that wrapper dominates:
    swapping it for a direct erf call alone is a ~58x speedup on the benchmark
    ladder, larger than the win from vectorizing. Guarding against a well-meaning
    'use the standard scipy API' edit.
    """
    import inspect

    src = inspect.getsource(bs)
    # Check the imports, not the prose -- the module comment names
    # scipy.stats.norm.cdf precisely in order to explain why it is avoided.
    assert "from scipy.stats" not in src
    assert "import scipy.stats" not in src
    assert "from scipy.special import ndtr" in src
