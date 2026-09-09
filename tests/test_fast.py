"""The Numba kernel must be a drop-in for the NumPy solver.

A faster implementation that disagrees is not a faster implementation, so these
tests are about equivalence, not speed.
"""

import numpy as np
import pytest

from ivlib import bs, fast, seed, solver

T30 = bs.year_fraction(30)


@pytest.mark.parametrize("is_call", [True, False])
def test_matches_numpy_solver_across_strikes(is_call):
    F, df = 100.0, 0.995
    K = np.linspace(55, 165, 500)
    px = bs.price(F, K, 0.33, T30, df, is_call)
    a = fast.implied_vol(px, F, K, T30, df, is_call)
    b = solver.implied_vol(px, F, K, T30, df, is_call, seed_fn=seed.corrado_miller)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    m = np.isfinite(a)
    # 1e-6 is far tighter than the vega-limited resolution of the problem
    # (~1e-9 relative at best, and much looser in the wings). The two solvers
    # take different step sequences, so they land on different sides of the
    # same root at that scale; requiring bit-equality would be requiring noise.
    np.testing.assert_allclose(a[m], b[m], atol=1e-6)


def test_matches_numpy_on_a_random_grid():
    rng = np.random.default_rng(11)
    n = 30_000
    F = rng.uniform(50, 500, n)
    K = F * rng.uniform(0.6, 1.6, n)
    T = bs.year_fraction(rng.integers(5, 400, n))
    df = rng.uniform(0.97, 1.0, n)
    sig = rng.uniform(0.05, 1.5, n)
    px = bs.call_price(F, K, sig, T, df)

    a = fast.implied_vol(px, F, K, T, df, True)
    b = solver.implied_vol(px, F, K, T, df, True, seed_fn=seed.corrado_miller)
    assert np.array_equal(np.isfinite(a), np.isfinite(b))
    m = np.isfinite(a)

    # Both solvers must price back to the quote they were given. This is the
    # assertion that actually means "same answer" -- it holds everywhere.
    for out in (a, b):
        resid = bs.call_price(F[m], K[m], out[m], T[m], df[m]) - px[m]
        assert np.abs(resid).max() < 1e-7

    # They agree on sigma wherever sigma is determined. Where vega is tiny the
    # two take different step sequences to the same flat region of the price
    # curve and land microscopically apart; that is the problem's conditioning,
    # not a discrepancy between implementations.
    v = bs.vega(F[m], K[m], sig[m], T[m], df[m])
    good = v > 1e-2
    np.testing.assert_allclose(a[m][good], b[m][good], rtol=1e-7)
    np.testing.assert_allclose(a[m][good], sig[m][good], rtol=1e-5)


def test_rejects_the_same_infeasible_quotes():
    F, K, df = 100.0, 90.0, 1.0
    bad = np.array([5.0, 150.0, 0.0, -1.0])   # below intrinsic, above cap, zero, negative
    a = fast.implied_vol(bad, F, K, T30, df, True)
    b = solver.implied_vol(bad, F, K, T30, df, True)
    assert np.isnan(a).all()
    assert np.array_equal(np.isnan(a), np.isnan(b))


def test_reports_iteration_counts():
    K = np.linspace(80, 130, 400)
    px = bs.call_price(100.0, K, 0.3, T30, 1.0)
    _, info = fast.implied_vol(px, 100.0, K, T30, 1.0, True, return_info=True)
    assert 0 < info["mean_iterations"] < 15
    assert info["max_iterations"] <= 60


def test_scalar_broadcast_arguments():
    """F, T, df given as scalars must broadcast against an array of strikes."""
    K = np.array([90.0, 100.0, 110.0])
    px = bs.call_price(100.0, K, 0.3, T30, 1.0)
    out = fast.implied_vol(px, 100.0, K, T30, 1.0, True)
    np.testing.assert_allclose(out, 0.3, rtol=1e-8)
