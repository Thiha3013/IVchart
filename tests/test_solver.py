"""Tests for the implied-vol solver.

Several of these encode failure modes the original implementation hit silently.
"""

import numpy as np
import pytest

from ivlib import bs, solver

T30 = bs.year_fraction(30)


# --------------------------------------------------------------------------
# Round trip -- the core correctness property
# --------------------------------------------------------------------------

@pytest.mark.parametrize("is_call", [True, False])
def test_round_trip_recovers_sigma(is_call):
    """price(sigma) -> implied_vol -> sigma, across the strike range."""
    F, df = 100.0, 0.995
    K = np.linspace(60, 160, 200)
    sig = np.full_like(K, 0.35)
    px = bs.price(F, K, sig, T30, df, is_call)
    out = solver.implied_vol(px, F, K, T30, df, is_call)
    good = np.isfinite(out)
    assert good.sum() > 150

    # Price residual is the honest correctness check. Sigma error is amplified
    # by 1/vega, so deep in the wings a perfectly-solved price still maps to a
    # loose sigma -- there, the price genuinely does not pin the vol down.
    resid = bs.price(F, K[good], out[good], T30, df, is_call) - px[good]
    np.testing.assert_allclose(resid, 0.0, atol=1e-7)

    # 1e-7 sits above the double-precision floor. For the deepest ITM strikes vega
    # is ~2e-5, so a price known to ~1e-15 relative pins sigma only to ~1e-9.
    np.testing.assert_allclose(out[good], sig[good], rtol=1e-7)


def test_round_trip_across_vol_and_maturity_grid():
    F, K, df = 100.0, 105.0, 1.0
    for T in (bs.year_fraction(d) for d in (7, 30, 90, 365)):
        for sig in (0.08, 0.25, 0.60, 1.20):
            px = bs.call_price(F, K, sig, T, df)
            got = solver.implied_vol(px, F, K, T, df, True)
            assert float(got) == pytest.approx(sig, rel=1e-6)


# --------------------------------------------------------------------------
# Failure modes the original code did not handle
# --------------------------------------------------------------------------

def test_infeasible_quotes_return_nan_not_garbage():
    """Below intrinsic and above the upper bound: no sigma exists."""
    F, K, df = 100.0, 90.0, 1.0
    below = solver.implied_vol(5.0, F, K, T30, df, True)     # intrinsic is 10
    above = solver.implied_vol(150.0, F, K, T30, df, True)   # cap is F = 100
    assert np.isnan(below) and np.isnan(above)


def test_zero_and_negative_prices_return_nan():
    assert np.isnan(solver.implied_vol(0.0, 100.0, 100.0, T30))
    assert np.isnan(solver.implied_vol(-1.0, 100.0, 100.0, T30))


def test_expired_options_return_nan():
    assert np.isnan(solver.implied_vol(5.0, 100.0, 100.0, 0.0))


def test_deep_wings_do_not_produce_nan_where_solvable():
    """Tiny vega is exactly where the old Newton loop blew up to NaN."""
    F, df = 100.0, 1.0
    K = np.array([40.0, 50.0, 200.0, 250.0])
    px = bs.call_price(F, K, 0.45, T30, df)
    out = solver.implied_vol(px, F, K, T30, df, True)
    feas = solver.feasible(px, F, K, T30, df, True)
    assert not np.isnan(out[feas]).any()


def test_no_nan_leaks_from_solver_on_random_feasible_input():
    rng = np.random.default_rng(0)
    n = 20_000
    F = rng.uniform(50, 500, n)
    K = F * rng.uniform(0.5, 1.8, n)
    T = bs.year_fraction(rng.integers(1, 400, n))
    sig = rng.uniform(0.05, 1.5, n)
    px = bs.call_price(F, K, sig, T, 1.0)
    out, info = solver.implied_vol(px, F, K, T, 1.0, True, return_info=True)
    feas = info["feasible"]
    assert not np.isnan(out[feas]).any()

    # Price is always solved to tight tolerance...
    resid = bs.call_price(F[feas], K[feas], out[feas], T[feas], 1.0) - px[feas]
    assert np.abs(resid).max() < 1e-6

    # ...but sigma accuracy is bounded by vega, and that is a property of the
    # problem rather than of the solver. Vega is dPrice/dSigma, so where vega is
    # ~0 the price simply carries no information about volatility: a deep ITM
    # option trades at intrinsic whatever the vol is. Measured on this grid,
    # worst-case relative sigma error by vega floor:
    #     vega > 1e-8 -> 1.8e-05      vega > 1e-4 -> 2.1e-09
    #     vega > 1e-6 -> 3.0e-07      vega > 1e-2 -> 4.1e-11
    v = bs.vega(F[feas], K[feas], sig[feas], T[feas], 1.0)
    informative = v > 1e-4
    assert informative.mean() > 0.95
    np.testing.assert_allclose(out[feas][informative], sig[feas][informative], rtol=1e-6)


def test_sigma_accuracy_is_bounded_by_vega_not_by_the_solver():
    """Pins the relationship above: tighter vega floor => tighter recovery."""
    rng = np.random.default_rng(0)
    n = 20_000
    F = rng.uniform(50, 500, n)
    K = F * rng.uniform(0.5, 1.8, n)
    T = bs.year_fraction(rng.integers(1, 400, n))
    sig = rng.uniform(0.05, 1.5, n)
    px = bs.call_price(F, K, sig, T, 1.0)
    out, info = solver.implied_vol(px, F, K, T, 1.0, True, return_info=True)

    f = info["feasible"]
    rel = np.abs(out[f] / sig[f] - 1.0)
    v = bs.vega(F[f], K[f], sig[f], T[f], 1.0)

    worst = [rel[v > thr].max() for thr in (1e-8, 1e-6, 1e-4, 1e-2)]
    assert worst == sorted(worst, reverse=True)  # monotone: more vega, less error
    assert worst[2] < 1e-8


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_coverage_reports_rejected_quotes():
    F, K, df = 100.0, 100.0, 1.0
    px = np.array([3.0, 0.0, 200.0, 5.0])  # two good, two impossible
    _, info = solver.implied_vol(px, F, K, T30, df, True, return_info=True)
    cov = solver.coverage(info)
    assert cov["total"] == 4
    assert cov["solved"] == 2
    assert cov["rejected_no_arb_pct"] == 50.0


def test_typical_option_converges_in_a_handful_of_iterations():
    """The original ran a fixed 100 iterations for every option regardless.

    Mean iterations is the number to watch: the wings are slow because the ATM
    seed is poor there, which is precisely what a better initial guess fixes.
    """
    K = np.linspace(80, 130, 500)
    px = bs.call_price(100.0, K, 0.3, T30, 1.0)
    _, info = solver.implied_vol(px, 100.0, K, T30, 1.0, True, return_info=True)
    assert info["mean_iterations"] < 12
    assert info["max_iterations"] <= 60


def test_every_element_actually_converges():
    K = np.linspace(70, 140, 400)
    px = bs.call_price(100.0, K, 0.42, T30, 1.0)
    out, info = solver.implied_vol(px, 100.0, K, T30, 1.0, True, return_info=True)
    assert info["converged"][info["feasible"]].all()
