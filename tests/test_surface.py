"""Tests for surface construction."""

import numpy as np
import pandas as pd
import pytest

from ivlib import bs, surface


def synth_chain(dates=("2022-01-03",), dtes=(14, 30, 90), n_strikes=21, spot=150.0):
    """A clean synthetic chain with a known smile: iv = base + 0.4 * k**2 - 0.3 * k."""
    rows = []
    for qd in dates:
        for dte in dtes:
            T = dte / 365.0
            F = spot * 1.001
            K = F * np.exp(np.linspace(-0.2, 0.2, n_strikes))
            k = np.log(K / F)
            iv = 0.25 + 0.4 * k ** 2 - 0.3 * k
            c = bs.call_price(F, K, iv, T, 1.0)
            p = bs.put_price(F, K, iv, T, 1.0)
            rows.append(pd.DataFrame({
                "QUOTE_DATE": qd, "EXPIRE_DATE": f"exp{dte}", "DTE": float(dte),
                "UNDERLYING_LAST": spot, "STRIKE": K,
                "C_BID": c * 0.999, "C_ASK": c * 1.001,
                "P_BID": p * 0.999, "P_ASK": p * 1.001,
            }))
    return pd.concat(rows, ignore_index=True)


def test_log_moneyness_is_zero_at_the_forward():
    assert surface.log_moneyness(100.0, 100.0) == pytest.approx(0.0)
    assert surface.log_moneyness(110.0, 100.0) > 0
    assert surface.log_moneyness(90.0, 100.0) < 0


def test_build_iv_table_recovers_the_input_smile():
    tab = surface.build_iv_table(synth_chain())
    assert len(tab) > 50
    k = tab["log_moneyness"].values
    expected = 0.25 + 0.4 * k ** 2 - 0.3 * k
    np.testing.assert_allclose(tab["iv"].values, expected, rtol=2e-3)


def test_build_iv_table_uses_the_otm_side():
    """Puts below the forward, calls above -- the market convention."""
    tab = surface.build_iv_table(synth_chain())
    below = tab[tab["log_moneyness"] < -0.02]
    above = tab[tab["log_moneyness"] > 0.02]
    np.testing.assert_allclose(below["iv"], below["iv_put"], rtol=1e-9)
    np.testing.assert_allclose(above["iv"], above["iv_call"], rtol=1e-9)


def test_smile_interpolates_onto_fixed_moneyness():
    tab = surface.build_iv_table(synth_chain())
    s = surface.smile(tab, at=(-0.1, 0.0, 0.1))
    assert len(s) == 3                       # three expiries, one date
    for col, k in (("k-0.10", -0.1), ("k+0.00", 0.0), ("k+0.10", 0.1)):
        vals = s[col].dropna()
        assert len(vals) >= 1
        np.testing.assert_allclose(vals, 0.25 + 0.4 * k ** 2 - 0.3 * k, rtol=3e-3)


def test_short_expiries_cover_a_narrower_strike_range():
    """Not every expiry reaches every moneyness, and that is correct.

    A 14-day call 10% out of the money is worth almost nothing, so its quote is
    dropped by the min-mid and spread filters. The smile is then NaN there rather
    than extrapolated -- the market did not price that strike, so neither do we.
    """
    tab = surface.build_iv_table(synth_chain())
    reach = tab.groupby("dte")["log_moneyness"].max()
    assert reach.loc[14] < reach.loc[90]


def test_smile_does_not_extrapolate():
    """Outside the quoted strikes the answer is NaN, not an invented vol."""
    tab = surface.build_iv_table(synth_chain())
    s = surface.smile(tab, at=(-5.0, 5.0))
    assert s[["k-5.00", "k+5.00"]].isna().all().all()


def test_skew_is_positive_for_a_downward_sloping_smile():
    tab = surface.build_iv_table(synth_chain())
    s = surface.skew(tab, wing=0.10)
    # iv(-0.1) - iv(+0.1) = -0.3*(-0.1) - (-0.3*0.1) = 0.06
    np.testing.assert_allclose(s["skew"], 0.06, rtol=5e-3)


def test_constant_maturity_interpolates_in_total_variance():
    """Between a 14d and a 90d expiry, the 30d point must sit in between."""
    tab = surface.build_iv_table(synth_chain())
    term = surface.atm_term_structure(tab)
    cm = surface.constant_maturity(term, days=30)
    assert len(cm) == 1
    atm = term.set_index("dte")["atm_iv"]
    assert atm.min() - 1e-6 <= cm["atm_iv_30d"].iloc[0] <= atm.max() + 1e-6


def test_constant_maturity_skips_days_that_do_not_bracket_the_target():
    tab = surface.build_iv_table(synth_chain(dtes=(60, 90)))
    cm = surface.constant_maturity(surface.atm_term_structure(tab), days=30)
    assert len(cm) == 0


def test_surface_grid_shape():
    tab = surface.build_iv_table(synth_chain())
    kg, dtes, Z = surface.surface_grid(tab, "2022-01-03", k_grid=np.linspace(-0.15, 0.15, 11))
    assert Z.shape == (len(dtes), len(kg))
    assert list(dtes) == sorted(dtes)
    # Finite across the core of the surface; the far wings of short expiries are
    # legitimately unquoted (see test_short_expiries_cover_a_narrower_strike_range).
    core = (np.abs(kg) <= 0.09)
    assert np.isfinite(Z[:, core]).all()
    assert np.isfinite(Z).mean() > 0.8


def test_surface_grid_on_a_missing_day_is_empty_not_an_error():
    tab = surface.build_iv_table(synth_chain())
    _, dtes, Z = surface.surface_grid(tab, "1999-01-01")
    assert len(dtes) == 0 and Z.size == 0
