"""Quote filtering and implied forward."""

import numpy as np
import pytest

from ivlib import market as qf, market as parity, pricing as bs, solver

def test_mid_is_the_midpoint():
    np.testing.assert_allclose(qf.mid([1.0, 2.0], [1.2, 3.0]), [1.1, 2.5])


def test_spread_pct():
    # (1.10 - 1.00) / 1.05 = 0.095238...
    np.testing.assert_allclose(qf.spread_pct([1.0], [1.1]), [0.0952381], rtol=1e-6)
    np.testing.assert_allclose(qf.spread_pct([1.0], [3.0]), [1.0], rtol=1e-12)
    assert qf.spread_pct([0.0], [0.0])[0] == np.inf


# --------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------

def test_rejects_zero_bid():
    """The 61-of-114 case from real data: listed strikes nobody bids on."""
    mask, rep = qf.filter_quotes([0.0, 1.0], [0.5, 1.05])
    assert list(mask) == [False, True]
    assert rep["rejected_by"]["zero_bid"] == 1


def test_rejects_crossed_market():
    mask, rep = qf.filter_quotes([2.0], [1.5])
    assert not mask[0]
    assert rep["rejected_by"]["crossed"] == 1


def test_rejects_wide_spread():
    mask, _ = qf.filter_quotes([1.0, 1.0], [1.05, 9.0])
    assert list(mask) == [True, False]


def test_spread_threshold_is_configurable():
    wide_bid, wide_ask = [1.0], [1.5]           # 40% spread
    assert not qf.filter_quotes(wide_bid, wide_ask, max_spread_pct=0.20)[0][0]
    assert qf.filter_quotes(wide_bid, wide_ask, max_spread_pct=0.50)[0][0]


def test_rejects_sub_penny_mid():
    mask, _ = qf.filter_quotes([0.001], [0.004])
    assert not mask[0]


def test_dte_range_is_applied_only_when_supplied():
    assert qf.filter_quotes([1.0], [1.05])[0][0]                      # no dte given
    assert not qf.filter_quotes([1.0], [1.05], dte=[1])[0][0]         # too short
    assert not qf.filter_quotes([1.0], [1.05], dte=[900])[0][0]       # too long
    assert qf.filter_quotes([1.0], [1.05], dte=[30])[0][0]


def test_stale_check_is_opt_in():
    kw = dict(last=[5.0], max_stale_pct=0.10)
    assert not qf.filter_quotes([1.0], [1.05], **kw)[0][0]   # last far from mid
    assert qf.filter_quotes([1.0], [1.05], last=[5.0])[0][0]  # check disabled


def test_nan_quotes_rejected():
    mask, _ = qf.filter_quotes([np.nan, 1.0], [1.0, 1.05])
    assert list(mask) == [False, True]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def test_waterfall_counts_sum_to_total_rejected():
    """Each quote is attributed to its FIRST failure, so counts do not overlap."""
    rng = np.random.default_rng(3)
    n = 5000
    bid = rng.uniform(-0.1, 5, n)
    ask = bid + rng.uniform(-0.2, 2, n)
    mask, rep = qf.filter_quotes(bid, ask, dte=rng.integers(0, 500, n))
    assert rep["kept"] == int(mask.sum())
    assert sum(rep["rejected_by"].values()) == rep["total"] - rep["kept"]


def test_report_percentages_are_consistent():
    mask, rep = qf.filter_quotes([1.0, 0.0, 1.0], [1.05, 0.5, 1.05])
    assert rep["kept"] == 2
    assert rep["kept_pct"] == pytest.approx(66.67, abs=0.01)


def test_format_report_omits_empty_reasons():
    _, rep = qf.filter_quotes([1.0], [1.05])
    text = qf.format_report(rep)
    assert "1/1 kept" in text
    assert "zero_bid" not in text


# --------------------------------------------------------------------------
# Pairing
# --------------------------------------------------------------------------

def test_paired_mask_requires_both_legs():
    """Parity needs C and P at the SAME strike; a good call with a dead put
    cannot contribute to the forward fit."""
    calls = np.array([True, True, False])
    puts = np.array([True, False, True])
    np.testing.assert_array_equal(qf.paired_mask(calls, puts), [True, False, False])


# ---------------------------------------------------------------- parity

def synth(F, df, K):
    """Arbitrage-consistent (C, P) pairs satisfying C - P = df*(F - K)."""
    y = df * (F - K)
    C = np.maximum(y, 0.0) + 5.0
    return C, C - y


# --------------------------------------------------------------------------
# Recovery
# --------------------------------------------------------------------------

def test_recovers_forward_and_discount_exactly():
    F, df = 137.42, 0.9963
    K = np.arange(120.0, 156.0, 2.5)
    out = parity.implied_forward(K, *synth(F, df, K))
    assert out["forward"][0] == pytest.approx(F, rel=1e-12)
    assert out["discount"][0] == pytest.approx(df, rel=1e-12)
    assert out["ok"][0]


def test_forward_is_not_the_intercept():
    """Guards the easy error: intercept = df*F, so F needs the division by df.

    With a small df the two differ materially; asserting they differ keeps a
    'simplification' from silently reintroducing the bug.
    """
    F, df = 200.0, 0.80
    K = np.linspace(180, 220, 9)
    out = parity.implied_forward(K, *synth(F, df, K))
    assert out["forward"][0] == pytest.approx(F, rel=1e-10)
    assert out["forward"][0] != pytest.approx(df * F, rel=1e-3)


def test_implied_rate_round_trips():
    T = bs.year_fraction(45)
    r_true = 0.0525
    df = np.exp(-r_true * T)
    K = np.linspace(90, 110, 9)
    out = parity.implied_forward(K, *synth(100.0, df, K))
    assert parity.implied_rate(out["discount"], T)[0] == pytest.approx(r_true, rel=1e-9)


def test_handles_many_expiries_at_once():
    Fs = np.array([100.0, 101.5, 103.0, 98.0])
    dfs = np.array([0.999, 0.996, 0.991, 0.985])
    K, C, P, g = [], [], [], []
    for i, (F, d) in enumerate(zip(Fs, dfs)):
        k = np.linspace(F - 15, F + 15, 11)
        c, p = synth(F, d, k)
        K.append(k); C.append(c); P.append(p); g.append(np.full(11, i))
    out = parity.implied_forward(*(np.concatenate(a) for a in (K, C, P, g)))
    np.testing.assert_allclose(out["forward"], Fs, rtol=1e-10)
    np.testing.assert_allclose(out["discount"], dfs, rtol=1e-10)
    assert out["ok"].all()


# --------------------------------------------------------------------------
# Robustness
# --------------------------------------------------------------------------

def test_survives_realistic_quote_noise():
    F, df = 412.30, 0.9948
    K = np.linspace(370, 455, 25)
    C, P = synth(F, df, K)
    rng = np.random.default_rng(7)
    C = C + rng.normal(0, 0.01, C.shape)   # ~1 cent of quote noise
    P = P + rng.normal(0, 0.01, P.shape)
    out = parity.implied_forward(K, C, P)
    assert out["forward"][0] == pytest.approx(F, rel=1e-3)
    assert out["discount"][0] == pytest.approx(df, rel=1e-3)


def test_trimming_rejects_a_stale_wing_quote():
    """One bad far-strike print should not tilt the line."""
    F, df = 100.0, 0.995
    K = np.linspace(70, 130, 25)
    C, P = synth(F, df, K)
    C[0] += 12.0                            # stale deep-ITM call
    trimmed = parity.implied_forward(K, C, P, trim=True)["forward"][0]
    raw = parity.implied_forward(K, C, P, trim=False)["forward"][0]
    assert abs(trimmed - F) < abs(raw - F)
    assert trimmed == pytest.approx(F, rel=1e-2)


def test_too_few_strikes_is_flagged_not_guessed():
    K = np.array([100.0, 105.0])
    out = parity.implied_forward(K, *synth(100.0, 0.99, K))
    assert not out["ok"][0]


def test_implausible_discount_is_flagged():
    K = np.linspace(90, 110, 9)
    out = parity.implied_forward(K, *synth(100.0, 0.30, K))  # df far too low
    assert not out["ok"][0]


# --------------------------------------------------------------------------
# End to end -- the actual point of this module
# --------------------------------------------------------------------------

def test_quotes_to_implied_vol_with_no_external_rate():
    """Full chain: option quotes -> forward -> implied vol.

    No interest rate and no dividend yield is supplied anywhere. Both are
    recovered from the quotes, which is what makes DGS10.csv unnecessary.
    """
    F_true, df_true, sigma_true = 155.75, 0.9942, 0.2875
    T = bs.year_fraction(31)
    K = np.linspace(130, 185, 23)

    C = bs.call_price(F_true, K, sigma_true, T, df_true)
    P = bs.put_price(F_true, K, sigma_true, T, df_true)

    fit = parity.implied_forward(K, C, P)
    assert fit["ok"][0]
    F, d = fit["forward"][0], fit["discount"][0]
    assert F == pytest.approx(F_true, rel=1e-9)

    iv = solver.implied_vol(C, F, K, T, d, is_call=True)
    np.testing.assert_allclose(iv, sigma_true, rtol=1e-6)


def test_group_codes_builds_one_group_per_expiry():
    dates = np.array(["2021-01-04"] * 6)
    exps = np.array(["2021-02-05"] * 3 + ["2021-03-05"] * 3)
    g = parity.group_codes(dates, exps)
    assert len(np.unique(g)) == 2
    assert g.shape == (6,)


def test_group_codes_handles_object_dtype():
    """pandas hands back object dtype for string columns -- the shape real data
    arrives in, and the case that a stack-then-unique implementation fails on."""
    dates = np.array(["2021-01-04"] * 4, dtype=object)
    exps = np.array(["2021-02-05", "2021-02-05", "2021-03-05", "2021-03-05"], dtype=object)
    g = parity.group_codes(dates, exps)
    assert len(np.unique(g)) == 2
    np.testing.assert_array_equal(g, [0, 0, 1, 1])


def test_group_codes_distinguishes_same_expiry_on_different_dates():
    d = np.array(["2021-01-04", "2021-01-05"], dtype=object)
    e = np.array(["2021-02-05", "2021-02-05"], dtype=object)
    assert len(np.unique(parity.group_codes(d, e))) == 2


def test_group_codes_single_key():
    g = parity.group_codes(np.array(["a", "b", "a"], dtype=object))
    np.testing.assert_array_equal(g, [0, 1, 0])
