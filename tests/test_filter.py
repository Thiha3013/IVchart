"""Tests for quote quality filtering."""

import numpy as np
import pytest

from ivlib import filter as qf


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
