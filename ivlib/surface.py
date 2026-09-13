"""Build implied volatility surfaces from an option chain.

The original project reduced each trading day to a single number: the implied
vol of the one near-ATM, near-30-day option. That is 467 points from 548,163
quotes. Everything else was discarded because a scalar Python loop could not
afford it.

With the solver at ~14M inversions/sec the whole chain is affordable, so the
natural object is no longer a time series of one number but a *surface*: implied
volatility as a function of strike and maturity, for every day.

Coordinates
-----------
Strike is expressed as log-moneyness

    k = ln(K / F)

rather than as a raw strike or a percentage. k = 0 is at-the-money-forward, it is
symmetric in the sense that a call at +k and a put at -k are mirror positions, and
it is comparable across days as the underlying moves. Raw strikes are not: AAPL's
$130 strike means something different in 2021 than in 2022.

OTM convention
--------------
At each strike both a call and a put trade, and in theory they imply the same
volatility. In practice the out-of-the-money one is more liquid and tighter, so
the market convention -- followed here -- is to take the put's implied vol below
the forward and the call's above it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivlib import bs, fast, filter as qf, parity, solver


def log_moneyness(K, F):
    """k = ln(K / F). Zero at the money forward."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float))


def build_iv_table(df, use_fast=True, min_strikes=5):
    """Run the full pipeline and return one tidy row per usable strike.

    Columns: quote_date, expire_date, dte, T, strike, forward, discount,
    log_moneyness, iv (OTM convention), iv_call, iv_put, group.
    """
    cmask, _ = qf.filter_quotes(df["C_BID"], df["C_ASK"], dte=df["DTE"])
    pmask, _ = qf.filter_quotes(df["P_BID"], df["P_ASK"], dte=df["DTE"])
    d = df[qf.paired_mask(cmask, pmask)].copy()

    K = d["STRIKE"].values
    C = qf.mid(d["C_BID"].values, d["C_ASK"].values)
    P = qf.mid(d["P_BID"].values, d["P_ASK"].values)
    spot = d["UNDERLYING_LAST"].values
    g = parity.group_codes(d["QUOTE_DATE"].values, d["EXPIRE_DATE"].values)
    w = 1.0 / (1.0 + np.abs(K - spot) / spot * 10.0)
    fit = parity.implied_forward(K, C, P, group=g, weights=w)

    ok = fit["ok"][g]
    d, K, C, P, g = d[ok], K[ok], C[ok], P[ok], g[ok]
    F, DF = fit["forward"][g], fit["discount"][g]
    T = d["DTE"].values / 365.0

    engine = fast.implied_vol if use_fast else solver.implied_vol
    iv_c = engine(C, F, K, T, DF, True)
    iv_p = engine(P, F, K, T, DF, False)

    k = log_moneyness(K, F)
    iv = np.where(k >= 0.0, iv_c, iv_p)          # OTM side
    iv = np.where(np.isfinite(iv), iv, np.where(k >= 0.0, iv_p, iv_c))

    out = pd.DataFrame({
        "quote_date": d["QUOTE_DATE"].values,
        "expire_date": d["EXPIRE_DATE"].values,
        "dte": d["DTE"].values,
        "T": T, "strike": K, "forward": F, "discount": DF,
        "log_moneyness": k, "iv": iv, "iv_call": iv_c, "iv_put": iv_p,
        "group": g,
    })
    out = out[np.isfinite(out["iv"]) & (out["iv"] > 0.01) & (out["iv"] < 4.0)]

    counts = out.groupby("group")["iv"].transform("size")
    return out[counts >= min_strikes].reset_index(drop=True)


def _interp_group(k, iv, targets):
    """Linear interpolation in log-moneyness, no extrapolation.

    Returns NaN outside the strikes actually quoted, rather than inventing a
    volatility for a strike the market never priced.
    """
    order = np.argsort(k)
    k, iv = k[order], iv[order]
    out = np.interp(targets, k, iv, left=np.nan, right=np.nan)
    return out


def smile(table, at=(-0.10, -0.05, 0.0, 0.05, 0.10)):
    """Implied vol at fixed log-moneyness points, per (date, expiry).

    Interpolating onto a fixed grid is what makes surfaces comparable across
    days: the listed strikes move with the underlying, these coordinates do not.
    """
    at = np.asarray(at, dtype=float)
    rows = []
    for (qd, ed), grp in table.groupby(["quote_date", "expire_date"], sort=False):
        vals = _interp_group(grp["log_moneyness"].values, grp["iv"].values, at)
        rows.append((qd, ed, grp["dte"].iloc[0], *vals))
    cols = ["quote_date", "expire_date", "dte"] + [f"k{v:+.2f}" for v in at]
    return pd.DataFrame(rows, columns=cols)


def atm_term_structure(table):
    """ATM-forward implied vol for every (date, expiry) -- the term structure."""
    s = smile(table, at=(0.0,)).rename(columns={"k+0.00": "atm_iv"})
    return s.dropna(subset=["atm_iv"]).reset_index(drop=True)


def skew(table, wing=0.10):
    """Put-minus-call wing spread at +/- `wing` in log-moneyness.

    Positive values mean downside strikes carry higher implied vol than upside
    ones -- the equity index skew, and a direct measure of what crash protection
    costs relative to upside.
    """
    s = smile(table, at=(-wing, 0.0, wing))
    lo, hi = f"k{-wing:+.2f}", f"k{wing:+.2f}"
    s["skew"] = s[lo] - s[hi]
    s["atm_iv"] = s["k+0.00"]
    return s.dropna(subset=["skew"]).reset_index(drop=True)


def constant_maturity(term, days=30):
    """Interpolate ATM vol to a fixed maturity, VIX-style.

    Listed expiries do not sit at a constant horizon -- a 30-day option today is
    a 23-day option a week later -- so comparing raw quotes across time conflates
    a change in volatility with a change in maturity. Interpolating in *total
    variance* (sigma^2 * T) rather than in volatility is the standard fix and is
    what keeps the series free of sawtooth artifacts as expiries roll.
    """
    rows = []
    for qd, grp in term.groupby("quote_date", sort=True):
        g = grp.dropna(subset=["atm_iv"]).sort_values("dte")
        d, v = g["dte"].values.astype(float), g["atm_iv"].values
        if len(d) < 2 or d.min() > days or d.max() < days:
            continue
        var = v * v * (d / 365.0)
        target_var = np.interp(days, d, var)
        rows.append((qd, float(np.sqrt(target_var / (days / 365.0)))))
    return pd.DataFrame(rows, columns=["quote_date", f"atm_iv_{days}d"])


def surface_grid(table, quote_date, k_grid=None, dte_grid=None):
    """A (log-moneyness x DTE) grid of implied vols for a single day."""
    k_grid = np.linspace(-0.25, 0.25, 41) if k_grid is None else np.asarray(k_grid)
    day = table[table["quote_date"] == quote_date]
    if day.empty:
        return k_grid, np.array([]), np.empty((0, len(k_grid)))

    expiries = day.groupby("expire_date")["dte"].first().sort_values()
    if dte_grid is not None:
        expiries = expiries[expiries.isin(dte_grid)]

    rows = []
    for ed in expiries.index:
        grp = day[day["expire_date"] == ed]
        rows.append(_interp_group(grp["log_moneyness"].values, grp["iv"].values, k_grid))
    return k_grid, expiries.values.astype(float), np.array(rows)
