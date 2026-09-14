"""Implied vol surfaces: smiles, skew, term structure, constant-maturity series.

Strike axis is log-moneyness k = ln(K/F): comparable across days as the underlying
moves. OTM convention: put IV below the forward, call IV above (the liquid side).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ivlib import market as mk, solver


def log_moneyness(K, F):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(K, dtype=float) / np.asarray(F, dtype=float))


def build_iv_table(df, use_fast=True, min_strikes=5):
    """Full pipeline: chain -> one row per usable strike with iv (OTM side), iv_call, iv_put."""
    cmask, _ = mk.filter_quotes(df["C_BID"], df["C_ASK"], dte=df["DTE"])
    pmask, _ = mk.filter_quotes(df["P_BID"], df["P_ASK"], dte=df["DTE"])
    d = df[mk.paired_mask(cmask, pmask)].copy()

    K = d["STRIKE"].values
    C = mk.mid(d["C_BID"].values, d["C_ASK"].values)
    P = mk.mid(d["P_BID"].values, d["P_ASK"].values)
    spot = d["UNDERLYING_LAST"].values
    g = mk.group_codes(d["QUOTE_DATE"].values, d["EXPIRE_DATE"].values)
    w = 1.0 / (1.0 + np.abs(K - spot) / spot * 10.0)   # weight toward the money
    fit = mk.implied_forward(K, C, P, group=g, weights=w)

    ok = fit["ok"][g]
    d, K, C, P, g = d[ok], K[ok], C[ok], P[ok], g[ok]
    F, DF = fit["forward"][g], fit["discount"][g]
    T = d["DTE"].values / 365.0

    engine = solver.implied_vol_fast if use_fast else solver.implied_vol
    iv_c, iv_p = engine(C, F, K, T, DF, True), engine(P, F, K, T, DF, False)

    k = log_moneyness(K, F)
    iv = np.where(k >= 0.0, iv_c, iv_p)
    iv = np.where(np.isfinite(iv), iv, np.where(k >= 0.0, iv_p, iv_c))

    out = pd.DataFrame({
        "quote_date": d["QUOTE_DATE"].values, "expire_date": d["EXPIRE_DATE"].values,
        "dte": d["DTE"].values, "T": T, "strike": K, "forward": F, "discount": DF,
        "log_moneyness": k, "iv": iv, "iv_call": iv_c, "iv_put": iv_p, "group": g,
    })
    out = out[np.isfinite(out["iv"]) & (out["iv"] > 0.01) & (out["iv"] < 4.0)]
    counts = out.groupby("group")["iv"].transform("size")
    return out[counts >= min_strikes].reset_index(drop=True)


def _interp_group(k, iv, targets):
    """Linear in k, NaN outside the quoted range -- never extrapolate a vol the market didn't price."""
    order = np.argsort(k)
    return np.interp(targets, k[order], iv[order], left=np.nan, right=np.nan)


def smile(table, at=(-0.10, -0.05, 0.0, 0.05, 0.10)):
    """IV at fixed log-moneyness per (date, expiry)."""
    at = np.asarray(at, dtype=float)
    rows = [(qd, ed, grp["dte"].iloc[0], *_interp_group(grp["log_moneyness"].values, grp["iv"].values, at))
            for (qd, ed), grp in table.groupby(["quote_date", "expire_date"], sort=False)]
    cols = ["quote_date", "expire_date", "dte"] + [f"k{v:+.2f}" for v in at]
    return pd.DataFrame(rows, columns=cols)


def atm_term_structure(table):
    s = smile(table, at=(0.0,)).rename(columns={"k+0.00": "atm_iv"})
    return s.dropna(subset=["atm_iv"]).reset_index(drop=True)


def skew(table, wing=0.10):
    """IV(-wing) - IV(+wing). Positive = downside strikes richer."""
    s = smile(table, at=(-wing, 0.0, wing))
    s["skew"] = s[f"k{-wing:+.2f}"] - s[f"k{wing:+.2f}"]
    s["atm_iv"] = s["k+0.00"]
    return s.dropna(subset=["skew"]).reset_index(drop=True)


def constant_maturity(term, days=30):
    """VIX-style fixed-horizon ATM vol, interpolated in total variance (no roll sawtooth)."""
    rows = []
    for qd, grp in term.groupby("quote_date", sort=True):
        g = grp.dropna(subset=["atm_iv"]).sort_values("dte")
        d, v = g["dte"].values.astype(float), g["atm_iv"].values
        if len(d) < 2 or d.min() > days or d.max() < days:
            continue
        var = np.interp(days, d, v * v * (d / 365.0))
        rows.append((qd, float(np.sqrt(var / (days / 365.0)))))
    return pd.DataFrame(rows, columns=["quote_date", f"atm_iv_{days}d"])


def surface_grid(table, quote_date, k_grid=None, dte_grid=None):
    """(log-moneyness x DTE) grid for one day. Returns (k_grid, dtes, Z)."""
    k_grid = np.linspace(-0.25, 0.25, 41) if k_grid is None else np.asarray(k_grid)
    day = table[table["quote_date"] == quote_date]
    if day.empty:
        return k_grid, np.array([]), np.empty((0, len(k_grid)))
    expiries = day.groupby("expire_date")["dte"].first().sort_values()
    if dte_grid is not None:
        expiries = expiries[expiries.isin(dte_grid)]
    rows = [_interp_group(day[day["expire_date"] == ed]["log_moneyness"].values,
                          day[day["expire_date"] == ed]["iv"].values, k_grid) for ed in expiries.index]
    return k_grid, expiries.values.astype(float), np.array(rows)
