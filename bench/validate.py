"""Phase 2: validate the engine against the vendor's own implied vols.

aapl_2021_2023.csv ships C_IV and P_IV columns computed by the data vendor. They
are an independent answer key that the original project never used.

This script runs the full pipeline over every row and compares, under two time
conventions:

    fixed   T = calendar_days / 365      (correct)
    buggy   T = calendar_days / 252      (what the original code did)

The buggy run should come back low by sqrt(252/365) = 0.8309, which is the
prediction the whole day-count argument rests on.

Usage:  python bench/validate.py [parquet]
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from ivlib import bs, filter as qf, parity, solver


def load(path):
    df = pd.read_parquet(path)
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


def build_forwards(d):
    """Fit one forward and discount factor per (quote date, expiry)."""
    K = d["STRIKE"].values
    C = qf.mid(d["C_BID"].values, d["C_ASK"].values)
    P = qf.mid(d["P_BID"].values, d["P_ASK"].values)
    spot = d["UNDERLYING_LAST"].values
    g = parity.group_codes(d["QUOTE_DATE"].values, d["EXPIRE_DATE"].values)
    # Weight toward the money: near-ATM quotes are the tight, reliable ones.
    w = 1.0 / (1.0 + np.abs(K - spot) / spot * 10.0)
    return g, parity.implied_forward(K, C, P, group=g, weights=w)


def solve_side(price, F, K, T, df, is_call):
    t0 = time.perf_counter()
    iv, info = solver.implied_vol(price, F, K, T, df, is_call, return_info=True)
    return iv, info, time.perf_counter() - t0


def iv_uncertainty(bid, ask, vega):
    """How tightly the quoted market pins down implied volatility.

    Half the bid-ask spread, converted from price into vol units by dividing by
    vega. A quote whose spread spans 10 vol points cannot agree with anyone's
    implied vol to better than 10 vol points -- so this is the market's own
    resolution limit, not a defect of the solver.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        return 0.5 * (np.asarray(ask) - np.asarray(bid)) / np.maximum(vega, 1e-12)


def compare(iv, vendor, label):
    m = np.isfinite(iv) & np.isfinite(vendor) & (vendor > 0.01) & (vendor < 5.0)
    r = iv[m] / vendor[m]
    return {
        "label": label, "n": int(m.sum()),
        "median": float(np.median(r)),
        "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90)),
        "within_2pct": float(np.mean(np.abs(r - 1) < 0.02) * 100),
        "within_5pct": float(np.mean(np.abs(r - 1) < 0.05) * 100),
        "mask": m, "ratio": r,
    }


def main(path="aapl_2021_2023.parquet"):
    t0 = time.perf_counter()
    df = load(path)
    print(f"loaded {len(df):,} rows from parquet in {time.perf_counter()-t0:.2f}s\n")

    cm, crep = qf.filter_quotes(df["C_BID"], df["C_ASK"], dte=df["DTE"])
    pm, prep = qf.filter_quotes(df["P_BID"], df["P_ASK"], dte=df["DTE"])
    print(qf.format_report(crep, "call quotes"))
    print(qf.format_report(prep, "put quotes"))

    d = df[qf.paired_mask(cm, pm)].copy()
    print(f"\npaired strikes: {len(d):,}")

    t1 = time.perf_counter()
    g, fit = build_forwards(d)
    print(f"forwards: {fit['ok'].sum():,}/{len(fit['forward']):,} usable "
          f"({100*fit['ok'].mean():.1f}%)  median R2={np.nanmedian(fit['r2']):.7f}  "
          f"[{time.perf_counter()-t1:.3f}s]")

    ok = fit["ok"][g]
    F, DF = fit["forward"][g][ok], fit["discount"][g][ok]
    K = d["STRIKE"].values[ok]
    days = d["DTE"].values[ok]
    C = qf.mid(d["C_BID"].values, d["C_ASK"].values)[ok]
    vendor = d["C_IV"].values[ok]

    results = {}
    for label, T in (("fixed", days / 365.0), ("buggy", days / 252.0)):
        iv, info, secs = solve_side(C, F, K, T, DF, True)
        cov = solver.coverage(info)
        results[label] = compare(iv, vendor, label)
        results[label].update(iv=iv, secs=secs, cov=cov)
        print(f"\n[{label}] solved {cov['solved']:,}/{cov['total']:,} in {secs:.3f}s "
              f"({cov['total']/secs/1e6:.2f}M/s) mean iters {cov['mean_iterations']} "
              f"coverage {cov['coverage_pct']}%")

    print(f"\n{'':8}{'n':>10}{'median':>9}{'p10':>8}{'p90':>8}{'<2%':>8}{'<5%':>8}")
    for k in ("fixed", "buggy"):
        r = results[k]
        print(f"{k:8}{r['n']:>10,}{r['median']:>9.4f}{r['p10']:>8.4f}"
              f"{r['p90']:>8.4f}{r['within_2pct']:>7.1f}%{r['within_5pct']:>7.1f}%")

    obs = results["buggy"]["median"] / results["fixed"]["median"]
    print(f"\nday-count bias   predicted sqrt(252/365) = {np.sqrt(252/365):.4f}")
    print(f"                 observed                 = {obs:.4f}")

    # Where does the residual disagreement live? Almost entirely in quotes whose
    # own bid-ask spread does not determine a volatility to begin with.
    iv = results["fixed"]["iv"]
    m = results["fixed"]["mask"]
    T = days / 365.0
    veg = bs.vega(F[m], K[m], iv[m], T[m], DF[m])
    unc = iv_uncertainty(d["C_BID"].values[ok][m], d["C_ASK"].values[ok][m], veg)
    r = results["fixed"]["ratio"]

    buckets = [(0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, np.inf)]
    rows = []
    print(f"\nagreement vs the market's own resolution (half-spread / vega):")
    print(f"{'IV uncertainty':>18}{'n':>10}{'median':>9}{'<2%':>8}")
    for lo, hi in buckets:
        s = (unc >= lo) & (unc < hi)
        if s.sum():
            row = (lo, hi, int(s.sum()), float(np.median(r[s])),
                   float(np.mean(np.abs(r[s] - 1) < 0.02) * 100))
            rows.append(row)
            hi_s = "inf" if hi == np.inf else f"{hi:.3f}"
            print(f"{f'±{lo:.3f}-{hi_s}':>18}{row[2]:>10,}{row[3]:>9.4f}{row[4]:>7.1f}%")

    good = unc < 0.02
    print(f"\nrestricted to quotes the market pins to ±2 vol points "
          f"({good.sum():,}, {100*good.mean():.1f}%):")
    print(f"  median {np.median(r[good]):.4f}   "
          f"within 2% {100*np.mean(np.abs(r[good]-1)<0.02):.1f}%   "
          f"within 5% {100*np.mean(np.abs(r[good]-1)<0.05):.1f}%")

    results["quality"] = {"buckets": rows, "unc": unc, "good": good}
    return results, vendor


if __name__ == "__main__":
    main(*sys.argv[1:])
