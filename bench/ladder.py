"""Phase 3: the optimization ladder.

Five implementations of the same inversion, measured on the same real quotes and
checked against each other for agreement. The point is not only the final number
but *where each speedup comes from* -- memory layout, algorithm, or hardware.

  L0  scalar loop, scipy.stats.norm.cdf   the original code's shape
  L1  scalar loop, math.erf               same algorithm, cheaper normal CDF
  L2  vectorized NumPy, ATM seed          memory layout win
  L3  vectorized NumPy, Corrado-Miller    algorithmic win (fewer iterations)
  L4  Numba, fused + parallel             hardware win

L0 and L1 are timed on a subset because they are slow; throughput (options per
second) is the comparable figure across all rungs.
"""

from __future__ import annotations

import math
import time

import numpy as np

from ivlib import bs, fast, filter as qf, seed, solver
from bench.validate import build_forwards, load

SCALAR_N = 20_000       # subset size for the two scalar rungs
REPEATS = 5


# ---------------------------------------------------------------- rungs

def _scalar_newton(target, F, K, T, df, cdf, pdf, max_iter=60, tol=1e-12):
    """One option at a time, mirroring the original iterrows structure."""
    out = np.full(len(target), np.nan)
    for i in range(len(target)):
        t, f, k, tt, d = target[i], F[i], K[i], T[i], df[i]
        intr = max(f - k, 0.0) * d
        if not (t > intr + 1e-12) or not (t < d * f - 1e-12) or tt <= 0:
            continue
        s = 0.3
        lo, hi = 1e-6, 5.0
        for _ in range(max_iter):
            vst = s * math.sqrt(tt)
            d1 = (math.log(f / k) + 0.5 * vst * vst) / vst
            d2 = d1 - vst
            px = d * (f * cdf(d1) - k * cdf(d2))
            v = d * f * pdf(d1) * math.sqrt(tt)
            diff = px - t
            if diff > 0.0:
                hi = s
            else:
                lo = s
            step = s - diff / v if v > 1e-300 else 0.5 * (lo + hi)
            if not (lo < step < hi):
                step = 0.5 * (lo + hi)
            moved = abs(step - s)
            s = step
            if moved < tol or (hi - lo) < tol:
                break
        out[i] = s
    return out


def l0(a):
    from scipy.stats import norm
    return _scalar_newton(*a, cdf=lambda x: norm.cdf(x), pdf=lambda x: norm.pdf(x))


def l1(a):
    return _scalar_newton(
        *a,
        cdf=lambda x: 0.5 * (1.0 + math.erf(x * 0.7071067811865476)),
        pdf=lambda x: 0.3989422804014327 * math.exp(-0.5 * x * x),
    )


def l2(a):
    return solver.implied_vol(*a, is_call=True)


def l3(a):
    return solver.implied_vol(*a, is_call=True, seed_fn=seed.corrado_miller)


def l4(a):
    return fast.implied_vol(*a, is_call=True)


RUNGS = [
    ("L0  scalar + scipy.stats.norm", l0, True, "the original shape"),
    ("L1  scalar + math.erf", l1, True, "cheaper CDF"),
    ("L2  NumPy vectorized", l2, False, "memory layout"),
    ("L3  NumPy + Corrado-Miller seed", l3, False, "fewer iterations"),
    ("L4  Numba fused + parallel", l4, False, "registers, 8 cores"),
]


# ---------------------------------------------------------------- harness

def load_quotes():
    d0 = load("aapl_2021_2023.parquet")
    cm, _ = qf.filter_quotes(d0["C_BID"], d0["C_ASK"], dte=d0["DTE"])
    pm, _ = qf.filter_quotes(d0["P_BID"], d0["P_ASK"], dte=d0["DTE"])
    d = d0[qf.paired_mask(cm, pm)].copy()
    g, fit = build_forwards(d)
    ok = fit["ok"][g]
    return (
        qf.mid(d["C_BID"].values, d["C_ASK"].values)[ok],
        fit["forward"][g][ok],
        d["STRIKE"].values[ok],
        d["DTE"].values[ok] / 365.0,
        fit["discount"][g][ok],
    )


def bench(fn, args, repeats=REPEATS):
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        out = fn(args)
        best = min(best, time.perf_counter() - t)
    return out, best


def main():
    full = load_quotes()
    n_full = len(full[0])
    sub = tuple(a[:SCALAR_N] for a in full)
    print(f"dataset: {n_full:,} real call quotes  (scalar rungs timed on {SCALAR_N:,})\n")

    fast.implied_vol(*sub, is_call=True)  # warm the JIT before timing

    ref = solver.implied_vol(*full, is_call=True)
    results = []
    for name, fn, scalar, why in RUNGS:
        args = sub if scalar else full
        n = SCALAR_N if scalar else n_full
        out, secs = bench(fn, args, repeats=1 if scalar else REPEATS)
        rate = n / secs
        r = ref[:n]
        m = np.isfinite(out) & np.isfinite(r)
        agree = float(np.abs(out[m] - r[m]).max()) if m.any() else float("nan")
        results.append({"name": name, "why": why, "n": n, "secs": secs,
                        "rate": rate, "agree": agree})
        print(f"{name:34}{secs*1000:9.1f} ms  {rate/1e6:7.3f} M/s   "
              f"max|diff| {agree:.1e}")

    base = results[0]["rate"]
    print(f"\n{'rung':34}{'M/s':>9}{'vs L0':>10}{'vs prev':>10}   source")
    prev = None
    for r in results:
        vs_prev = f"{r['rate']/prev:6.2f}x" if prev else "     --"
        print(f"{r['name']:34}{r['rate']/1e6:9.3f}{r['rate']/base:9.0f}x{vs_prev:>10}   {r['why']}")
        prev = r["rate"]

    total = results[-1]["rate"] / base
    print(f"\ntotal speedup L0 -> L4: {total:,.0f}x")
    print(f"extrapolated L0 time for all {n_full:,} quotes: "
          f"{n_full / base / 60:.1f} min  ->  L4: {n_full / results[-1]['rate'] * 1000:.0f} ms")
    return results


if __name__ == "__main__":
    main()
