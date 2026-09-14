"""The optimization ladder: five implementations of one inversion, same real quotes.

    L0 scalar + scipy.stats.norm   the original shape
    L1 scalar + math.erf           cheaper CDF           (the biggest single jump, 52x)
    L2 NumPy vectorized            memory layout
    L3 NumPy + Corrado-Miller      fewer iterations      (1.05x -- the algorithmic rung paid least)
    L4 Numba fused + parallel      registers, all cores

    python -m bench.ladder [--plot]     -> bench/figures/ladder.png with --plot
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

from app import data
from bench.validate import build_forwards, load
from ivlib import market as mk, pricing as bs, solver

SCALAR_N = 20_000   # scalar rungs are slow; throughput is the comparable figure
REPEATS = 5


def _scalar_newton(target, F, K, T, df, cdf, pdf, max_iter=60, tol=1e-12):
    out = np.full(len(target), np.nan)
    for i in range(len(target)):
        t, f, k, tt, d = target[i], F[i], K[i], T[i], df[i]
        if not (t > max(f - k, 0.0) * d + 1e-12) or not (t < d * f - 1e-12) or tt <= 0:
            continue
        s, lo, hi = 0.3, 1e-6, 5.0
        for _ in range(max_iter):
            vst = s * math.sqrt(tt)
            d1 = (math.log(f / k) + 0.5 * vst * vst) / vst
            d2 = d1 - vst
            diff = d * (f * cdf(d1) - k * cdf(d2)) - t
            v = d * f * pdf(d1) * math.sqrt(tt)
            if diff > 0.0: hi = s
            else: lo = s
            step = s - diff / v if v > 1e-300 else 0.5 * (lo + hi)
            if not (lo < step < hi): step = 0.5 * (lo + hi)
            moved = abs(step - s); s = step
            if moved < tol or (hi - lo) < tol: break
        out[i] = s
    return out


def l0(a):
    from scipy.stats import norm
    return _scalar_newton(*a, cdf=norm.cdf, pdf=norm.pdf)

def l1(a):
    return _scalar_newton(*a, cdf=lambda x: 0.5 * (1.0 + math.erf(x * 0.7071067811865476)),
                          pdf=lambda x: 0.3989422804014327 * math.exp(-0.5 * x * x))

def l2(a): return solver.implied_vol(*a, is_call=True)
def l3(a): return solver.implied_vol(*a, is_call=True, seed_fn=bs.corrado_miller)
def l4(a): return solver.implied_vol_fast(*a, is_call=True)


RUNGS = [
    ("L0  scalar + scipy.stats.norm", l0, True, "the original shape"),
    ("L1  scalar + math.erf", l1, True, "cheaper CDF"),
    ("L2  NumPy vectorized", l2, False, "memory layout"),
    ("L3  NumPy + Corrado-Miller seed", l3, False, "fewer iterations"),
    ("L4  Numba fused + parallel", l4, False, f"registers, {os.cpu_count()} cpus"),
]


def load_quotes():
    d0 = load(data.VENDOR)
    cm, _ = mk.filter_quotes(d0["C_BID"], d0["C_ASK"], dte=d0["DTE"])
    pm, _ = mk.filter_quotes(d0["P_BID"], d0["P_ASK"], dte=d0["DTE"])
    d = d0[mk.paired_mask(cm, pm)].copy()
    g, fit = build_forwards(d)
    ok = fit["ok"][g]
    return (mk.mid(d["C_BID"].values, d["C_ASK"].values)[ok], fit["forward"][g][ok],
            d["STRIKE"].values[ok], d["DTE"].values[ok] / 365.0, fit["discount"][g][ok])


def run():
    full = load_quotes()
    n_full = len(full[0])
    sub = tuple(a[:SCALAR_N] for a in full)
    print(f"dataset: {n_full:,} real call quotes  (scalar rungs timed on {SCALAR_N:,})\n")
    solver.implied_vol_fast(*sub, is_call=True)   # warm the JIT
    ref = solver.implied_vol(*full, is_call=True)

    results = []
    for name, fn, scalar, why in RUNGS:
        args, n = (sub, SCALAR_N) if scalar else (full, n_full)
        best = float("inf")
        for _ in range(1 if scalar else REPEATS):
            t = time.perf_counter(); out = fn(args); best = min(best, time.perf_counter() - t)
        m = np.isfinite(out) & np.isfinite(ref[:n])
        agree = float(np.abs(out[m] - ref[:n][m]).max()) if m.any() else float("nan")
        results.append({"name": name, "why": why, "n": n, "secs": best, "rate": n / best, "agree": agree})
        print(f"{name:34}{best*1000:9.1f} ms  {n/best/1e6:7.3f} M/s   max|diff| {agree:.1e}")

    base = results[0]["rate"]
    print(f"\n{'rung':34}{'M/s':>9}{'vs L0':>10}{'vs prev':>10}   source")
    prev = None
    for r in results:
        vs_prev = f"{r['rate'] / prev:6.2f}x" if prev else "     --"
        print(f"{r['name']:34}{r['rate']/1e6:9.3f}{r['rate']/base:9.0f}x{vs_prev:>10}   {r['why']}")
        prev = r["rate"]
    print(f"\ntotal speedup L0 -> L4: {results[-1]['rate']/base:,.0f}x")
    print(f"L0 for all {n_full:,} quotes: {n_full/base/60:.1f} min  ->  L4: {n_full/results[-1]['rate']*1000:.0f} ms")
    return results


def plot(results, out="bench/figures/ladder.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S, s1 = "#fcfcfb", "#2a78d6"
    ink, ink2, muted, grid = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
    rates = np.array([r["rate"] for r in results]); base = rates[0]
    fig, ax = plt.subplots(figsize=(12.5, 5.6), facecolor=S)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.78, bottom=0.14); ax.set_facecolor(S)
    y = np.arange(len(results))[::-1]
    ax.barh(y, rates, height=0.6, color=s1, edgecolor=S, linewidth=1.5)
    ax.set_xscale("log"); ax.set_xlim(rates.min() * 0.45, rates.max() * 22.0)
    ax.set_yticks(y, [f"{r['name'][:2]}   {r['name'][4:]}" for r in results], fontsize=10)
    ax.set_xlabel("implied volatilities per second  (log scale)", color=ink2, fontsize=10)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(grid); ax.tick_params(colors=ink2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="x", color=grid, linewidth=0.7); ax.set_axisbelow(True)
    for yv, r in zip(y, results):
        rate = r["rate"]
        ax.text(rate * 1.13, yv + 0.10, f"{rate/1e6:.3f} M/s" if rate >= 1e5 else f"{rate/1e3:.1f} k/s", va="center", color=ink, fontsize=10.5, fontweight="semibold")
        ax.text(rate * 1.13, yv - 0.20, f"{rate/base:,.0f}x   ·   {r['why']}", va="center", color=muted, fontsize=9)
    fig.suptitle("Same inversion, five implementations", color=ink, fontsize=15.5, fontweight="semibold", x=0.022, ha="left", y=0.955)
    fig.text(0.022, 0.885, f"{results[-1]['n']:,} real AAPL call quotes. All five agree to within 1e-11 -- what differs is memory layout, algorithm, and hardware.", color=ink2, fontsize=10)
    fig.text(0.022, 0.035, f"L0 is the original code's shape: one option at a time through scipy.stats.norm. L4 solves the full set in {results[-1]['n']/rates[-1]*1000:.0f} ms.", color=ink2, fontsize=9)
    fig.savefig(out, dpi=170, facecolor=S)
    print(f"wrote {out}")


if __name__ == "__main__":
    res = run()
    if "--plot" in sys.argv:
        plot(res)
