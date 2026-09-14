"""Validate the engine against the vendor's own implied vols, all 548k rows.

Runs both time conventions: T=DTE/365 (fixed) and T=DTE/252 (the original bug).
The buggy run should come back low by sqrt(252/365) = 0.8309.

    python -m bench.validate [--plot]      -> bench/figures/validation.png with --plot
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from app import data
from ivlib import market as mk, pricing as bs, solver

_STYLE = dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8983", grid="#e4e3df",
              s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#4a3aa7")


def load(path=data.VENDOR):
    df = pd.read_parquet(path)
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


def build_forwards(d):
    """One forward + discount per (quote date, expiry), weighted toward the money."""
    K = d["STRIKE"].values
    C, P = mk.mid(d["C_BID"].values, d["C_ASK"].values), mk.mid(d["P_BID"].values, d["P_ASK"].values)
    spot = d["UNDERLYING_LAST"].values
    g = mk.group_codes(d["QUOTE_DATE"].values, d["EXPIRE_DATE"].values)
    w = 1.0 / (1.0 + np.abs(K - spot) / spot * 10.0)
    return g, mk.implied_forward(K, C, P, group=g, weights=w)


def compare(iv, vendor):
    m = np.isfinite(iv) & np.isfinite(vendor) & (vendor > 0.01) & (vendor < 5.0)
    r = iv[m] / vendor[m]
    return {"n": int(m.sum()), "median": float(np.median(r)), "p10": float(np.percentile(r, 10)),
            "p90": float(np.percentile(r, 90)), "within_2pct": float(np.mean(np.abs(r - 1) < 0.02) * 100),
            "within_5pct": float(np.mean(np.abs(r - 1) < 0.05) * 100), "mask": m, "ratio": r}


def run(path=data.VENDOR):
    t0 = time.perf_counter()
    df = load(path)
    print(f"loaded {len(df):,} rows in {time.perf_counter()-t0:.2f}s\n")

    cm, crep = mk.filter_quotes(df["C_BID"], df["C_ASK"], dte=df["DTE"])
    pm, prep = mk.filter_quotes(df["P_BID"], df["P_ASK"], dte=df["DTE"])
    print(mk.format_report(crep, "call quotes")); print(mk.format_report(prep, "put quotes"))
    d = df[mk.paired_mask(cm, pm)].copy()
    print(f"\npaired strikes: {len(d):,}")

    g, fit = build_forwards(d)
    print(f"forwards: {fit['ok'].sum():,}/{len(fit['forward']):,} usable  median R2={np.nanmedian(fit['r2']):.7f}")

    ok = fit["ok"][g]
    F, DF, K = fit["forward"][g][ok], fit["discount"][g][ok], d["STRIKE"].values[ok]
    days = d["DTE"].values[ok]
    C = mk.mid(d["C_BID"].values, d["C_ASK"].values)[ok]
    vendor = d["C_IV"].values[ok]

    results = {}
    for label, T in (("fixed", days / 365.0), ("buggy", days / 252.0)):
        t = time.perf_counter()
        iv, info = solver.implied_vol(C, F, K, T, DF, True, return_info=True)
        cov = solver.coverage(info)
        results[label] = {**compare(iv, vendor), "iv": iv, "cov": cov}
        print(f"\n[{label}] solved {cov['solved']:,}/{cov['total']:,} in {time.perf_counter()-t:.3f}s "
              f"mean iters {cov['mean_iterations']} coverage {cov['coverage_pct']}%")

    print(f"\n{'':8}{'n':>10}{'median':>9}{'p10':>8}{'p90':>8}{'<2%':>8}{'<5%':>8}")
    for k in ("fixed", "buggy"):
        r = results[k]
        print(f"{k:8}{r['n']:>10,}{r['median']:>9.4f}{r['p10']:>8.4f}{r['p90']:>8.4f}{r['within_2pct']:>7.1f}%{r['within_5pct']:>7.1f}%")
    print(f"\nday-count bias   predicted sqrt(252/365) = {np.sqrt(252/365):.4f}")
    print(f"                 observed                 = {results['buggy']['median'] / results['fixed']['median']:.4f}")

    # residual disagreement vs the market's own resolution: half-spread / vega
    m = results["fixed"]["mask"]
    veg = bs.vega(F[m], K[m], results["fixed"]["iv"][m], days[m] / 365.0, DF[m])
    unc = 0.5 * (d["C_ASK"].values[ok][m] - d["C_BID"].values[ok][m]) / np.maximum(veg, 1e-12)
    r = results["fixed"]["ratio"]
    rows = []
    print(f"\nagreement vs quote resolution (half-spread / vega):")
    for lo, hi in ((0, 0.005), (0.005, 0.02), (0.02, 0.05), (0.05, 0.2), (0.2, np.inf)):
        s = (unc >= lo) & (unc < hi)
        if s.sum():
            rows.append((lo, hi, int(s.sum()), float(np.median(r[s])), float(np.mean(np.abs(r[s] - 1) < 0.02) * 100)))
            print(f"  +/-{lo:.3f}-{'inf' if hi == np.inf else f'{hi:.3f}':<6} n={rows[-1][2]:>8,}  median {rows[-1][3]:.4f}  <2% {rows[-1][4]:5.1f}%")
    good = unc < 0.02
    print(f"\nquotes the market pins to +/-2 vol points ({good.sum():,}, {100*good.mean():.1f}%): "
          f"median {np.median(r[good]):.4f}  <5% {100*np.mean(np.abs(r[good]-1)<0.05):.1f}%")
    results["quality"] = rows
    return results, vendor


def plot(results, vendor, out="bench/figures/validation.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    S = _STYLE
    seq = LinearSegmentedColormap.from_list("b", ["#eaf1fb", "#9cc1ea", S["s1"], "#123a6b"])
    fixed, buggy = results["fixed"], results["buggy"]
    m = fixed["mask"]

    def style(ax):
        ax.set_facecolor(S["surface"])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(S["grid"])
        ax.tick_params(colors=S["ink2"], labelsize=9, length=3, width=0.8)
        ax.grid(True, color=S["grid"], linewidth=0.7); ax.set_axisbelow(True)

    fig, (axL, axR, axQ) = plt.subplots(1, 3, figsize=(18.5, 5.6), facecolor=S["surface"],
                                        gridspec_kw={"width_ratios": [1.05, 1.0, 0.85]})
    fig.subplots_adjust(left=0.043, right=0.985, top=0.795, bottom=0.145, wspace=0.26)

    style(axL); lim = (0.0, 1.2)
    hb = axL.hexbin(vendor[m], fixed["iv"][m], gridsize=110, extent=(*lim, *lim), cmap=seq, norm=LogNorm(1, 3000), linewidths=0, mincnt=1)
    axL.plot(lim, lim, color=S["ink"], linewidth=1.4, linestyle=(0, (5, 3)), zorder=5)
    axL.annotate("y = x", xy=(0.92, 0.92), xytext=(0.99, 0.845), color=S["ink"], fontsize=9.5)
    axL.set_xlim(lim); axL.set_ylim(lim)
    axL.set_xlabel("vendor implied volatility", color=S["ink2"], fontsize=10); axL.set_ylabel("ivlib implied volatility", color=S["ink2"], fontsize=10)
    axL.set_title("Agreement with vendor marks", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=10)
    cb = fig.colorbar(hb, ax=axL, pad=0.015, fraction=0.045); cb.set_label("quotes per cell", color=S["ink2"], fontsize=9)
    cb.ax.tick_params(colors=S["ink2"], labelsize=8, length=2); cb.outline.set_visible(False)
    axL.text(0.035, 0.955, f"median ratio {fixed['median']:.4f}\n{fixed['within_5pct']:.0f}% within 5%\nn = {fixed['n']:,}",
             transform=axL.transAxes, va="top", fontsize=9.5, color=S["ink2"], linespacing=1.5)

    style(axR); bins = np.linspace(0.6, 1.4, 170)
    for res, color, label in ((fixed, S["s1"], "T = DTE / 365  (fixed)"), (buggy, S["s2"], "T = DTE / 252  (original)")):
        axR.hist(res["ratio"], bins=bins, color=color, alpha=0.85, edgecolor=S["surface"], linewidth=0.4, label=label)
        axR.axvline(res["median"], color=color, linewidth=1.6, zorder=6)
    axR.axvline(1.0, color=S["muted"], linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)
    axR.set_xlim(0.6, 1.4); axR.set_xlabel("ivlib implied vol  ÷  vendor implied vol", color=S["ink2"], fontsize=10); axR.set_ylabel("quotes", color=S["ink2"], fontsize=10)
    axR.set_title("Effect of the day-count convention", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=10)
    leg = axR.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=S["ink2"], handlelength=1.1, handleheight=1.1)
    for h in leg.legend_handles: h.set_edgecolor(S["surface"])
    ymax = axR.get_ylim()[1]
    axR.annotate(f"{buggy['median']:.4f}", xy=(buggy["median"], ymax*0.62), xytext=(buggy["median"]-0.085, ymax*0.70), color=S["ink"], fontsize=10, fontweight="semibold")
    axR.annotate(f"{fixed['median']:.4f}", xy=(fixed["median"], ymax*0.62), xytext=(fixed["median"]+0.022, ymax*0.70), color=S["ink"], fontsize=10, fontweight="semibold")
    axR.text(0.5, -0.165, f"shift = {buggy['median']/fixed['median']:.4f}      predicted  √(252/365) = {np.sqrt(252/365):.4f}",
             transform=axR.transAxes, ha="center", color=S["ink2"], fontsize=9.5)

    style(axQ); rows = results["quality"]
    labels = ["< 0.5", "0.5 - 2", "2 - 5", "5 - 20", "> 20"][:len(rows)]
    within, ns = [r[4] for r in rows], [r[2] for r in rows]
    ypos = np.arange(len(rows))[::-1]
    axQ.barh(ypos, within, height=0.62, color=S["s1"], edgecolor=S["surface"], linewidth=1.2)
    axQ.set_yticks(ypos, labels, fontsize=9.5); axQ.set_xlim(0, 100)
    axQ.set_xlabel("quotes matching vendor within 2%  (%)", color=S["ink2"], fontsize=10)
    axQ.set_ylabel("IV uncertainty from the quoted spread\n(vol points)", color=S["ink2"], fontsize=10)
    axQ.set_title("Disagreement tracks quote quality", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=10)
    axQ.grid(axis="y", visible=False)
    for yv, w, n in zip(ypos, within, ns):
        axQ.text(w + 1.8, yv, f"{w:.0f}%", va="center", color=S["ink"], fontsize=9.5, fontweight="semibold")
        axQ.text(2.0 if w > 28 else w + 9.5, yv, f"n={n:,}", va="center", color=S["surface"] if w > 28 else S["muted"], fontsize=8.5)
    axQ.text(0.0, -0.165, "a quote whose spread spans 10 vol points cannot pin IV to better than 10", transform=axQ.transAxes, color=S["ink2"], fontsize=9)

    fig.suptitle("Implied volatility engine validated against 251,086 vendor marks", color=S["ink"], fontsize=15.5, fontweight="semibold", x=0.043, ha="left", y=0.955)
    fig.text(0.043, 0.888, "AAPL options, 2021-2023, 548,163 quotes ingested. Forward and discount recovered from put-call parity; no external rate or dividend input.",
             color=S["ink2"], fontsize=10, ha="left")
    fig.savefig(out, dpi=170, facecolor=S["surface"])
    print(f"wrote {out}")


if __name__ == "__main__":
    res, vend = run()
    if "--plot" in sys.argv:
        plot(res, vend)
