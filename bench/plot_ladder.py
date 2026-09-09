"""Phase 3 figure: the optimization ladder.

One series (throughput), so no legend -- the title names it and every bar is
directly labelled. A log x-axis, because the rungs span several orders of
magnitude and a linear axis would collapse everything below L4 into nothing.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bench.ladder import main as run_ladder

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"


def build(results, out="bench/ladder.png"):
    names = [r["name"][4:] for r in results]
    tags = [r["name"][:2] for r in results]
    rates = np.array([r["rate"] for r in results])
    whys = [r["why"] for r in results]
    base = rates[0]

    fig, ax = plt.subplots(figsize=(12.5, 5.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.78, bottom=0.14)
    ax.set_facecolor(SURFACE)

    y = np.arange(len(results))[::-1]
    ax.barh(y, rates, height=0.6, color=SERIES_1, edgecolor=SURFACE, linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlim(rates.min() * 0.45, rates.max() * 22.0)
    ax.set_yticks(y, [f"{t}   {n}" for t, n in zip(tags, names)], fontsize=10)
    ax.set_xlabel("implied volatilities per second  (log scale)", color=INK_2, fontsize=10)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)

    for yv, rate, why in zip(y, rates, whys):
        mult = rate / base
        label = f"{rate/1e6:.3f} M/s" if rate >= 1e5 else f"{rate/1e3:.1f} k/s"
        ax.text(rate * 1.13, yv + 0.10, label, va="center", ha="left",
                color=INK, fontsize=10.5, fontweight="semibold")
        ax.text(rate * 1.13, yv - 0.20,
                f"{mult:,.0f}x" + (f"   ·   {why}" if why else ""),
                va="center", ha="left", color=MUTED, fontsize=9)

    fig.suptitle("Same inversion, five implementations", color=INK, fontsize=15.5,
                 fontweight="semibold", x=0.022, ha="left", y=0.955)
    fig.text(0.022, 0.885,
             f"262,796 real AAPL call quotes. All five agree to within 1e-11 -- what differs "
             f"is memory layout, algorithm, and hardware.",
             color=INK_2, fontsize=10, ha="left")
    fig.text(0.022, 0.035,
             f"L0 is the original code's shape: one option at a time through scipy.stats.norm. "
             f"L4 solves the full set in {262796/rates[-1]*1000:.0f} ms.",
             color=INK_2, fontsize=9, ha="left")

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    build(run_ladder())
