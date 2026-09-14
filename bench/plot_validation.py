"""Phase 2 figure: the engine's implied vols against the vendor's own marks.

Two panels, two different jobs:

  left   magnitude/agreement -- a 2-D density of computed vs vendor implied vol.
         251k quotes overplot hopelessly as a scatter, so density is the honest
         form; the y=x line is the reference the cloud should sit on.

  middle polarity/shift -- the distribution of the ratio under both time
         conventions. This is where the day-count bug becomes visible as a clean
         translation of the whole distribution rather than added noise.

  right  where the residual disagreement lives. Quotes are bucketed by how
         tightly their own bid-ask spread pins down a volatility (half-spread
         divided by vega). Agreement tracks that resolution almost perfectly,
         which says the remaining spread is the market's, not the solver's.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm

from bench.validate import main as run_validation

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8983"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"   # categorical slot 1 -- fixed
SERIES_2 = "#eb6834"   # categorical slot 2 -- buggy
# Sequential ramp: one hue, light -> dark. Never a rainbow.
SEQ = LinearSegmentedColormap.from_list("blues", ["#eaf1fb", "#9cc1ea", "#2a78d6", "#123a6b"])


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.9)
    ax.set_axisbelow(True)


def build(results, vendor, out="bench/validation.png"):
    fixed, buggy = results["fixed"], results["buggy"]
    m = fixed["mask"]
    x = vendor[m]
    y = fixed["iv"][m]

    fig, (axL, axR, axQ) = plt.subplots(
        1, 3, figsize=(18.5, 5.6), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1.05, 1.0, 0.85]})
    fig.subplots_adjust(left=0.043, right=0.985, top=0.795, bottom=0.145, wspace=0.26)

    # ---------------- left: agreement ----------------
    style(axL)
    lim = (0.0, 1.2)
    hb = axL.hexbin(x, y, gridsize=110, extent=(*lim, *lim), cmap=SEQ,
                    norm=LogNorm(vmin=1, vmax=3000), linewidths=0, mincnt=1)
    axL.plot(lim, lim, color=INK, linewidth=1.4, linestyle=(0, (5, 3)), zorder=5)
    axL.annotate("y = x", xy=(0.92, 0.92), xytext=(0.99, 0.845),
                 color=INK, fontsize=9.5, ha="left")
    axL.set_xlim(lim); axL.set_ylim(lim)
    axL.set_xlabel("vendor implied volatility", color=INK_2, fontsize=10)
    axL.set_ylabel("ivlib implied volatility", color=INK_2, fontsize=10)
    axL.set_title("Agreement with vendor marks", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=10)

    cb = fig.colorbar(hb, ax=axL, pad=0.015, fraction=0.045)
    cb.set_label("quotes per cell", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8, length=2)
    cb.outline.set_visible(False)

    axL.text(0.035, 0.955, f"median ratio {fixed['median']:.4f}\n"
                           f"{fixed['within_5pct']:.0f}% within 5%\n"
                           f"n = {fixed['n']:,}",
             transform=axL.transAxes, va="top", ha="left", fontsize=9.5,
             color=INK_2, linespacing=1.5)

    # ---------------- right: the day-count shift ----------------
    style(axR)
    bins = np.linspace(0.6, 1.4, 170)
    for res, color, label in ((fixed, SERIES_1, "T = DTE / 365  (fixed)"),
                              (buggy, SERIES_2, "T = DTE / 252  (original)")):
        axR.hist(res["ratio"], bins=bins, color=color, alpha=0.85,
                 edgecolor=SURFACE, linewidth=0.4, label=label)

    for xv, color in ((buggy["median"], SERIES_2), (fixed["median"], SERIES_1)):
        axR.axvline(xv, color=color, linewidth=1.6, zorder=6)
    axR.axvline(1.0, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=4)

    axR.set_xlim(0.6, 1.4)
    axR.set_xlabel("ivlib implied vol  ÷  vendor implied vol", color=INK_2, fontsize=10)
    axR.set_ylabel("quotes", color=INK_2, fontsize=10)
    axR.set_title("Effect of the day-count convention", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=10)

    leg = axR.legend(loc="upper left", frameon=False, fontsize=9.5,
                     labelcolor=INK_2, handlelength=1.1, handleheight=1.1)
    for h in leg.legend_handles:
        h.set_edgecolor(SURFACE)

    ymax = axR.get_ylim()[1]
    axR.annotate(f"{buggy['median']:.4f}", xy=(buggy["median"], ymax * 0.62),
                 xytext=(buggy["median"] - 0.085, ymax * 0.70),
                 color=INK, fontsize=10, fontweight="semibold")
    axR.annotate(f"{fixed['median']:.4f}", xy=(fixed["median"], ymax * 0.62),
                 xytext=(fixed["median"] + 0.022, ymax * 0.70),
                 color=INK, fontsize=10, fontweight="semibold")

    obs = buggy["median"] / fixed["median"]
    axR.text(0.5, -0.165,
             f"shift = {obs:.4f}      predicted  √(252/365) = {np.sqrt(252/365):.4f}",
             transform=axR.transAxes, ha="center", color=INK_2, fontsize=9.5)

    # ---------------- right: where the disagreement lives ----------------
    style(axQ)
    rows = results["quality"]["buckets"]
    labels = [("<0.5" if lo == 0 else f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}")
              for lo, hi, *_ in rows]
    labels = ["< 0.5", "0.5 - 2", "2 - 5", "5 - 20", "> 20"][:len(rows)]
    within = [r[4] for r in rows]
    ns = [r[2] for r in rows]
    ypos = np.arange(len(rows))[::-1]

    axQ.barh(ypos, within, height=0.62, color=SERIES_1, edgecolor=SURFACE, linewidth=1.2)
    axQ.set_yticks(ypos, labels, fontsize=9.5)
    axQ.set_xlim(0, 100)
    axQ.set_xlabel("quotes matching vendor within 2%  (%)", color=INK_2, fontsize=10)
    axQ.set_ylabel("IV uncertainty from the quoted spread\n(vol points)",
                   color=INK_2, fontsize=10)
    axQ.set_title("Disagreement tracks quote quality", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=10)
    axQ.grid(axis="y", visible=False)
    for yv, w, n in zip(ypos, within, ns):
        axQ.text(w + 1.8, yv, f"{w:.0f}%", va="center", ha="left",
                 color=INK, fontsize=9.5, fontweight="semibold")
        # The count sits inside the bar only when the bar is long enough to hold
        # it; otherwise it trails the percentage in ink so the two never collide.
        if w > 28:
            axQ.text(2.0, yv, f"n={n:,}", va="center", ha="left",
                     color=SURFACE, fontsize=8.5)
        else:
            axQ.text(w + 9.5, yv, f"n={n:,}", va="center", ha="left",
                     color=MUTED, fontsize=8.5)

    axQ.text(0.0, -0.165,
             "a quote whose spread spans 10 vol points cannot pin IV to better than 10",
             transform=axQ.transAxes, ha="left", color=INK_2, fontsize=9)

    fig.suptitle("Implied volatility engine validated against 251,086 vendor marks",
                 color=INK, fontsize=15.5, fontweight="semibold", x=0.043, ha="left", y=0.955)
    fig.text(0.043, 0.888,
             "AAPL options, 2021-2023, 548,163 quotes ingested. Forward and discount recovered from "
             "put-call parity; no external rate or dividend input.",
             color=INK_2, fontsize=10, ha="left")

    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print(f"wrote {out}")
    return out


if __name__ == "__main__":
    res, vend = run_validation()
    build(res, vend)
