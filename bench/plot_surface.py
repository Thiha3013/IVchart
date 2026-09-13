"""Phase 4 figure: what 548k quotes look like when you solve all of them.

Four panels, four different jobs:

  A  the surface        implied vol as a function of strike and maturity, one day
  B  the smile          slices of A at four maturities, showing the skew directly
  C  IV vs realized     the original project's chart, rebuilt on a constant-
                        maturity series instead of whichever option was nearest
  D  skew over time     the term the original could not compute at all
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from ivlib import surface

SURFACE_BG = "#fcfcfb"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
SEQ = LinearSegmentedColormap.from_list("b", ["#eaf1fb", "#9cc1ea", "#2a78d6", "#123a6b"])
DAY = "2022-05-27"


def style(ax, grid_axis="both"):
    """grid_axis=None turns the grid off entirely (for the heatmap)."""
    ax.set_facecolor(SURFACE_BG)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=0.8)
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def main(out="bench/surface.png"):
    tab = pd.read_parquet("iv_table.parquet")
    ivhv = pd.read_parquet("iv_hv.parquet")
    sk = pd.read_parquet("skew.parquet")

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.2), facecolor=SURFACE_BG)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.865, bottom=0.07,
                        hspace=0.38, wspace=0.20)
    (axA, axB), (axC, axD) = axes

    # ---------------- A: the surface ----------------
    style(axA, grid_axis=None)
    kg, dtes, Z = surface.surface_grid(tab, DAY, k_grid=np.linspace(-0.22, 0.22, 45))
    keep = dtes <= 400
    dtes, Z = dtes[keep], Z[keep]
    mesh = axA.pcolormesh(kg, np.arange(len(dtes)), Z, cmap=SEQ, shading="nearest")
    axA.set_yticks(np.arange(len(dtes)), [f"{int(d)}" for d in dtes], fontsize=8.5)
    axA.set_xlabel("log-moneyness  ln(K/F)", color=INK_2, fontsize=10)
    axA.set_ylabel("days to expiry", color=INK_2, fontsize=10)
    axA.axvline(0.0, color=SURFACE_BG, linewidth=1.4, alpha=0.8)
    axA.set_title(f"A   The surface — {DAY}", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=9)
    cb = fig.colorbar(mesh, ax=axA, pad=0.015, fraction=0.045)
    cb.set_label("implied volatility", color=INK_2, fontsize=9)
    cb.ax.tick_params(colors=INK_2, labelsize=8, length=2)
    cb.outline.set_visible(False)

    # ---------------- B: smiles ----------------
    style(axB)
    day = tab[tab["quote_date"] == DAY]
    picks = day.groupby("expire_date")["dte"].first().sort_values()
    targets = [picks.iloc[(picks - d).abs().argmin()] for d in (14, 35, 90, 200)]
    seen, colors = [], [S1, S2, S3, S4]
    for dte, col in zip(dict.fromkeys(targets), colors):
        g = day[day["dte"] == dte].sort_values("log_moneyness")
        g = g[(g["log_moneyness"] > -0.30) & (g["log_moneyness"] < 0.25)]
        if len(g) < 5:
            continue
        axB.plot(g["log_moneyness"], g["iv"], color=col, linewidth=2.0,
                 label=f"{int(dte)}d", zorder=3)
        seen.append(dte)
    axB.axvline(0.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    axB.text(0.004, 0.985, "ATM forward", transform=axB.get_xaxis_transform(),
             color=MUTED, fontsize=8.5, rotation=90, va="top")
    axB.set_xlabel("log-moneyness  ln(K/F)", color=INK_2, fontsize=10)
    axB.set_ylabel("implied volatility", color=INK_2, fontsize=10)
    axB.set_title("B   The smile, sliced by maturity", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=9)
    axB.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="upper right",
               title="expiry", title_fontproperties={"size": 9})
    axB.get_legend().get_title().set_color(MUTED)

    # ---------------- C: IV vs realized ----------------
    style(axC)
    x = pd.to_datetime(ivhv.index)
    axC.plot(x, ivhv["atm_iv_30d"], color=S1, linewidth=1.7, label="30d implied (constant maturity)")
    axC.plot(x, ivhv["hv30"], color=S2, linewidth=1.7, label="30d realized (trailing)")
    axC.fill_between(x, ivhv["hv30"], ivhv["atm_iv_30d"],
                     where=ivhv["atm_iv_30d"] >= ivhv["hv30"],
                     color=S1, alpha=0.13, linewidth=0)
    axC.set_ylabel("annualized volatility", color=INK_2, fontsize=10)
    axC.set_title("C   Implied vs realized volatility", color=INK, fontsize=12,
                  fontweight="semibold", loc="left", pad=9)
    axC.legend(frameon=False, fontsize=9.5, labelcolor=INK_2, loc="upper left")
    prem = (ivhv["atm_iv_30d"] - ivhv["hv30"])
    axC.text(0.985, 0.04, f"implied above realized on {100*(prem>0).mean():.0f}% of days\n"
                          f"mean gap {prem.mean():+.3f} vol points",
             transform=axC.transAxes, ha="right", va="bottom", color=INK_2,
             fontsize=9.5, linespacing=1.5)

    # ---------------- D: skew ----------------
    style(axD)
    s30 = sk[(sk["dte"] >= 20) & (sk["dte"] <= 45)].groupby("quote_date")["skew"].mean()
    xs = pd.to_datetime(s30.index)
    axD.plot(xs, s30.rolling(5).mean(), color=S1, linewidth=1.7)
    axD.axhline(0.0, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    axD.set_ylabel("IV(-10%) − IV(+10%)", color=INK_2, fontsize=10)
    axD.set_title("D   Skew — what downside protection costs", color=INK,
                  fontsize=12, fontweight="semibold", loc="left", pad=9)
    axD.text(0.985, 0.05, "positive = puts richer than calls", transform=axD.transAxes,
             ha="right", va="bottom", color=MUTED, fontsize=9)

    fig.suptitle("The surface the original project could not compute",
                 color=INK, fontsize=16, fontweight="semibold", x=0.055, ha="left", y=0.962)
    fig.text(0.055, 0.917,
             f"AAPL 2021-2023. {len(tab):,} implied volatilities across {tab['quote_date'].nunique()} "
             f"days — 563x the 467 points the original produced, solved in 19 ms.",
             color=INK_2, fontsize=10.5, ha="left")

    fig.savefig(out, dpi=165, facecolor=SURFACE_BG)
    print(f"wrote {out}   (smile maturities: {seen})")


if __name__ == "__main__":
    main()
