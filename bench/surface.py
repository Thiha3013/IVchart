"""The surface figure: 262,752 implied vols across 570 days -- 563x the 467 points the original produced.

    python -m bench.surface     -> bench/figures/surface.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app import pipeline
from bench.validate import load
from ivlib import surface

DAY = "2022-05-27"
S = dict(bg="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8a8983", grid="#e4e3df",
         s1="#2a78d6", s2="#eb6834", s3="#1baf7a", s4="#4a3aa7")


def build_series():
    """IV table, skew, and the 30d implied-vs-realized join, from the vendor parquet."""
    df = load()
    tab = surface.build_iv_table(df)
    term = surface.atm_term_structure(tab)
    sk = surface.skew(tab, wing=0.10)
    cm30 = surface.constant_maturity(term, days=30).set_index("quote_date")
    spot = pd.read_parquet(load.__defaults__[0], columns=["QUOTE_DATE", "UNDERLYING_LAST"])
    spot = spot.groupby("QUOTE_DATE")["UNDERLYING_LAST"].first().astype(float).sort_index()
    spot.index = pd.to_datetime(spot.index)
    cm30.index = pd.to_datetime(cm30.index)
    ivhv = cm30.join(pipeline.realized_vol(spot)).dropna()
    return tab, sk, ivhv


def plot(tab, sk, ivhv, out="bench/figures/surface.png"):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    seq = LinearSegmentedColormap.from_list("b", ["#eaf1fb", "#9cc1ea", S["s1"], "#123a6b"])

    def style(ax, grid="both"):
        ax.set_facecolor(S["bg"])
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color(S["grid"])
        ax.tick_params(colors=S["ink2"], labelsize=9, length=3, width=0.8)
        ax.grid(grid is not None, axis=grid or "both", color=S["grid"], linewidth=0.7); ax.set_axisbelow(True)

    fig, ((axA, axB), (axC, axD)) = plt.subplots(2, 2, figsize=(15.5, 10.2), facecolor=S["bg"])
    fig.subplots_adjust(left=0.055, right=0.975, top=0.865, bottom=0.07, hspace=0.38, wspace=0.20)

    # A: surface heatmap
    style(axA, None)
    kg, dtes, Z = surface.surface_grid(tab, DAY, k_grid=np.linspace(-0.22, 0.22, 45))
    keep = dtes <= 400; dtes, Z = dtes[keep], Z[keep]
    mesh = axA.pcolormesh(kg, np.arange(len(dtes)), Z, cmap=seq, shading="nearest")
    axA.set_yticks(np.arange(len(dtes)), [f"{int(d)}" for d in dtes], fontsize=8.5)
    axA.set_xlabel("log-moneyness  ln(K/F)", color=S["ink2"], fontsize=10); axA.set_ylabel("days to expiry", color=S["ink2"], fontsize=10)
    axA.axvline(0.0, color=S["bg"], linewidth=1.4, alpha=0.8)
    axA.set_title(f"A   The surface — {DAY}", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=9)
    cb = fig.colorbar(mesh, ax=axA, pad=0.015, fraction=0.045); cb.set_label("implied volatility", color=S["ink2"], fontsize=9)
    cb.ax.tick_params(colors=S["ink2"], labelsize=8, length=2); cb.outline.set_visible(False)

    # B: smiles
    style(axB)
    day = tab[tab["quote_date"] == DAY]
    picks = day.groupby("expire_date")["dte"].first().sort_values()
    targets = dict.fromkeys(picks.iloc[(picks - d).abs().argmin()] for d in (14, 35, 90, 200))
    for dte, col in zip(targets, (S["s1"], S["s2"], S["s3"], S["s4"])):
        g = day[day["dte"] == dte].sort_values("log_moneyness")
        g = g[(g["log_moneyness"] > -0.30) & (g["log_moneyness"] < 0.25)]
        if len(g) >= 5:
            axB.plot(g["log_moneyness"], g["iv"], color=col, linewidth=2.0, label=f"{int(dte)}d", zorder=3)
    axB.axvline(0.0, color=S["muted"], linewidth=1.0, linestyle=(0, (4, 3)))
    axB.text(0.004, 0.985, "ATM forward", transform=axB.get_xaxis_transform(), color=S["muted"], fontsize=8.5, rotation=90, va="top")
    axB.set_xlabel("log-moneyness  ln(K/F)", color=S["ink2"], fontsize=10); axB.set_ylabel("implied volatility", color=S["ink2"], fontsize=10)
    axB.set_title("B   The smile, sliced by maturity", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=9)
    axB.legend(frameon=False, fontsize=9.5, labelcolor=S["ink2"], loc="upper right", title="expiry", title_fontproperties={"size": 9})
    axB.get_legend().get_title().set_color(S["muted"])

    # C: implied vs realized
    style(axC)
    x = pd.to_datetime(ivhv.index)
    axC.plot(x, ivhv["atm_iv_30d"], color=S["s1"], linewidth=1.7, label="30d implied (constant maturity)")
    axC.plot(x, ivhv["rv21_trailing"], color=S["s2"], linewidth=1.7, label="30d realized (trailing)")
    axC.fill_between(x, ivhv["rv21_trailing"], ivhv["atm_iv_30d"], where=ivhv["atm_iv_30d"] >= ivhv["rv21_trailing"], color=S["s1"], alpha=0.13, linewidth=0)
    axC.set_ylabel("annualized volatility", color=S["ink2"], fontsize=10)
    axC.set_title("C   Implied vs realized volatility", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=9)
    axC.legend(frameon=False, fontsize=9.5, labelcolor=S["ink2"], loc="upper left")
    prem = ivhv["atm_iv_30d"] - ivhv["rv21_trailing"]
    axC.text(0.985, 0.04, f"implied above realized on {100*(prem>0).mean():.0f}% of days\nmean gap {prem.mean():+.3f} vol points",
             transform=axC.transAxes, ha="right", va="bottom", color=S["ink2"], fontsize=9.5, linespacing=1.5)

    # D: skew over time
    style(axD)
    s30 = sk[(sk["dte"] >= 20) & (sk["dte"] <= 45)].groupby("quote_date")["skew"].mean()
    axD.plot(pd.to_datetime(s30.index), s30.rolling(5).mean(), color=S["s1"], linewidth=1.7)
    axD.axhline(0.0, color=S["muted"], linewidth=1.0, linestyle=(0, (4, 3)))
    axD.set_ylabel("IV(-10%) − IV(+10%)", color=S["ink2"], fontsize=10)
    axD.set_title("D   Skew — what downside protection costs", color=S["ink"], fontsize=12, fontweight="semibold", loc="left", pad=9)
    axD.text(0.985, 0.05, "positive = puts richer than calls", transform=axD.transAxes, ha="right", va="bottom", color=S["muted"], fontsize=9)

    fig.suptitle("The surface the original project could not compute", color=S["ink"], fontsize=16, fontweight="semibold", x=0.055, ha="left", y=0.962)
    fig.text(0.055, 0.917, f"AAPL 2021-2023. {len(tab):,} implied volatilities across {tab['quote_date'].nunique()} days — 563x the 467 points the original produced, solved in 19 ms.",
             color=S["ink2"], fontsize=10.5, ha="left")
    fig.savefig(out, dpi=165, facecolor=S["bg"])
    print(f"wrote {out}")


if __name__ == "__main__":
    plot(*build_series())
