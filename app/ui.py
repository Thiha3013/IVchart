"""Streamlit page: type a ticker, see implied vs realized volatility over time.

This is the product the original project set out to build. Everything above it
in the repo exists so that this page can be trusted.

Run:  streamlit run app/ui.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run app/ui.py` puts app/ on sys.path, not the repo root. The package
# is pip-installed in the normal setup; this guard covers a bare checkout.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import compute, realized, snapshot, store
from app.sources import cboe, yahoo
from ivlib import filter as qf, parity, fast, surface

# Validated categorical palette (see bench/plot_*.py). Fixed slot order, never cycled.
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8983", "#e4e3df"

st.set_page_config(page_title="IVchart", page_icon="📈", layout="wide")


# ----------------------------------------------------------------- data access

@st.cache_data(ttl=300, show_spinner=False)
def live_chain(ticker: str) -> pd.DataFrame:
    return yahoo.fetch(ticker)


@st.cache_data(ttl=3600, show_spinner=False)
def metrics(ticker: str) -> pd.DataFrame:
    m = store.read_metrics(ticker)
    if m.empty:
        m = compute.build(ticker)
        store.write_metrics(ticker, m)
    return m


def latest_stored_chain(ticker: str) -> pd.DataFrame:
    days = store.chain_days(ticker)
    if not days:
        return pd.DataFrame()
    df = pd.read_parquet(store.chain_path(ticker, days[-1]))
    return df.astype({c: "float64" for c in df.select_dtypes("float32").columns})


# ----------------------------------------------------------------- figures

def _base_layout(fig, height=420, ytitle=None):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=12),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID, zeroline=False, showline=False),
        yaxis=dict(gridcolor=GRID, zeroline=False, showline=False, title=ytitle,
                   tickformat=".0%"),
    )
    return fig


def fig_timeseries(m: pd.DataFrame, show_cboe: bool) -> go.Figure:
    fig = go.Figure()
    series = [
        ("atm_iv_30d", "30d implied (ATM, ivlib)", S1, "solid"),
        ("rv21_trailing", "realized, trailing 21d", S2, "solid"),
        ("rv21_forward", "realized, next 21d", S3, "dot"),
    ]
    if show_cboe and "cboe_iv30" in m:
        series.append(("cboe_iv30", "Cboe vol index (variance strip)", MUTED, "dash"))
    for col, name, color, dash in series:
        if col not in m:
            continue
        s = m[col].dropna()
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, name=name, mode="lines",
            line=dict(color=color, width=2, dash=dash),
            hovertemplate="%{y:.1%}<extra>" + name + "</extra>",
        ))
    _base_layout(fig, height=440, ytitle="annualized volatility")
    fig.update_xaxes(rangeselector=dict(buttons=[
        dict(count=3, label="3m", step="month", stepmode="backward"),
        dict(count=1, label="1y", step="year", stepmode="backward"),
        dict(count=3, label="3y", step="year", stepmode="backward"),
        dict(step="all", label="all"),
    ], bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2)))
    return fig


def fig_smile(table: pd.DataFrame, day: str) -> go.Figure:
    fig = go.Figure()
    d = table[table["quote_date"] == day]
    picks = d.groupby("expire_date")["dte"].first().sort_values()
    targets = []
    for want in (14, 35, 90, 200):
        if len(picks):
            targets.append(picks.iloc[(picks - want).abs().argmin()])
    colors = [S1, S2, S3, S4]
    for dte, color in zip(dict.fromkeys(targets), colors):
        g = d[d["dte"] == dte].sort_values("log_moneyness")
        g = g[(g["log_moneyness"] > -0.30) & (g["log_moneyness"] < 0.25)]
        if len(g) < 4:
            continue
        fig.add_trace(go.Scatter(
            x=g["log_moneyness"], y=g["iv"], mode="lines", name=f"{int(dte)}d",
            line=dict(color=color, width=2),
            hovertemplate="k=%{x:+.2f}  iv=%{y:.1%}<extra>" + f"{int(dte)}d" + "</extra>",
        ))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1, dash="dash"))
    _base_layout(fig, height=360, ytitle="implied volatility")
    fig.update_xaxes(title="log-moneyness  ln(K / F)")
    fig.update_layout(hovermode="closest")
    return fig


def fig_funnel(report: dict) -> go.Figure:
    rows = [("kept", report["kept"])] + [(k, v) for k, v in report["rejected_by"].items() if v]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [S1 if l == "kept" else GRID for l in labels]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:,}" for v in vals], textposition="outside", textfont=dict(color=INK),
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    ))
    _base_layout(fig, height=60 + 28 * len(rows))
    fig.update_layout(showlegend=False, hovermode="closest", bargap=0.35)
    fig.update_yaxes(tickfont=dict(size=11))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    return fig


# ----------------------------------------------------------------- page

st.title("IVchart")
st.caption("Implied volatility from the option chain, solved by ivlib; realized volatility "
           "from price history. Implied history for a ticker starts the day it is first "
           "snapshotted -- AAPL has 2021-23 from a vendor CSV, five names have Cboe indices "
           "back to 2010.")

col_in, col_btn, col_sp = st.columns([2, 1, 4])
ticker = col_in.text_input("Ticker", value="AAPL", label_visibility="collapsed").upper().strip()
if col_btn.button("Snapshot now", help="Store today's chain. Only works during regular market hours."):
    status, detail = snapshot.snapshot_one(ticker)
    (st.success if status == "stored" else st.warning)(detail)
    live_chain.clear(); metrics.clear()

if not ticker:
    st.stop()

# ---- metrics -----------------------------------------------------------
try:
    with st.spinner(f"building series for {ticker}..."):
        m = metrics(ticker)
except yahoo.ChainUnavailable as e:
    st.error(str(e)); st.stop()

has_iv = "atm_iv_30d" in m and m["atm_iv_30d"].notna().any()
last_iv = m["atm_iv_30d"].dropna() if has_iv else pd.Series(dtype=float)
last_rv = m["rv21_trailing"].dropna()

t1, t2, t3, t4, t5 = st.columns(5)
t1.metric("30d implied", f"{last_iv.iloc[-1]:.1%}" if len(last_iv) else "—",
          help=f"as of {last_iv.index[-1].date()}" if len(last_iv) else "no snapshots yet")
t2.metric("21d realized", f"{last_rv.iloc[-1]:.1%}" if len(last_rv) else "—",
          help=f"as of {last_rv.index[-1].date()}" if len(last_rv) else None)
if len(last_iv) and len(last_rv):
    gap = last_iv.iloc[-1] - last_rv.iloc[-1]
    t3.metric("implied − realized", f"{gap:+.1%}",
              help="positive: options price more movement than the stock has recently shown")
else:
    t3.metric("implied − realized", "—")
sk = m["skew30"].dropna() if "skew30" in m else pd.Series(dtype=float)
t4.metric("30d skew", f"{sk.iloc[-1]:+.1%}" if len(sk) else "—",
          help="IV(−10%) − IV(+10%). Positive: downside strikes carry more vol.")
t5.metric("days of implied history", f"{len(last_iv):,}",
          help="one row per stored chain; grows by one each trading day the snapshot runs")

# ---- time series ---------------------------------------------------------
st.subheader("Implied vs realized")
if not has_iv:
    st.info(f"No implied-vol history for {ticker} yet. Realized vol is shown; implied "
            f"history begins with the first snapshot (add it to `app/watchlist.txt`, or press "
            f"**Snapshot now** during market hours).")
show_cboe = cboe.available(ticker) and st.toggle(
    "show Cboe vol index", value=True,
    help="Cboe's VIX-methodology index for this name. It is a variance-strip rate, so it "
         "prices in the skew and runs above at-the-money vol; the two track each other "
         "(AAPL: correlation 0.98) but are not the same quantity.")
st.plotly_chart(fig_timeseries(m, show_cboe), width='stretch')

with st.expander("table view"):
    cols = [c for c in ("atm_iv_30d", "atm_iv_90d", "rv21_trailing", "rv21_forward",
                        "skew30", "cboe_iv30", "coverage", "close") if c in m]
    st.dataframe(m[cols].dropna(how="all").sort_index(ascending=False).round(4),
                 width='stretch', height=300)

# ---- today's smile -------------------------------------------------------
st.subheader("Smile")
chain, label = pd.DataFrame(), ""
try:
    chain = live_chain(ticker)
    if yahoo.is_live(chain):
        label = f"live, {chain['QUOTE_DATE'].iloc[0]}"
    else:
        stored = latest_stored_chain(ticker)
        if not stored.empty:
            chain, label = stored, f"last stored chain, {stored['QUOTE_DATE'].iloc[0]}"
        else:
            state = chain["MARKET_STATE"].iloc[0]
            st.warning(f"Market is {state}: Yahoo returns empty bid/ask outside regular hours, "
                       f"and there is no stored chain for {ticker} yet. Smile unavailable.")
            chain = pd.DataFrame()
except yahoo.ChainUnavailable as e:
    st.warning(str(e))

if not chain.empty:
    cm, crep = qf.filter_quotes(chain["C_BID"], chain["C_ASK"], dte=chain["DTE"])
    pm, prep = qf.filter_quotes(chain["P_BID"], chain["P_ASK"], dte=chain["DTE"])
    table = surface.build_iv_table(chain)
    left, right = st.columns([3, 2])
    with left:
        st.caption(label)
        if table.empty:
            st.warning("No solvable quotes in this chain.")
        else:
            st.plotly_chart(fig_smile(table, str(chain["QUOTE_DATE"].iloc[0])),
                            width='stretch')
    with right:
        st.caption(f"quote quality — calls, {crep['kept']:,}/{crep['total']:,} usable "
                   f"({crep['kept_pct']:.0f}%)")
        st.plotly_chart(fig_funnel(crep), width='stretch')
        if not table.empty:
            n_exp = table["expire_date"].nunique()
            st.caption(f"{len(table):,} implied vols across {n_exp} expiries, "
                       f"forward + discount implied from put-call parity per expiry.")
