"""Tests for the app layer. No network: sources are exercised on canned data."""

import numpy as np
import pandas as pd
import pytest

from app import data as schema, data as store, pipeline as snapshot, pipeline as compute, pipeline as realized, sources as yahoo, sources as cboe
from ivlib import pricing as bs


# ------------------------------------------------------------------ fixtures

def synth_chain(day="2026-03-02", spot=200.0, dtes=(14, 35, 91), state="REGULAR",
                two_sided=True, ticker="TEST"):
    """A schema-shaped chain with a known smile, as the Yahoo source would emit it."""
    rows = []
    for dte in dtes:
        T = dte / 365.0
        F = spot * 1.001
        K = np.round(F * np.exp(np.linspace(-0.25, 0.25, 21)) / 2.5) * 2.5
        k = np.log(K / F)
        iv = 0.28 + 0.5 * k ** 2 - 0.25 * k
        c = bs.call_price(F, K, iv, T, 1.0)
        p = bs.put_price(F, K, iv, T, 1.0)
        bid_c, ask_c = (c * 0.995, c * 1.005) if two_sided else (0.0 * c, 0.0 * c)
        bid_p, ask_p = (p * 0.995, p * 1.005) if two_sided else (0.0 * p, 0.0 * p)
        exp = (pd.Timestamp(day) + pd.Timedelta(days=dte)).date().isoformat()
        rows.append(pd.DataFrame({
            "QUOTE_DATE": day, "EXPIRE_DATE": exp, "DTE": float(dte),
            "UNDERLYING_LAST": spot, "STRIKE": K,
            "C_BID": bid_c, "C_ASK": ask_c, "P_BID": bid_p, "P_ASK": ask_p,
            "C_LAST": c, "P_LAST": p, "TICKER": ticker, "SOURCE": "test",
            "MARKET_STATE": state, "QUOTE_UNIXTIME": 1_700_000_000,
        }))
    return schema.validate(pd.concat(rows, ignore_index=True))


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.setattr(store, "CHAINS", tmp_path / "chains")
    monkeypatch.setattr(store, "METRICS", tmp_path / "metrics")
    return tmp_path


# ------------------------------------------------------------------ schema

def test_validate_requires_core_columns():
    with pytest.raises(ValueError, match="missing required"):
        schema.validate(pd.DataFrame({"STRIKE": [1.0]}))


def test_validate_rejects_negative_dte():
    ch = synth_chain()
    ch.loc[0, "DTE"] = -1
    with pytest.raises(ValueError, match="negative DTE"):
        schema.validate(ch)


def test_validate_fills_extra_columns_and_orders():
    ch = synth_chain().drop(columns=["C_OI", "P_OI"])
    out = schema.validate(ch)
    assert list(out.columns) == schema.COLUMNS
    assert out["C_OI"].isna().all()


def test_compact_trims_wings_and_downcasts():
    ch = synth_chain(dtes=(14, 35, 91, 500))
    c = schema.compact(ch, moneyness=0.10, max_dte=400)
    assert (c["DTE"] <= 400).all()
    spot = c["UNDERLYING_LAST"]
    assert (c["STRIKE"] >= spot * 0.9).all() and (c["STRIKE"] <= spot * 1.1).all()
    assert c["C_BID"].dtype == np.float32
    assert len(c) < len(ch)


def test_compact_is_lossless_for_the_engine():
    """Trimming must not change the ATM vol the engine recovers."""
    from ivlib import surface
    ch = synth_chain()
    full = surface.constant_maturity(surface.atm_term_structure(surface.build_iv_table(ch)), 30)
    f32 = schema.compact(ch)
    small = f32.astype({c: "float64" for c in f32.select_dtypes("float32").columns})
    trimmed = surface.constant_maturity(surface.atm_term_structure(surface.build_iv_table(small)), 30)
    assert full["atm_iv_30d"].iloc[0] == pytest.approx(trimmed["atm_iv_30d"].iloc[0], rel=1e-4)


# ------------------------------------------------------------------ yahoo (offline parts)

def test_is_live_requires_regular_hours_and_real_quotes():
    assert yahoo.is_live(synth_chain(state="REGULAR", two_sided=True))
    assert not yahoo.is_live(synth_chain(state="PRE", two_sided=True))
    assert not yahoo.is_live(synth_chain(state="REGULAR", two_sided=False))


def test_merge_sides_pairs_call_and_put_by_strike():
    calls = pd.DataFrame({"strike": [100.0, 105.0], "bid": [5, 2], "ask": [5.2, 2.2],
                          "lastPrice": [5.1, 2.1], "volume": [10, 5], "openInterest": [100, 50]})
    puts = pd.DataFrame({"strike": [105.0, 110.0], "bid": [3, 6], "ask": [3.2, 6.2],
                         "lastPrice": [3.1, 6.1], "volume": [7, 8], "openInterest": [70, 80]})
    m = yahoo._merge_sides(calls, puts, "2026-04-17")
    assert list(m["STRIKE"]) == [100.0, 105.0, 110.0]
    row = m[m["STRIKE"] == 105.0].iloc[0]
    assert row["C_BID"] == 2 and row["P_BID"] == 3
    assert np.isnan(m[m["STRIKE"] == 100.0]["P_BID"].iloc[0])


# ------------------------------------------------------------------ cboe

def test_cboe_parse_handles_fred_format_and_missing_values():
    csv = "observation_date,VXAPLCLS\n2024-01-02,25.5\n2024-01-03,.\n2024-01-04,27.0\n"
    s = cboe.parse_fred(csv)
    assert len(s) == 2                             # the '.' row is dropped
    assert s.iloc[0] == pytest.approx(0.255)       # percent -> decimal
    assert s.index[1] == pd.Timestamp("2024-01-04")


def test_cboe_coverage_is_exactly_five():
    assert cboe.cboe_available("AAPL") and cboe.cboe_available("gs")
    assert not cboe.cboe_available("MSFT")
    assert cboe.cboe_index("MSFT").empty                # no network call for unknown names


# ------------------------------------------------------------------ realized

def test_realized_trailing_and_forward_are_shifted_views_of_the_same_thing():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))),
                      index=pd.bdate_range("2024-01-01", periods=300))
    both = realized.realized_vol(close, window=21)
    tr, fw = both["rv21_trailing"], both["rv21_forward"]
    # The forward window ending at t+21 is the trailing window at t+21.
    i = 100
    assert fw.iloc[i] == pytest.approx(tr.iloc[i + 21], rel=1e-12)
    assert fw.iloc[-21:].isna().all()              # the future has not happened yet


def test_realized_annualization():
    """Constant daily sigma s -> annualized s*sqrt(252)."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, 5000)
    close = pd.Series(100 * np.exp(np.cumsum(r)), index=pd.bdate_range("2010-01-01", periods=5000))
    tr = realized.realized_vol(close, window=2000)["rv2000_trailing"].dropna()
    assert tr.mean() == pytest.approx(0.01 * np.sqrt(252), rel=0.03)


# ------------------------------------------------------------------ store

def test_store_roundtrip(tmp_store):
    ch = schema.compact(synth_chain(day="2026-03-02"))
    p = store.write_chain(ch)
    assert p.exists() and p.name == "2026-03-02.parquet"
    assert store.has_chain("TEST", "2026-03-02")
    assert store.chain_days("TEST") == ["2026-03-02"]
    assert store.tickers() == ["TEST"]
    back = store.read_chains("TEST")
    assert len(back) == len(ch)
    assert back["C_BID"].dtype == np.float64        # widened for the engine


def test_store_concatenates_days_in_order(tmp_store):
    for d in ("2026-03-03", "2026-03-02", "2026-03-04"):
        store.write_chain(schema.compact(synth_chain(day=d)))
    assert store.chain_days("TEST") == ["2026-03-02", "2026-03-03", "2026-03-04"]
    assert store.read_chains("TEST")["QUOTE_DATE"].nunique() == 3


# ------------------------------------------------------------------ snapshot

def test_snapshot_refuses_pre_market(tmp_store, monkeypatch):
    monkeypatch.setattr(yahoo, "fetch_chain", lambda t, **kw: synth_chain(state="PRE"))
    status, detail = snapshot.snapshot_one("TEST")
    assert status == "skipped" and "PRE" in detail
    assert not store.chain_days("TEST")


def test_snapshot_stores_live_chain_once(tmp_store, monkeypatch):
    monkeypatch.setattr(yahoo, "fetch_chain", lambda t, **kw: synth_chain(state="REGULAR"))
    assert snapshot.snapshot_one("TEST")[0] == "stored"
    assert snapshot.snapshot_one("TEST")[0] == "skipped"    # same day, no-op
    assert store.chain_days("TEST") == ["2026-03-02"]


def test_snapshot_force_overrides_guards(tmp_store, monkeypatch):
    monkeypatch.setattr(yahoo, "fetch_chain", lambda t, **kw: synth_chain(state="PRE"))
    assert snapshot.snapshot_one("TEST", force=True)[0] == "stored"


def test_snapshot_one_bad_ticker_does_not_stop_the_rest(tmp_store, monkeypatch, capsys):
    def fake(t, **kw):
        if t == "BAD":
            raise yahoo.ChainUnavailable("BAD: no listed options")
        return synth_chain(ticker=t)
    monkeypatch.setattr(yahoo, "fetch_chain", fake)
    rc = snapshot.main(["snapshot", "GOOD", "BAD", "ALSO"])
    assert rc == 1                                   # failure is reported...
    assert store.tickers() == ["ALSO", "GOOD"]       # ...but the others still stored


def test_watchlist_parsing(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("# header\nAAPL   # comment\n\n msft \nGME\n")
    assert snapshot.load_watchlist(p) == ["AAPL", "MSFT", "GME"]


# ------------------------------------------------------------------ compute

def test_implied_series_one_row_per_stored_day():
    chains = pd.concat([synth_chain(day="2026-03-02"), synth_chain(day="2026-03-03")],
                       ignore_index=True)
    s = compute.implied_series(chains)
    assert list(s.index) == [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")]
    assert s["atm_iv_30d"].between(0.25, 0.32).all()   # smile base was 0.28
    assert (s["coverage"] > 0.5).all()
    assert s["skew30"].gt(0).all()                     # smile slope was negative in k


def test_implied_series_empty_input():
    assert compute.implied_series(pd.DataFrame()).empty


# ------------------------------------------------------------------ watchlist add

def test_add_to_watchlist_validates_and_appends(tmp_path, monkeypatch):
    p = tmp_path / "w.txt"
    p.write_text("AAPL\n")
    monkeypatch.setattr(yahoo, "fetch_chain", lambda t, **kw: synth_chain(ticker=t))
    added, msg = snapshot.add_to_watchlist("nvda", p)
    assert added and snapshot.load_watchlist(p) == ["AAPL", "NVDA"]
    added, _ = snapshot.add_to_watchlist("NVDA", p)          # idempotent
    assert not added and snapshot.load_watchlist(p) == ["AAPL", "NVDA"]


def test_add_to_watchlist_rejects_unknown_and_garbage(tmp_path, monkeypatch):
    p = tmp_path / "w.txt"
    def fake(t, **kw):
        raise yahoo.ChainUnavailable(f"{t}: no listed options")
    monkeypatch.setattr(yahoo, "fetch_chain", fake)
    assert not snapshot.add_to_watchlist("ZZZZ", p)[0]
    assert not snapshot.add_to_watchlist("../etc", p)[0]
    assert not p.exists() or snapshot.load_watchlist(p) == []


# ------------------------------------------------------------------ remote (GitHub-backed watchlist)

def test_remote_append_is_idempotent_and_commits_once(monkeypatch):
    from app import sources as remote
    import base64
    monkeypatch.setenv("GITHUB_TOKEN", "x"); monkeypatch.setenv("GITHUB_REPO", "o/r")
    calls = []
    def fake_request(method, url, body=None):
        calls.append((method, body))
        if method == "GET":
            return {"content": base64.b64encode(b"AAPL   # note\nMSFT\n").decode(), "sha": "abc"}
        return {}
    monkeypatch.setattr(remote, "_gh", fake_request)

    added, _ = remote.github_append_ticker("msft")
    assert not added and [m for m, _ in calls] == ["GET"]        # no PUT for a duplicate

    calls.clear()
    added, _ = remote.github_append_ticker("nvda")
    assert added and [m for m, _ in calls] == ["GET", "PUT"]
    put = calls[1][1]
    assert put["sha"] == "abc"
    assert base64.b64decode(put["content"]).decode() == "AAPL   # note\nMSFT\nNVDA\n"


def test_remote_not_configured_without_env(monkeypatch):
    from app import sources as remote
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert not remote.github_configured()
