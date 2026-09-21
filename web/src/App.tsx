import { useEffect, useState } from "react";
import { api, type Metrics, type Smile, type TickerInfo } from "./api";
import { VolChart } from "./components/VolChart";
import { Funnel, SmileChart } from "./components/SmileChart";

const pct = (v: number | null | undefined, signed = false) =>
  v == null ? "—" : `${signed && v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`;

export default function App() {
  const [input, setInput] = useState("AAPL");
  const [ticker, setTicker] = useState("AAPL");
  const [known, setKnown] = useState<TickerInfo[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [smile, setSmile] = useState<Smile | null>(null);
  const [showCboe, setShowCboe] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [flash, setFlash] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => { api.tickers().then(setKnown).catch(() => {}); }, []);

  useEffect(() => {
    let dead = false;
    setLoading(true); setError(null); setMetrics(null); setSmile(null);
    Promise.all([api.metrics(ticker), api.smile(ticker).catch((e) => ({ ticker, available: false, reason: e.message, market_state: "?" } as Smile))])
      .then(([m, s]) => { if (!dead) { setMetrics(m); setSmile(s); } })
      .catch((e) => { if (!dead) setError(e.message); })
      .finally(() => { if (!dead) setLoading(false); });
    return () => { dead = true; };
  }, [ticker]);

  const submit = (e: React.FormEvent) => { e.preventDefault(); const t = input.trim().toUpperCase(); if (t) setTicker(t); };

  const snapshot = async () => {
    setFlash(null);
    const r = await api.snapshot(ticker);
    setFlash({ ok: r.status === "stored", text: r.detail });
    if (r.status === "stored") { api.metrics(ticker).then(setMetrics); api.smile(ticker).then(setSmile); }
  };

  const track = async () => {
    setFlash(null);
    try {
      const r = await api.track(ticker);
      setFlash({ ok: r.added, text: r.detail });
      if (r.added) api.tickers().then(setKnown);
    } catch (e: any) { setFlash({ ok: false, text: e.message }); }
  };

  const s = metrics?.summary;
  const info = known.find((k) => k.ticker === ticker);
  const watched = info?.watched ?? false;

  return (
    <>
      <header className="top">
        <div>
          <h1>IVchart</h1>
          <p className="sub">
            Implied volatility solved from the option chain by <code>ivlib</code>, against realized volatility from
            price history. Implied history for a ticker begins the day it is first snapshotted.
          </p>
        </div>
        <form className="search" onSubmit={submit}>
          <input list="tickers" value={input} onChange={(e) => setInput(e.target.value)} placeholder="ticker" aria-label="ticker" />
          <datalist id="tickers">{known.map((k) => <option key={k.ticker} value={k.ticker} />)}</datalist>
          <button className="primary" type="submit">Show</button>
          {metrics && !watched && (
            <button type="button" onClick={track} title="Add to the watchlist so the daily snapshot starts building history for it.">Track {ticker}</button>
          )}
          <button type="button" onClick={snapshot} title="Store today's chain. Only works during regular market hours.">Snapshot now</button>
        </form>
      </header>

      {known.length > 0 && (
        <div className="chips" aria-label="tracked tickers">
          <span className="muted">tracked, snapshotted daily at 14:00 ET —</span>
          {known.map((k) => (
            <button key={k.ticker} type="button" className={`chip ${k.ticker === ticker ? "on" : ""}`}
                    onClick={() => { setInput(k.ticker); setTicker(k.ticker); }}
                    title={k.days_stored ? `${k.days_stored} days of snapshots` : "no snapshots yet"}>
              {k.ticker}
              <span className="n">{historyNote(k)}</span>
            </button>
          ))}
        </div>
      )}

      {flash && <div className={`notice ${flash.ok ? "ok" : ""}`}>{flash.text}</div>}
      {error && <div className="notice err">{error}</div>}
      {loading && <p className="muted">loading {ticker}…</p>}

      {s && (
        <>
          <div className="tiles">
            <Tile label="30d implied" value={pct(s.iv30)} note={s.iv30_date ? `as of ${s.iv30_date}` : "no snapshots yet"} />
            <Tile label="21d realized" value={pct(s.rv21)} note={s.rv21_date ? `as of ${s.rv21_date}` : ""} />
            <Tile label="implied − realized" value={pct(s.gap, true)} note={s.gap_date ? `same day, ${s.gap_date}` : "needs both on one day"} />
            <Tile label="30d skew" value={pct(s.skew30, true)} note="IV(−10%) − IV(+10%)" />
            <Tile label="days of implied history" value={s.days_implied.toLocaleString()} note={watched ? "on the watchlist" : "not tracked yet"} />
          </div>

          <h2>Implied vs realized</h2>
          {s.days_implied === 0 && (
            <div className="notice">
              No implied-vol history for {ticker} yet — realized vol is shown. Implied history begins with the first
              snapshot: add {ticker} to <code>app/watchlist.txt</code>, or press <b>Snapshot now</b> during market hours.
            </div>
          )}
          {s.cboe_index && (
            <label className="muted" style={{ display: "inline-block", marginBottom: 8, cursor: "pointer" }}>
              <input type="checkbox" checked={showCboe} onChange={(e) => setShowCboe(e.target.checked)} />{" "}
              show Cboe vol index — a variance-strip rate that prices in the skew, so it runs above ATM vol; the two track
              (AAPL: correlation 0.98) but are not the same quantity
            </label>
          )}
          <VolChart series={metrics!.series} showCboe={showCboe && s.cboe_index} />
        </>
      )}

      {smile && (
        <>
          <h2>Smile</h2>
          <div className="two">
            <SmileChart smile={smile} />
            <Funnel smile={smile} />
          </div>
        </>
      )}
    </>
  );
}

function historyNote(k: TickerInfo): string {
  if (k.days_stored) return `${k.days_stored}d`;
  if (k.vendor_history) return "2021–23";
  if (k.cboe_index) return "Cboe";
  return "";
}

function Tile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="tile">
      <div className="label">{label}</div>
      <div className="value mono">{value}</div>
      {note && <div className="note">{note}</div>}
    </div>
  );
}
