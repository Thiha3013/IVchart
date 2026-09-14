import { useMemo, useState } from "react";
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import type { Point } from "../api";
import { usePalette } from "../palette";

type Range = "3m" | "1y" | "3y" | "all";
const RANGE_DAYS: Record<Range, number> = { "3m": 92, "1y": 366, "3y": 1096, all: 1e9 };

interface SeriesDef { key: keyof Point; name: string; color: string; dash?: string; note?: string }

const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);
const fmtDate = (t: number) =>
  new Date(t).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
const fmtTick = (t: number) =>
  new Date(t).toLocaleDateString(undefined, { year: "2-digit", month: "short" });

export function VolChart({ series, showCboe }: { series: Point[]; showCboe: boolean }) {
  const pal = usePalette();
  const [range, setRange] = useState<Range>("3y");
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  const defs: SeriesDef[] = [
    { key: "atm_iv_30d", name: "30d implied (ATM)", color: pal.s1 },
    { key: "rv21_trailing", name: "realized, trailing 21d", color: pal.s2 },
    { key: "rv21_forward", name: "realized, next 21d", color: pal.s3, dash: "2 3" },
    ...(showCboe ? [{ key: "cboe_iv30" as const, name: "Cboe vol index", color: pal.muted, dash: "6 4" }] : []),
  ];

  const data = useMemo(() => {
    const rows = series.map((p) => ({ ...p, t: Date.parse(p.date) }));
    const last = rows.length ? rows[rows.length - 1].t : 0;
    const cutoff = last - RANGE_DAYS[range] * 86_400_000;
    return rows.filter((r) => r.t >= cutoff);
  }, [series, range]);

  const toggle = (k: string) =>
    setHidden((h) => { const n = new Set(h); n.has(k) ? n.delete(k) : n.add(k); return n; });

  return (
    <div className="card">
      <div className="card-head">
        <div className="legend">
          {defs.map((d) => (
            <label key={d.key} className={hidden.has(d.key) ? "off" : ""} onClick={() => toggle(d.key)}>
              <span className={`sw ${d.dash ? (d.dash === "2 3" ? "dot" : "dash") : ""}`} style={{ borderColor: d.color }} />
              {d.name}
            </label>
          ))}
        </div>
        <div className="seg">
          {(["3m", "1y", "3y", "all"] as Range[]).map((r) => (
            <button key={r} className={range === r ? "on" : ""} onClick={() => setRange(r)}>{r}</button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={380}>
        <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
          <CartesianGrid stroke={pal.line} vertical={false} />
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} scale="time"
                 tickFormatter={fmtTick} stroke={pal.line} tick={{ fill: pal.ink2, fontSize: 11 }}
                 tickLine={false} minTickGap={48} />
          <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="transparent"
                 tick={{ fill: pal.ink2, fontSize: 11 }} width={44} domain={[0, "auto"]} />
          <Tooltip content={<Tip defs={defs} />} cursor={{ stroke: pal.muted, strokeWidth: 1 }} />
          {defs.map((d) => (
            <Line key={d.key} type="monotone" dataKey={d.key} name={d.name} stroke={d.color}
                  strokeWidth={2} strokeDasharray={d.dash} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: pal.surface }}
                  connectNulls={false} isAnimationActive={false} hide={hidden.has(d.key)} />
          ))}
        </LineChart>
      </ResponsiveContainer>

      <details>
        <summary>table view</summary>
        <div className="scroll">
          <table>
            <thead><tr><th>date</th>{defs.map((d) => <th key={d.key}>{d.name}</th>)}<th>close</th></tr></thead>
            <tbody>
              {[...data].reverse().filter((r) => defs.some((d) => r[d.key] != null)).slice(0, 400).map((r) => (
                <tr key={r.date} className="mono">
                  <td>{r.date}</td>
                  {defs.map((d) => <td key={d.key}>{pct(r[d.key] as number | null)}</td>)}
                  <td>{r.close == null ? "—" : r.close.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}

function Tip({ active, payload, defs }: { active?: boolean; payload?: any[]; defs: SeriesDef[] }) {
  if (!active || !payload?.length) return null;
  const row = payload[0].payload;
  return (
    <div className="tip">
      <div className="d">{fmtDate(row.t)}</div>
      {defs.map((d) => row[d.key] == null ? null : (
        <div className="r" key={d.key}>
          <span className="k" style={{ ["--c" as any]: d.color }}>{d.name}</span>
          <span className="mono">{pct(row[d.key])}</span>
        </div>
      ))}
      {row.close != null && <div className="r"><span className="k" style={{ ["--c" as any]: "transparent" }}>close</span><span className="mono">{row.close.toFixed(2)}</span></div>}
    </div>
  );
}
