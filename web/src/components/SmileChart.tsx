import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Smile } from "../api";
import { usePalette } from "../palette";

const SOURCE_LABEL = {
  live: "live chain",
  stored: "last stored chain",
  vendor: "vendor chain (2021-23 dataset)",
};

export function SmileChart({ smile }: { smile: Smile }) {
  const pal = usePalette();
  if (!smile.available || !smile.curves?.length) {
    return (
      <div className="card">
        <div className="notice">
          Smile unavailable — {smile.reason ?? "no solvable quotes"}.
          {smile.market_state !== "REGULAR" && (
            <> Yahoo returns empty bid/ask outside regular hours; a stored chain appears here once the daily snapshot has run for this ticker.</>
          )}
        </div>
      </div>
    );
  }

  const colors = [pal.s1, pal.s2, pal.s3, pal.s4];
  // One row per k across all curves so Recharts can draw them on a shared axis.
  const ks = new Set<number>();
  smile.curves.forEach((c) => c.points.forEach((p) => ks.add(+p.k.toFixed(4))));
  const rows = [...ks].sort((a, b) => a - b).map((k) => {
    const r: Record<string, number> = { k };
    smile.curves!.forEach((c) => {
      const p = c.points.find((q) => +q.k.toFixed(4) === k);
      if (p) r[`d${c.dte}`] = p.iv;
    });
    return r;
  });

  return (
    <div className="card">
      <div className="card-head">
        <div className="legend">
          {smile.curves.map((c, i) => (
            <span key={c.dte}><span className="sw" style={{ borderColor: colors[i] }} />{c.dte}d <span className="muted">({c.expiry})</span></span>
          ))}
        </div>
        <span className="muted">
          {SOURCE_LABEL[smile.source!]} · {smile.date} · spot {smile.spot?.toFixed(2)}
        </span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={rows} margin={{ top: 8, right: 12, left: 0, bottom: 4 }}>
          <CartesianGrid stroke={pal.line} vertical={false} />
          <XAxis dataKey="k" type="number" domain={[-0.3, 0.25]} tickFormatter={(v) => (v > 0 ? `+${v.toFixed(2)}` : v.toFixed(2))}
                 stroke={pal.line} tick={{ fill: pal.ink2, fontSize: 11 }} tickLine={false}
                 label={{ value: "log-moneyness  ln(K / F)", position: "insideBottom", offset: -2, fill: pal.ink2, fontSize: 11 }} />
          <YAxis tickFormatter={(v) => `${Math.round(v * 100)}%`} stroke="transparent" tick={{ fill: pal.ink2, fontSize: 11 }} width={44} domain={["auto", "auto"]} />
          <ReferenceLine x={0} stroke={pal.muted} strokeDasharray="4 3" label={{ value: "ATM fwd", fill: pal.muted, fontSize: 10, position: "insideTopRight" }} />
          <Tooltip
            formatter={(v: number, name: string) => [`${(v * 100).toFixed(1)}%`, name.replace("d", "") + "d"]}
            labelFormatter={(k: number) => `k = ${k > 0 ? "+" : ""}${k.toFixed(3)}`}
            contentStyle={{ background: pal.surface, border: `1px solid ${pal.line}`, borderRadius: 8, fontSize: 12.5 }}
          />
          {smile.curves.map((c, i) => (
            <Line key={c.dte} type="monotone" dataKey={`d${c.dte}`} name={`${c.dte}d`} stroke={colors[i]}
                  strokeWidth={2} dot={false} connectNulls isAnimationActive={false} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Funnel({ smile }: { smile: Smile }) {
  if (!smile.available || !smile.quality) return null;
  const q = smile.quality;
  const rows: [string, number, boolean][] = [["usable", q.kept, true],
    ...Object.entries(q.rejected_by).map(([k, v]) => [k.replace("_", " "), v, false] as [string, number, boolean])];
  const max = Math.max(...rows.map((r) => r[1]));
  return (
    <div className="card">
      <div className="card-head">
        <span>Quote quality — calls</span>
        <span className="muted">{q.kept.toLocaleString()} / {q.total.toLocaleString()} usable ({q.kept_pct.toFixed(0)}%)</span>
      </div>
      <div className="funnel">
        {rows.map(([name, n, kept]) => (
          <div className="row" key={name}>
            <span style={{ color: kept ? "var(--ink)" : "var(--ink-2)" }}>{name}</span>
            <div className={`bar ${kept ? "kept" : ""}`} style={{ width: `${(100 * n) / max}%` }} />
            <span className="n mono">{n.toLocaleString()}</span>
          </div>
        ))}
      </div>
      <p className="muted" style={{ marginBottom: 4 }}>
        {smile.n_solved?.toLocaleString()} implied vols across {smile.n_expiries} expiries. Forward and discount implied
        from put-call parity per expiry; no rate or dividend input.
      </p>
    </div>
  );
}
