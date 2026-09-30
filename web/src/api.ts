// Typed client for app/api.py. One function per endpoint, nothing clever.

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export interface TickerInfo {
  ticker: string;
  watched: boolean;
  days_stored: number;
  cboe_index: boolean;
  vendor_history: boolean;
}

export interface Point {
  date: string;
  atm_iv_30d: number | null;
  atm_iv_90d: number | null;
  rv21_trailing: number | null;
  rv21_forward: number | null;
  skew30: number | null;
  cboe_iv30: number | null;
  coverage: number | null;
  close: number | null;
}

export interface Summary {
  iv30: number | null; iv30_date: string | null;
  rv21: number | null; rv21_date: string | null;
  gap: number | null; gap_date: string | null;
  skew30: number | null;
  days_implied: number;
  days_total: number;
  cboe_index: boolean;
}

export interface Metrics { ticker: string; summary: Summary; series: Point[] }

export interface Curve { dte: number; expiry: string; points: { k: number; iv: number; strike: number }[] }

export interface Smile {
  ticker: string;
  available: boolean;
  reason?: string;
  source?: "live" | "stored" | "vendor";
  date?: string;
  spot?: number;
  market_state: string;
  n_solved?: number;
  n_expiries?: number;
  quality?: { total: number; kept: number; kept_pct: number; rejected_by: Record<string, number> };
  curves?: Curve[];
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Retries what a waking or redeploying API returns (network error, 502-504; 503 = busy), after 3 s then 6 s.
async function get<T>(path: string, tries = 3): Promise<T> {
  for (let i = 0; ; i++) {
    let r: Response;
    try {
      r = await fetch(BASE + path);
    } catch {
      if (i < tries - 1) { await sleep(3000 * (i + 1)); continue; }
      throw new Error("API unreachable — try again in a minute");
    }
    if ([502, 503, 504].includes(r.status) && i < tries - 1) { await sleep(3000 * (i + 1)); continue; }
    if (!r.ok) {
      let msg = `${r.status}`;
      try { msg = (await r.json()).detail ?? msg; } catch { /* keep status */ }
      throw new Error(msg);
    }
    return r.json();
  }
}

export const api = {
  tickers: () => get<TickerInfo[]>("/api/tickers"),
  metrics: (t: string) => get<Metrics>(`/api/metrics/${encodeURIComponent(t)}`),
  smile: (t: string) => get<Smile>(`/api/smile/${encodeURIComponent(t)}`),
  track: async (t: string) => {
    const r = await fetch(BASE + `/api/watchlist/${encodeURIComponent(t)}`, { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail ?? `${r.status}`);
    return j as { added: boolean; detail: string; watchlist: string[] };
  },
};
