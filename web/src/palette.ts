// Read the design tokens off :root so charts follow the same light/dark switch
// as the rest of the page. Recharts takes color strings, not CSS variables.
import { useEffect, useState } from "react";

export interface Palette {
  s1: string; s2: string; s3: string; s4: string;
  ink: string; ink2: string; muted: string; line: string; surface: string;
}

function read(): Palette {
  const cs = getComputedStyle(document.documentElement);
  const v = (n: string) => cs.getPropertyValue(n).trim();
  return {
    s1: v("--s1"), s2: v("--s2"), s3: v("--s3"), s4: v("--s4"),
    ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"), line: v("--line"), surface: v("--surface"),
  };
}

export function usePalette(): Palette {
  const [p, setP] = useState<Palette>(read);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const on = () => setP(read());
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);
  return p;
}
