import type { BenchmarkSummary, Catalog, ReplayBundle } from "./types";

const root = `${import.meta.env.BASE_URL}static-api`;

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${root}/${path}`);
  if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
  return (await response.json()) as T;
}

export const loadCatalog = () => getJson<Catalog>("catalog.json");
export const loadBenchmark = () => getJson<BenchmarkSummary>("benchmark-summary.json");

export async function loadReplay(id: string): Promise<ReplayBundle> {
  const base = `replays/${encodeURIComponent(id)}`;
  const [manifest, frames, diagnostics, events] = await Promise.all([
    getJson<ReplayBundle["manifest"]>(`${base}/manifest.json`),
    getJson<ReplayBundle["frames"]>(`${base}/frames.json`),
    getJson<ReplayBundle["diagnostics"]>(`${base}/diagnostics.json`),
    getJson<ReplayBundle["events"]>(`${base}/events.json`),
  ]);
  return { manifest, frames, diagnostics, events };
}
