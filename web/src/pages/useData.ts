import { useEffect, useState } from "react";
import { loadBenchmark, loadCatalog, loadReplay } from "../replay/ReplaySource";
import type { BenchmarkSummary, Catalog, ReplayBundle } from "../replay/types";

type AsyncState<T> = { data?: T; error?: string };

function useAsync<T>(loader: () => Promise<T>, key: string): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({});
  useEffect(() => {
    let active = true;
    setState({});
    loader().then((data) => active && setState({ data })).catch((error: unknown) => {
      if (active) setState({ error: error instanceof Error ? error.message : String(error) });
    });
    return () => { active = false; };
  }, [key]);
  return state;
}

export const useCatalog = () => useAsync<Catalog>(loadCatalog, "catalog");
export const useBenchmark = () => useAsync<BenchmarkSummary>(loadBenchmark, "benchmark");
export const useReplay = (id: string) => useAsync<ReplayBundle>(() => loadReplay(id), id);
