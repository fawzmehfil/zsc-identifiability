import { useState } from "react";
import { Brand } from "../components/Brand";
import { ReplayPlayer } from "../components/ReplayPlayer";
import { Loading } from "./Loading";
import { useReplay } from "./useData";

export function ComparePage({ leftId, rightId }: { leftId: string; rightId: string }) {
  const left = useReplay(leftId);
  const right = useReplay(rightId);
  const [frame, setFrame] = useState(0);
  const error = left.error ?? right.error;
  if (error) return <Loading error={error} />;
  if (!left.data || !right.data) return <Loading />;
  return (
    <main className="compare-page">
      <header className="compare-header"><Brand /><div><span>Synchronized diagnostic</span><strong>Same question. Different evidence timing.</strong></div><a href={import.meta.env.BASE_URL}>Exit compare</a></header>
      <div className="compare-explainer">Both replays share one playhead. The decoder only updates at registered checkpoints, and the coral boundary remains specific to each episode.</div>
      <div className="compare-grid">
        <section><div className="compare-title"><span>Replay A</span><h1>{left.data.manifest.title}</h1></div><ReplayPlayer replay={left.data} presentation controlledFrame={frame} onControlledFrame={setFrame} /></section>
        <section><div className="compare-title"><span>Replay B</span><h1>{right.data.manifest.title}</h1></div><ReplayPlayer replay={right.data} presentation controlledFrame={frame} onControlledFrame={setFrame} /></section>
      </div>
    </main>
  );
}
