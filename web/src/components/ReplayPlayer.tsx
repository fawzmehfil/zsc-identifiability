import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { clampFrame, nearestDiagnostic, nextFrame, SPEEDS } from "../replay/ReplayController";
import type { ReplayBundle } from "../replay/types";
import { CommitmentTimeline } from "./CommitmentTimeline";
import { DecisionConsole } from "./DecisionConsole";

const ReplayCanvas = lazy(() =>
  import("./ReplayCanvas").then((module) => ({ default: module.ReplayCanvas })),
);

export function ReplayPlayer({
  replay,
  presentation = false,
  initialPlaying = false,
  initialFrame = 0,
  controlledFrame,
  onControlledFrame,
}: {
  replay: ReplayBundle;
  presentation?: boolean;
  initialPlaying?: boolean;
  initialFrame?: number;
  controlledFrame?: number;
  onControlledFrame?: (frame: number) => void;
}) {
  const [internalFrame, setInternalFrame] = useState(() =>
    clampFrame(initialFrame, replay.frames.length),
  );
  const [playing, setPlaying] = useState(initialPlaying);
  const [speed, setSpeed] = useState<(typeof SPEEDS)[number]>(1);
  const frameIndex = controlledFrame ?? internalFrame;
  const setFrame = (value: number) => {
    const next = clampFrame(value, replay.frames.length);
    if (onControlledFrame) onControlledFrame(next);
    else setInternalFrame(next);
  };
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setFrame(nextFrame(frameIndex, replay.frames.length)), 180 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, frameIndex, replay.frames.length]);
  const frame = replay.frames[frameIndex] ?? replay.frames[0];
  const diagnostic = useMemo(
    () => nearestDiagnostic(replay.diagnostics, frameIndex),
    [replay.diagnostics, frameIndex],
  );
  const firstDelivery = replay.manifest.summary.first_delivery_step;
  return (
    <section className={`replay-machine ${presentation ? "replay-machine--presentation" : ""}`}>
      <div className="machine-topbar">
        <div><span className="machine-signal" /> Official frozen replay</div>
        <div className="machine-meta">
          <span>{replay.manifest.source.layout_id.replaceAll("_", " ")}</span>
          <span>Step {String(frame.action_step ?? 0).padStart(3, "0")} / {replay.manifest.summary.steps_elapsed}</span>
          <span className={frame.phase === "after" ? "phase-after" : "phase-before"}>{frame.phase === "after" ? "After commitment" : frame.phase === "commitment" ? "Committing" : "Evidence window"}</span>
        </div>
      </div>
      <div className="machine-grid">
        <div className="world-pane">
          <Suspense fallback={<div className="canvas-loading">Preparing exact replay</div>}>
            <ReplayCanvas manifest={replay.manifest} frame={frame} />
          </Suspense>
          <div className="agent-legend">
            <span><i className="shape shape--diamond" /> Ego agent</span>
            <span><i className="shape shape--circle" /> Unfamiliar partner</span>
          </div>
          {frame.phase === "commitment" && <div className="commit-flash">Committed</div>}
          {frame.sparse_reward > 0 && <div className="reward-flash">+{frame.sparse_reward} delivery</div>}
        </div>
        <DecisionConsole manifest={replay.manifest} diagnostic={diagnostic} frame={frame} />
      </div>
      <CommitmentTimeline manifest={replay.manifest} frame={frameIndex} diagnostics={replay.diagnostics} events={replay.events} onSeek={setFrame} />
      <div className="machine-controls">
        <button className="control-main" onClick={() => setPlaying((value) => !value)}>{playing ? "Ⅱ Pause" : "▶ Play"}</button>
        <button onClick={() => setFrame(frameIndex - 1)} aria-label="Previous frame">‹</button>
        <button onClick={() => setFrame(frameIndex + 1)} aria-label="Next frame">›</button>
        <div className="speed-controls" aria-label="Playback speed">
          {SPEEDS.map((value) => <button key={value} className={speed === value ? "is-active" : ""} onClick={() => setSpeed(value)}>{value}×</button>)}
        </div>
        <div className="run-stats">
          <span>Return <b>{replay.manifest.summary.sparse_return.toFixed(0)}</b></span>
          <span>First delivery <b>{firstDelivery === null ? "none" : `step ${firstDelivery}`}</b></span>
        </div>
      </div>
    </section>
  );
}
