import type { DiagnosticPoint, ReplayEvent, ReplayManifest } from "../replay/types";

export function CommitmentTimeline({
  manifest,
  frame,
  diagnostics,
  events,
  onSeek,
}: {
  manifest: ReplayManifest;
  frame: number;
  diagnostics: DiagnosticPoint[];
  events: ReplayEvent[];
  onSeek: (frame: number) => void;
}) {
  const maximum = manifest.frame_count - 1;
  const commit = manifest.commitment.frame;
  const percent = (value: number) => `${(value / maximum) * 100}%`;
  const deliveries = events.filter((event) => event.kind === "delivery");
  return (
    <div className="timeline-wrap">
      <div className="timeline-labels">
        <span>Evidence window</span>
        <span>{commit === null ? "No commitment observed" : `Commit at step ${manifest.commitment.step}`}</span>
        <span>Outcome unfolds</span>
      </div>
      <div className="timeline-track">
        {commit !== null && <i className="timeline-after" style={{ left: percent(commit) }} />}
        {diagnostics.filter((point) => typeof point.prefix === "number" && point.frame > 0).map((point) => (
          <button key={point.key} className="timeline-tick" style={{ left: percent(point.frame) }} onClick={() => onSeek(point.frame)} aria-label={`Seek to decoder checkpoint ${point.frame}`} />
        ))}
        {deliveries.map((event, index) => (
          <button key={`${event.frame}-${index}`} className="timeline-delivery" style={{ left: percent(event.frame) }} onClick={() => onSeek(event.frame)} aria-label={`Seek to delivery at step ${event.step}`} />
        ))}
        {commit !== null && <button className="timeline-commit" style={{ left: percent(commit) }} onClick={() => onSeek(commit)} aria-label="Seek to commitment"><span>COMMIT</span></button>}
        <i className="timeline-playhead" style={{ left: percent(frame) }} />
      </div>
      <input
        className="timeline-input"
        type="range"
        min="0"
        max={maximum}
        value={frame}
        onChange={(event) => onSeek(Number(event.target.value))}
        aria-label="Replay step"
      />
    </div>
  );
}
