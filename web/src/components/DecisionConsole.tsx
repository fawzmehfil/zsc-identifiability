import type { DiagnosticPoint, ReplayFrame, ReplayManifest } from "../replay/types";

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

export function DecisionConsole({
  manifest,
  diagnostic,
  frame,
}: {
  manifest: ReplayManifest;
  diagnostic?: DiagnosticPoint;
  frame: ReplayFrame;
}) {
  const after = frame.phase === "after";
  const commitmentReached = manifest.commitment.frame !== null && frame.frame >= manifest.commitment.frame;
  const left = diagnostic?.probability_left ?? 0.5;
  const right = diagnostic?.probability_right ?? 0.5;
  return (
    <aside className="decision-console">
      <div className="console-kicker"><span className="status-light" /> Frozen decision decoder</div>
      <h3>Which response fits?</h3>
      <p className="console-explainer">Support inferred from the interaction history available at this checkpoint.</p>
      <div className="response-meter">
        <div className="response-row">
          <span><i className="shape shape--diamond" /> Response A</span>
          <strong>{percent(left)}</strong>
        </div>
        <div className="meter"><i style={{ width: percent(left) }} /></div>
        <div className="response-row">
          <span><i className="shape shape--circle" /> Response B</span>
          <strong>{percent(right)}</strong>
        </div>
        <div className="meter meter--orange"><i style={{ width: percent(right) }} /></div>
      </div>
      <div className="console-readout">
        <span>Recommended</span>
        <strong>{diagnostic?.recommended_response_alias ?? "No preference"}</strong>
      </div>
      <div className="console-readout">
        <span>Estimated residual risk</span>
        <strong>{diagnostic?.estimated_expected_risk.toFixed(3) ?? "0.500"}</strong>
      </div>
      <div className={`commitment-status ${commitmentReached ? "is-committed" : ""}`}>
        <span>{commitmentReached ? "Committed" : "Decision still repairable"}</span>
        <small>{commitmentReached ? (after ? "Later evidence cannot change this score" : "Response boundary reached") : "Evidence still counts"}</small>
      </div>
      {after && <div className="late-label">Too late for this decision</div>}
    </aside>
  );
}
