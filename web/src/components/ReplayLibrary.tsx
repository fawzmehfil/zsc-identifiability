import type { CatalogItem } from "../replay/types";
import { replayPath } from "../paths";

const categoryLabel: Record<string, string> = {
  clear_before_commitment: "Clear early",
  late_evidence: "Evidence arrives late",
  evidence_outcome_gap: "Evidence ≠ outcome",
  robust_success: "Success under ambiguity",
  intervention_control: "Matched control",
  intervention_treatment: "Natural intervention",
};

export function ReplayLibrary({ items }: { items: CatalogItem[] }) {
  return (
    <section className="replay-library" id="replays">
      <div className="section-heading">
        <div><span className="section-label">Replay library</span><h2>Six ways coordination can look</h2></div>
        <p>Curated examples explain distinct failure modes. They are illustrative, not estimates of how often each case occurs.</p>
      </div>
      <div className="replay-cards">
        {items.map((item, index) => (
          <a className="replay-card" href={replayPath(item.replay_id)} key={item.replay_id}>
            <div className="card-index">0{index + 1}</div>
            <span className="card-category">{categoryLabel[item.category] ?? item.category}</span>
            <h3>{item.title}</h3>
            <p>{item.caption}</p>
            <div className="card-metrics">
              <span>DRI <b>{item.aggregate_dri_primary === null ? "n/a" : item.aggregate_dri_primary.toFixed(2)}</b></span>
              <span>Return <b>{item.sparse_return.toFixed(0)}</b></span>
              <span>Commit <b>{item.commitment_step ?? "none"}</b></span>
            </div>
            <span className="card-link">Inspect replay →</span>
          </a>
        ))}
      </div>
    </section>
  );
}
