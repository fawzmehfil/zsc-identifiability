import type { BenchmarkSummary } from "../replay/types";

export function BenchmarkCard({ summary }: { summary: BenchmarkSummary }) {
  return (
    <section className="benchmark-card" id="benchmark">
      <div className="section-label">Frozen benchmark audit</div>
      <div className="benchmark-grid">
        <div>
          <strong>{summary.partner_count}</strong>
          <span>official partners</span>
        </div>
        <div>
          <strong>{summary.method_family_count}</strong>
          <span>method families</span>
        </div>
        <div>
          <strong>{summary.layout_count} × {summary.seats_per_layout}</strong>
          <span>layouts × seats</span>
        </div>
        <div>
          <strong>{summary.confirmation_episode_count.toLocaleString()}</strong>
          <span>fresh episodes</span>
        </div>
      </div>
      <div className="benchmark-finding">
        <div><span>Held-out predictive gain</span><strong>ΔR² {summary.held_out_delta_r2.toFixed(3)}</strong></div>
        <div><span>Adjusted DRI coefficient</span><strong>{summary.adjusted_dri_coefficient.toFixed(3)}</strong><small>95% interval [{summary.coefficient_interval[0].toFixed(3)}, {summary.coefficient_interval[1].toFixed(3)}]</small></div>
        <p>Higher precommitment identifiability predicted lower response-library regret beyond the registered controls. The effect was modest, and neither tested natural intervention reliably raised information.</p>
      </div>
    </section>
  );
}
