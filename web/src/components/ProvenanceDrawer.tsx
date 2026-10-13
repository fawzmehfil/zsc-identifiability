import { useState } from "react";
import type { ReplayManifest } from "../replay/types";

export function ProvenanceDrawer({ manifest }: { manifest: ReplayManifest }) {
  const [open, setOpen] = useState(false);
  return (
    <section className={`provenance ${open ? "is-open" : ""}`}>
      <button onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span>Artifact provenance</span><b>{open ? "−" : "+"}</b>
      </button>
      {open && <div className="provenance-body">
        <dl>
          <div><dt>Partner</dt><dd>{manifest.source.partner_id}</dd></div>
          <div><dt>Environment key</dt><dd>{manifest.source.environment_key}</dd></div>
          <div><dt>Ego seat</dt><dd>{manifest.source.ego_seat}</dd></div>
          <div><dt>Evidence policy</dt><dd>{manifest.source.evidence_policy.replaceAll("_", " ")}</dd></div>
          <div><dt>Policy training</dt><dd>None. Frozen inference only.</dd></div>
          <div><dt>Schema</dt><dd>{manifest.schema_version}</dd></div>
        </dl>
        <p>{manifest.selection_note}</p>
        <code>{manifest.source.checkpoint_hashes.partner}</code>
      </div>}
    </section>
  );
}
