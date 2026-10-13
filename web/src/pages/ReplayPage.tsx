import { SiteHeader } from "../components/Brand";
import { ProvenanceDrawer } from "../components/ProvenanceDrawer";
import { ReplayPlayer } from "../components/ReplayPlayer";
import { comparePath, presentPath } from "../paths";
import { Loading } from "./Loading";
import { useReplay } from "./useData";

export function ReplayPage({ id }: { id: string }) {
  const replay = useReplay(id);
  if (replay.error) return <Loading error={replay.error} />;
  if (!replay.data) return <Loading />;
  const data = replay.data;
  const counterpart = id === "clear-before-commitment-v1" ? "answer-arrives-late-v1" : "clear-before-commitment-v1";
  return (
    <>
      <SiteHeader />
      <main className="detail-page">
        <div className="detail-heading">
          <div><span className="section-label">Curated diagnostic replay</span><h1>{data.manifest.title}</h1><p>{data.manifest.caption}</p></div>
          <div className="detail-actions"><a className="button button--quiet" href={comparePath(id, counterpart)}>Compare</a><a className="button button--primary" href={presentPath(id)}>Presentation mode</a></div>
        </div>
        <ReplayPlayer replay={data} />
        <div className="replay-interpretation">
          <article><span>At commitment</span><strong>{data.manifest.summary.aggregate_dri_primary === null ? "n/a" : `${Math.round(data.manifest.summary.aggregate_dri_primary * 100)}%`}</strong><p>of prior response-selection risk removed across this partner pair’s frozen history distribution.</p></article>
          <article><span>Recorded outcome</span><strong>{data.manifest.summary.sparse_return.toFixed(0)}</strong><p>sparse return. Outcome and identifiability are deliberately shown as separate measurements.</p></article>
          <article><span>Selected-response loss</span><strong>{data.manifest.summary.selected_response_loss_at_commitment.toFixed(3)}</strong><p>for this recorded partner mode at the frozen precommitment diagnostic.</p></article>
        </div>
        <ProvenanceDrawer manifest={data.manifest} />
      </main>
    </>
  );
}
