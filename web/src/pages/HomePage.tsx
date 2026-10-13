import { BenchmarkCard } from "../components/BenchmarkCard";
import { ReplayLibrary } from "../components/ReplayLibrary";
import { ReplayPlayer } from "../components/ReplayPlayer";
import { SiteHeader } from "../components/Brand";
import { comparePath, replayPath } from "../paths";
import { Loading } from "./Loading";
import { useBenchmark, useCatalog, useReplay } from "./useData";

const featuredId = "clear-before-commitment-v1";

export function HomePage() {
  const catalog = useCatalog();
  const benchmark = useBenchmark();
  const replay = useReplay(featuredId);
  if (catalog.error || benchmark.error || replay.error) return <Loading error={catalog.error ?? benchmark.error ?? replay.error} />;
  if (!catalog.data || !benchmark.data || !replay.data) return <Loading />;
  return (
    <>
      <SiteHeader />
      <main>
        <section className="hero">
          <div className="hero-copy">
            <div className="eyebrow"><span /> Multi-agent evaluation, replayed</div>
            <h1>Can an agent understand its teammate <em>before it is too late?</em></h1>
            <p>BeforeCommit turns a hidden coordination problem into something you can inspect frame by frame. Watch what the ego agent sees, where the team commits, and whether the useful evidence arrived in time.</p>
            <div className="hero-actions">
              <a className="button button--primary" href={replayPath(featuredId)}>Explore the replay</a>
              <a className="button button--quiet" href={comparePath("clear-before-commitment-v1", "answer-arrives-late-v1")}>Compare early vs late</a>
            </div>
          </div>
          <div className="hero-question">
            <span>The benchmark asks one question</span>
            <strong>Did the interaction reveal the right coordination response before the team committed?</strong>
          </div>
        </section>
        <section className="featured-section" id="featured">
          <div className="section-heading section-heading--replay">
            <div><span className="section-label">Featured run · clear before commitment</span><h2>The evidence arrives early enough</h2></div>
            <p>At each registered checkpoint, a frozen decoder estimates which response fits the unfamiliar partner. The coral boundary marks the first successful pot placement.</p>
          </div>
          <ReplayPlayer replay={replay.data} />
          <div className="metric-separation">
            <div><span>One replay shows</span><strong>Decoder support and residual risk for this recorded history</strong></div>
            <div><span>The benchmark reports</span><strong>DRI {replay.data.manifest.summary.aggregate_dri_primary?.toFixed(3)} across the pair’s history distribution</strong></div>
          </div>
        </section>
        <section className="how-it-works">
          <div className="section-heading"><div><span className="section-label">The evaluation loop</span><h2>Observe. Commit. Diagnose.</h2></div></div>
          <div className="steps-grid">
            <article><span>01</span><div className="step-icon step-icon--observe"><i /><i /></div><h3>Observe</h3><p>Two frozen agents coordinate. The ego does not receive the partner’s mode as a label.</p></article>
            <article><span>02</span><div className="step-icon step-icon--commit"><i /></div><h3>Commit</h3><p>The first successful ingredient placement marks the response boundary for this audit.</p></article>
            <article><span>03</span><div className="step-icon step-icon--diagnose"><i /><i /><i /></div><h3>Diagnose</h3><p>Compare decision risk before and after observing history. DRI is the fraction removed before commitment.</p></article>
          </div>
        </section>
        <ReplayLibrary items={catalog.data.items} />
        <BenchmarkCard summary={benchmark.data} />
        <section className="limits">
          <div><span className="section-label">Read this result correctly</span><h2>A diagnostic, not a victory lap.</h2></div>
          <div className="limits-copy">
            <p>DRI complements partner diversity and return. It measures whether response-relevant evidence exists before a named commitment boundary. It does not measure human-like understanding.</p>
            <p>The official audit covered two layouts, fixed partner modes, and an approximate response library. Predictive improvement was modest. The natural interventions did not reliably increase useful information.</p>
          </div>
        </section>
      </main>
      <footer><span>BeforeCommit</span><p>Built from frozen ZSC-Eval checkpoints. No new policy training.</p><a href="https://github.com/fawzmehfil/zsc-identifiability">Methodology and code ↗</a></footer>
    </>
  );
}
