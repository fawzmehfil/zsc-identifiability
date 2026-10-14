import { Brand } from "../components/Brand";
import { ReplayPlayer } from "../components/ReplayPlayer";
import { Loading } from "./Loading";
import { useReplay } from "./useData";

export function PresentPage({ id }: { id: string }) {
  const replay = useReplay(id);
  if (replay.error) return <Loading error={replay.error} />;
  if (!replay.data) return <Loading />;
  const search = new URLSearchParams(window.location.search);
  const initialFrame = Number(search.get("frame") ?? 0);
  const autoplay = search.get("static") !== "1";
  const capture = search.get("capture") === "1";
  return (
    <main className={`present-page ${capture ? "present-page--capture" : ""}`}>
      {!capture && <header><Brand compact /><div><span>{replay.data.manifest.title}</span><strong>{replay.data.manifest.caption}</strong></div><a href={`${import.meta.env.BASE_URL}replays/${id}`}>Exit</a></header>}
      <ReplayPlayer
        replay={replay.data}
        presentation
        initialPlaying={autoplay}
        initialFrame={initialFrame}
        initialSpeed={capture ? 2 : 1}
        forcePlayingIndicator={capture}
      />
    </main>
  );
}
