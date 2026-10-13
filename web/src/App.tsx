import { appPathname } from "./paths";
import { ComparePage } from "./pages/ComparePage";
import { HomePage } from "./pages/HomePage";
import { PresentPage } from "./pages/PresentPage";
import { ReplayPage } from "./pages/ReplayPage";

export default function App() {
  const path = appPathname();
  const search = new URLSearchParams(window.location.search);
  if (path === "/compare") {
    return <ComparePage leftId={search.get("left") ?? "clear-before-commitment-v1"} rightId={search.get("right") ?? "answer-arrives-late-v1"} />;
  }
  const replay = path.match(/^\/replays\/([^/]+)$/);
  if (replay) return <ReplayPage id={decodeURIComponent(replay[1])} />;
  const present = path.match(/^\/present\/([^/]+)$/);
  if (present) return <PresentPage id={decodeURIComponent(present[1])} />;
  return <HomePage />;
}
