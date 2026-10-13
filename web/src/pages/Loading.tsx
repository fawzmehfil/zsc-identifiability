import { Brand } from "../components/Brand";

export function Loading({ error }: { error?: string }) {
  return (
    <main className="loading-screen">
      <Brand />
      {error ? <><h1>Replay unavailable</h1><p>{error}</p></> : <><div className="loading-bars"><i /><i /><i /></div><p>Loading frozen replay</p></>}
    </main>
  );
}
