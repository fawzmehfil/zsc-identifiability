export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <a className={`brand ${compact ? "brand--compact" : ""}`} href={import.meta.env.BASE_URL} aria-label="BeforeCommit home">
      <span className="brand__mark" aria-hidden="true"><i /><i /></span>
      <span>BEFORE<span>COMMIT</span></span>
    </a>
  );
}

export function SiteHeader() {
  return (
    <header className="site-header">
      <Brand />
      <nav aria-label="Primary navigation">
        <a href={`${import.meta.env.BASE_URL}#replays`}>Replays</a>
        <a href={`${import.meta.env.BASE_URL}#benchmark`}>Benchmark</a>
        <a href="https://github.com/fawzmehfil/zsc-identifiability" target="_blank" rel="noreferrer">Code ↗</a>
      </nav>
    </header>
  );
}
