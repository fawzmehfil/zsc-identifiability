const base = import.meta.env.BASE_URL.replace(/\/$/, "");

export const homePath = () => `${base}/` || "/";
export const replayPath = (id: string) => `${base}/replays/${encodeURIComponent(id)}`;
export const presentPath = (id: string) => `${base}/present/${encodeURIComponent(id)}`;
export const comparePath = (left: string, right: string) => `${base}/compare?left=${encodeURIComponent(left)}&right=${encodeURIComponent(right)}`;

export function appPathname(): string {
  const pathname = window.location.pathname;
  return base && pathname.startsWith(base) ? pathname.slice(base.length) || "/" : pathname;
}
