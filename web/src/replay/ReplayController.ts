export const SPEEDS = [0.25, 0.5, 1, 2, 4] as const;

export function clampFrame(frame: number, frameCount: number): number {
  if (frameCount < 1) return 0;
  return Math.min(frameCount - 1, Math.max(0, Math.round(frame)));
}

export function nextFrame(frame: number, frameCount: number): number {
  return frame >= frameCount - 1 ? 0 : frame + 1;
}

export function nearestDiagnostic<T extends { frame: number }>(
  points: T[],
  frame: number,
): T | undefined {
  return points.reduce<T | undefined>(
    (best, point) => (point.frame <= frame && (!best || point.frame > best.frame) ? point : best),
    undefined,
  );
}
