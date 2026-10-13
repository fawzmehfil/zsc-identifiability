import Phaser from "phaser";
import { useEffect, useRef } from "react";
import { ReplayScene } from "../replay/ReplayScene";
import type { ReplayFrame, ReplayManifest } from "../replay/types";

export function ReplayCanvas({ manifest, frame }: { manifest: ReplayManifest; frame: ReplayFrame }) {
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<ReplayScene>();

  useEffect(() => {
    if (!host.current) return;
    const replayScene = new ReplayScene({ manifest, frame });
    scene.current = replayScene;
    const game = new Phaser.Game({
      type: Phaser.CANVAS,
      parent: host.current,
      width: host.current.clientWidth,
      height: host.current.clientHeight,
      backgroundColor: "#0e1219",
      render: { antialias: false, pixelArt: true, roundPixels: true },
      scale: { mode: Phaser.Scale.RESIZE, autoCenter: Phaser.Scale.CENTER_BOTH },
      scene: replayScene,
      audio: { noAudio: true },
    });
    return () => {
      scene.current = undefined;
      game.destroy(true);
    };
  }, [manifest]);

  useEffect(() => scene.current?.setReplayFrame({ manifest, frame }), [manifest, frame]);

  return <div className="replay-canvas" ref={host} aria-label={`Top-down replay at step ${frame.action_step ?? 0}`} />;
}
