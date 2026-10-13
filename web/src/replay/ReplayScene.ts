import Phaser from "phaser";
import { drawCook, drawObject, drawTile, PALETTE } from "./kitchenPixelArt";
import type { ReplayFrame, ReplayManifest } from "./types";

export interface ReplaySceneData {
  manifest: ReplayManifest;
  frame: ReplayFrame;
}

export class ReplayScene extends Phaser.Scene {
  private dataState!: ReplaySceneData;
  private world!: Phaser.GameObjects.Graphics;

  constructor(data: ReplaySceneData) {
    super("ReplayScene");
    this.dataState = data;
  }

  create(): void {
    this.cameras.main.setBackgroundColor(PALETTE.ink);
    this.world = this.add.graphics();
    this.renderWorld();
    this.scale.on("resize", () => this.renderWorld());
  }

  setReplayFrame(data: ReplaySceneData): void {
    this.dataState = data;
    if (this.world) this.renderWorld();
  }

  private renderWorld(): void {
    const { manifest, frame } = this.dataState;
    const rows = manifest.terrain.length;
    const columns = manifest.terrain[0].length;
    const tile = Math.floor(Math.min(this.scale.width / columns, this.scale.height / rows));
    const offsetX = Math.floor((this.scale.width - columns * tile) / 2);
    const offsetY = Math.floor((this.scale.height - rows * tile) / 2);
    this.world.clear();
    this.world.fillStyle(PALETTE.ink).fillRect(0, 0, this.scale.width, this.scale.height);
    manifest.terrain.forEach((row, y) => row.forEach((code, x) => {
      drawTile(this.world, code, offsetX + x * tile, offsetY + y * tile, tile);
    }));
    frame.state.objects.forEach((object) => drawObject(
      this.world,
      offsetX + object.x * tile,
      offsetY + object.y * tile,
      tile,
      object.item,
    ));
    [...frame.state.players]
      .sort((a, b) => a.y - b.y)
      .forEach((player) => drawCook(
        this.world,
        offsetX + player.x * tile,
        offsetY + player.y * tile,
        tile,
        player.role === "ego" ? PALETTE.ego : PALETTE.partner,
        player.facing,
        player.role === "ego",
        player.held,
      ));
    if (frame.phase === "commitment") {
      this.world.lineStyle(Math.max(3, tile * 0.06), PALETTE.coral, 1);
      manifest.terrain.forEach((row, y) => row.forEach((code, x) => {
        if (code === "P") this.world.strokeRect(offsetX + x * tile + 2, offsetY + y * tile + 2, tile - 4, tile - 4);
      }));
    }
  }
}
