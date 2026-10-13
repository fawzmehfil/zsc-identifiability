import Phaser from "phaser";
import type { Facing, KitchenItem } from "./types";

export const PALETTE = {
  ink: 0x0e1219,
  floorA: 0xd8c7a5,
  floorB: 0xcbb998,
  grout: 0xa99778,
  counter: 0x465463,
  steel: 0x758696,
  shadow: 0x202b35,
  cream: 0xf3ecd7,
  ego: 0x6574d9,
  partner: 0xd98245,
  cyan: 0x55c7b8,
  amber: 0xe7b84b,
  coral: 0xe45e52,
  tomato: 0xd65f51,
  onion: 0xead69a,
};

export function drawTile(
  graphics: Phaser.GameObjects.Graphics,
  code: string,
  x: number,
  y: number,
  size: number,
): void {
  const checker = (x / size + y / size) % 2 === 0;
  graphics.fillStyle(checker ? PALETTE.floorA : PALETTE.floorB).fillRect(x, y, size, size);
  graphics.lineStyle(1, PALETTE.grout, 0.35).strokeRect(x, y, size, size);
  if (code === " ") return;
  graphics.fillStyle(PALETTE.shadow).fillRect(x + 2, y + size - 8, size - 4, 7);
  graphics.fillStyle(PALETTE.counter).fillRect(x + 2, y + 3, size - 4, size - 10);
  graphics.fillStyle(PALETTE.steel).fillRect(x + 4, y + 5, size - 8, 5);
  if (code === "P") drawPot(graphics, x, y, size);
  if (code === "O" || code === "T") drawDispenser(graphics, x, y, size, code);
  if (code === "D") drawPlate(graphics, x, y, size);
  if (code === "S") drawServe(graphics, x, y, size);
}

function drawPot(g: Phaser.GameObjects.Graphics, x: number, y: number, s: number): void {
  g.fillStyle(PALETTE.ink).fillRect(x + s * 0.21, y + s * 0.28, s * 0.58, s * 0.46);
  g.fillStyle(PALETTE.steel).fillRect(x + s * 0.27, y + s * 0.34, s * 0.46, s * 0.3);
  g.fillStyle(PALETTE.ink).fillRect(x + s * 0.11, y + s * 0.43, s * 0.12, s * 0.12);
  g.fillRect(x + s * 0.77, y + s * 0.43, s * 0.12, s * 0.12);
}

function drawDispenser(
  g: Phaser.GameObjects.Graphics,
  x: number,
  y: number,
  s: number,
  code: string,
): void {
  const color = code === "T" ? PALETTE.tomato : PALETTE.onion;
  g.fillStyle(PALETTE.shadow).fillRect(x + s * 0.23, y + s * 0.24, s * 0.54, s * 0.5);
  g.fillStyle(color).fillCircle(x + s * 0.5, y + s * 0.48, s * 0.18);
  g.fillStyle(0x5e7f4b).fillRect(x + s * 0.47, y + s * 0.27, s * 0.08, s * 0.1);
}

function drawPlate(g: Phaser.GameObjects.Graphics, x: number, y: number, s: number): void {
  g.fillStyle(PALETTE.cream).fillEllipse(x + s * 0.5, y + s * 0.5, s * 0.5, s * 0.28);
  g.lineStyle(2, PALETTE.shadow).strokeEllipse(x + s * 0.5, y + s * 0.5, s * 0.5, s * 0.28);
}

function drawServe(g: Phaser.GameObjects.Graphics, x: number, y: number, s: number): void {
  g.fillStyle(PALETTE.cyan).fillRect(x + s * 0.2, y + s * 0.31, s * 0.6, s * 0.34);
  g.fillStyle(PALETTE.cream).fillRect(x + s * 0.29, y + s * 0.4, s * 0.42, s * 0.07);
}

export function drawCook(
  graphics: Phaser.GameObjects.Graphics,
  x: number,
  y: number,
  size: number,
  color: number,
  facing: Facing,
  isEgo: boolean,
  held: KitchenItem | null,
): void {
  const cx = x + size / 2;
  const cy = y + size / 2;
  graphics.fillStyle(PALETTE.ink, 0.45).fillEllipse(cx + 2, cy + size * 0.3, size * 0.45, size * 0.18);
  graphics.fillStyle(PALETTE.ink).fillRect(cx - size * 0.21, cy - size * 0.05, size * 0.42, size * 0.45);
  graphics.fillStyle(color).fillRect(cx - size * 0.17, cy, size * 0.34, size * 0.36);
  graphics.fillStyle(0xc98f66).fillRect(cx - size * 0.16, cy - size * 0.22, size * 0.32, size * 0.23);
  graphics.fillStyle(PALETTE.cream).fillRect(cx - size * 0.23, cy - size * 0.31, size * 0.46, size * 0.1);
  graphics.fillRect(cx - size * 0.17, cy - size * 0.39, size * 0.34, size * 0.1);
  const eyeX = facing === "east" ? cx + 6 : facing === "west" ? cx - 8 : cx - 7;
  graphics.fillStyle(PALETTE.ink).fillRect(eyeX, cy - size * 0.16, 3, 3);
  if (facing === "north") graphics.fillStyle(color).fillRect(cx - 2, cy - size * 0.35, 5, 4);
  if (isEgo) {
    graphics.lineStyle(3, PALETTE.cyan).strokePoints([
      new Phaser.Geom.Point(cx, y + 1),
      new Phaser.Geom.Point(x + size - 2, cy),
      new Phaser.Geom.Point(cx, y + size - 1),
      new Phaser.Geom.Point(x + 2, cy),
    ], true);
  } else {
    graphics.lineStyle(3, PALETTE.partner).strokeCircle(cx, cy, size * 0.43);
  }
  if (held) drawHeldItem(graphics, cx, y + 3, size, held);
}

function drawHeldItem(
  g: Phaser.GameObjects.Graphics,
  cx: number,
  top: number,
  size: number,
  item: KitchenItem,
): void {
  const color = item.kind === "tomato" ? PALETTE.tomato : item.kind === "dish" ? PALETTE.cream : PALETTE.onion;
  if (item.kind === "soup") {
    g.fillStyle(PALETTE.steel).fillRect(cx - size * 0.14, top - size * 0.05, size * 0.28, size * 0.12);
    g.fillStyle(PALETTE.tomato).fillRect(cx - size * 0.1, top - size * 0.04, size * 0.2, size * 0.05);
  } else {
    g.fillStyle(color).fillCircle(cx, top, size * 0.09);
  }
}

export function drawObject(
  g: Phaser.GameObjects.Graphics,
  x: number,
  y: number,
  size: number,
  item: KitchenItem,
): void {
  if (item.kind === "soup") {
    const fill = item.ready ? PALETTE.cyan : PALETTE.tomato;
    g.fillStyle(PALETTE.ink).fillRect(x + size * 0.22, y + size * 0.26, size * 0.56, size * 0.42);
    g.fillStyle(fill).fillRect(x + size * 0.3, y + size * 0.34, size * 0.4, size * 0.2);
    for (let i = 0; i < item.ingredients.length; i += 1) {
      g.fillStyle(PALETTE.onion).fillRect(x + size * (0.34 + i * 0.1), y + size * 0.38, 4, 4);
    }
    return;
  }
  const color = item.kind === "tomato" ? PALETTE.tomato : item.kind === "dish" ? PALETTE.cream : PALETTE.onion;
  g.fillStyle(PALETTE.ink).fillCircle(x + size / 2 + 1, y + size / 2 + 2, size * 0.15);
  g.fillStyle(color).fillCircle(x + size / 2, y + size / 2, size * 0.12);
}
