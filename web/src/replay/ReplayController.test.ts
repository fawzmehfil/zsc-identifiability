import { describe, expect, it } from "vitest";
import { clampFrame, nearestDiagnostic, nextFrame } from "./ReplayController";

describe("ReplayController", () => {
  it("clamps seeks to valid exact frames", () => {
    expect(clampFrame(-2, 10)).toBe(0);
    expect(clampFrame(4.6, 10)).toBe(5);
    expect(clampFrame(12, 10)).toBe(9);
  });

  it("loops only after displaying the final frame", () => {
    expect(nextFrame(2, 4)).toBe(3);
    expect(nextFrame(3, 4)).toBe(0);
  });

  it("does not leak a future diagnostic", () => {
    const points = [{ frame: 0 }, { frame: 8 }, { frame: 16 }];
    expect(nearestDiagnostic(points, 12)?.frame).toBe(8);
  });
});
