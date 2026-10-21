import { readFileSync, statSync } from "node:fs";
import { describe, expect, it } from "vitest";
import type { Robustness, RunIndex, Scene } from "./data";

// Contract between export_web.py and the viewer. If the exporter changes its
// binary layout, these fail before a broken deploy does.
const root = new URL("../public/data/", import.meta.url);
const read = <T>(name: string) => JSON.parse(readFileSync(new URL(name, root), "utf-8")) as T;

const scene = read<Scene>("scene.json");
const index = read<RunIndex>("runs.json");

describe("exported rollouts", () => {
  it("describes the Ant model", () => {
    expect(scene.bodies).toBe(13);
    expect(scene.dt).toBeCloseTo(0.05);
    expect(scene.geoms.filter((g) => g.foot).map((g) => g.foot).sort()).toEqual(["BL", "BR", "FL", "FR"]);
  });

  it("has the progression and the stress tests", () => {
    const ids = index.runs.map((r) => r.id);
    expect(ids).toContain("random");
    expect(ids).toContain("sac_1000k");
    expect(index.runs.some((r) => r.group === "stress")).toBe(true);
  });

  for (const run of index.runs) {
    it(`${run.id}: binary layout matches the file`, () => {
      const file = new URL(`runs/${run.id}.bin`, root);
      const buf = readFileSync(file);
      const n = run.frames;
      const L = run.layout;
      expect(L.poses.shape).toEqual([n, scene.bodies * 7]);
      expect(L.metrics.shape).toEqual([n, 4]);
      expect(L.contacts.shape).toEqual([n]);
      expect(L.torques.shape).toEqual([n, 8]);
      expect(L.poses.offset % 4).toBe(0);
      expect(L.metrics.offset % 4).toBe(0);
      expect(statSync(file).size).toBe(L.torques.offset + n * 8);

      const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      const poses = new Float32Array(ab, L.poses.offset, n * scene.bodies * 7);
      for (let f = 0; f < n; f += Math.max(1, Math.floor(n / 25))) {
        for (let b = 0; b < scene.bodies; b += 1) {
          const o = f * scene.bodies * 7 + b * 7;
          const q = Math.hypot(poses[o + 3], poses[o + 4], poses[o + 5], poses[o + 6]);
          expect(q).toBeCloseTo(1, 3);
        }
      }
      const contacts = new Uint8Array(ab, L.contacts.offset, n);
      expect(Math.max(...contacts)).toBeLessThanOrEqual(15);
      expect(run.eval.episodes).toBe(index.seeds.length);
    });
  }

  it("robustness grid is complete", () => {
    const grid = read<Robustness>("robustness.json");
    for (const p of Object.values(grid.policies)) {
      expect(p.cells).toHaveLength(grid.mass.length);
      for (const row of p.cells) expect(row).toHaveLength(grid.friction.length);
    }
  });
});
