import { readFileSync, statSync } from "node:fs";
import { describe, expect, it } from "vitest";

// Contract between export_nav.py and nav.ts.
const root = new URL("../public/data/nav/", import.meta.url);
const nav = JSON.parse(readFileSync(new URL("nav.json", root), "utf-8"));
const BODIES = 13;

type Layout = Record<string, { offset: number; dtype: string; shape: number[] }>;

function checkBin(file: string, layout: Layout): void {
  const n = layout.poses.shape[0];
  expect(layout.poses.shape).toEqual([n, BODIES * 7]);
  for (const key of ["metrics", "contacts", "torques", "commands", "track"]) expect(layout[key].shape[0]).toBe(n);
  expect(layout.track.shape).toEqual([n, 2]);
  for (const key of ["poses", "metrics", "commands", "track"]) expect(layout[key].offset % 4).toBe(0);
  const end = layout.track.offset + n * 2 * 4;
  expect(statSync(new URL(file, root)).size).toBe(end);
}

describe("navigation data", () => {
  it("has both steering clips with evaluation", () => {
    for (const k of ["forward", "steerable"]) {
      const p = nav.steer.pilots[k];
      expect(p.eval.episodes).toBe(nav.seeds.length);
      checkBin(`steer_${k === "steerable" ? "planner" : "forward"}.bin`, p.layout);
    }
    // The point of stage 3: the steerable walker follows commands, the forward walker doesn't.
    expect(nav.steer.pilots.steerable.eval.speed_along_command).toBeGreaterThan(nav.steer.pilots.forward.eval.speed_along_command * 3);
  });

  for (const [name, maze] of Object.entries<any>(nav.mazes)) {
    it(`${name}: walls, path and four pilots`, () => {
      expect(maze.walls.length).toBeGreaterThan(4);
      expect(maze.path.length).toBeGreaterThanOrEqual(2);
      expect(maze.path[0]).toEqual(maze.start);
      expect(maze.path[maze.path.length - 1]).toEqual(maze.goal);
      for (const pilot of ["random", "forward", "beeline", "planner"]) {
        const p = maze.pilots[pilot];
        expect(p.eval.episodes).toBe(nav.seeds.length);
        checkBin(`maze_${name}_${pilot}.bin`, p.layout);
      }
    });
  }
});
