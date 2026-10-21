export interface GeomSpec {
  name: string;
  body: number;
  type: "sphere" | "capsule";
  size: number[];
  pos: number[];
  quat: number[]; // MuJoCo order: w, x, y, z
  foot: string | null;
}

export interface Scene {
  bodies: number;
  dt: number;
  geoms: GeomSpec[];
  feet: string[];
}

export interface EvalSummary {
  episodes: number;
  return_mean: number;
  return_std: number;
  returns: number[];
  falls: number;
  length_mean: number;
  speed_mean: number;
  distance_mean: number;
  effort_mean: number;
}

export interface RunMeta {
  id: string;
  label: string;
  group: "progression" | "stress";
  physics: { mass: number; friction: number; push_force: number };
  frames: number;
  pushes: number[];
  layout: Record<"poses" | "metrics" | "contacts" | "torques", { offset: number; dtype: string; shape: number[] }>;
  recorded: { return: number; length: number; fell: boolean; distance: number; speed: number };
  gait: { duty?: Record<string, number>; mean_feet_down?: number; airborne?: number };
  eval: EvalSummary;
}

export interface RunIndex {
  seeds: number[];
  runs: RunMeta[];
}

export interface CurveSeries {
  timesteps: number[];
  mean: number[];
  std: number[];
}

export interface Robustness {
  mass: number[];
  friction: number[];
  episodes: number;
  policies: Record<string, { label: string; cells: { return: number; falls: number }[][] }>;
}

/** Decoded rollout. Poses are [frame][body][x y z qw qx qy qz]. */
export interface Rollout {
  meta: RunMeta;
  poses: Float32Array;
  /** [frame][x_velocity, torso_z, reward, cumulative_return] */
  metrics: Float32Array;
  contacts: Uint8Array;
  torques: Int8Array;
  frames: number;
  stride: number;
  /** Navigation clips: commanded heading per frame (rad, MuJoCo xy) and floor track. */
  commands?: Float32Array;
  track?: Float32Array;
}

/** A maze as exported by export_nav.py, in MuJoCo xy meters. */
export interface MazeData {
  grid: string[];
  cell: number;
  goal_radius: number;
  walls: [number, number, number, number][];
  path: [number, number][];
  start: [number, number];
  goal: [number, number];
}

async function json<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return (await res.json()) as T;
}

export const loadScene = () => json<Scene>("data/scene.json");
export const loadIndex = () => json<RunIndex>("data/runs.json");
export const loadCurves = () =>
  json<Record<string, CurveSeries>>("data/curves.json").catch(() => ({}) as Record<string, CurveSeries>);
export const loadRobustness = () => json<Robustness>("data/robustness.json").catch(() => null);

const cache = new Map<string, Promise<Rollout>>();

export function loadRollout(meta: RunMeta, bodies: number): Promise<Rollout> {
  let hit = cache.get(meta.id);
  if (!hit) {
    hit = fetch(`data/runs/${meta.id}.bin`)
      .then((r) => {
        if (!r.ok) throw new Error(`run ${meta.id}: ${r.status}`);
        return r.arrayBuffer();
      })
      .then((buf) => {
        const L = meta.layout;
        const n = meta.frames;
        return {
          meta,
          poses: new Float32Array(buf, L.poses.offset, n * bodies * 7),
          metrics: new Float32Array(buf, L.metrics.offset, n * 4),
          contacts: new Uint8Array(buf, L.contacts.offset, n),
          torques: new Int8Array(buf, L.torques.offset, n * 8),
          frames: n,
          stride: bodies * 7,
        };
      });
    cache.set(meta.id, hit);
  }
  return hit;
}
