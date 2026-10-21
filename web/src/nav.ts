import type { MazeData, Rollout, RunMeta, Scene } from "./data";
import { AntViewer, readTheme } from "./viewer";

/** Stages 3 and 4: steering in the open, and the maze. Data from export_nav.py. */

interface Layout {
  offset: number;
  dtype: string;
  shape: number[];
}

interface Clip {
  layout: Record<"poses" | "metrics" | "contacts" | "torques" | "commands" | "track", Layout>;
  frames?: number;
}

interface MazeEval {
  episodes: number;
  success: number;
  seconds: number | null;
  falls: number;
  closest_m: number;
  wall_contact_s: number;
  recoveries: number;
}

interface NavJson {
  walker: string;
  seeds: number[];
  steer: {
    schedule_deg: number[];
    seconds_each: number;
    pilots: Record<"forward" | "steerable", Clip & { frames: number; eval: { episodes: number; speed_along_command: number; sideways_drift: number; falls: number } }>;
  };
  mazes: Record<string, MazeData & { pilots: Record<string, Clip & { clip: { seed: number; success: boolean; seconds: number; frames: number }; eval: MazeEval }> }>;
}

const FPS = 20;
const PILOTS: { id: string; label: string; detail: string }[] = [
  { id: "random", label: "Random torques", detail: "stage 1" },
  { id: "forward", label: "Forward walker", detail: "stage 2, can't steer" },
  { id: "beeline", label: "Steerable, straight at the goal", detail: "stage 3, no planner" },
  { id: "planner", label: "Steerable + planner", detail: "stage 4" },
];
const PILOT_COLOR: Record<string, string> = { random: "#827c70", forward: "var(--sac)", beeline: "var(--legacy)", planner: "var(--ppo)" };

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const css = (v: string) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

async function fetchClip(file: string, clip: Clip, bodies: number): Promise<Rollout> {
  const res = await fetch(`data/nav/${file}`);
  if (!res.ok) throw new Error(`${file}: ${res.status}`);
  const buf = await res.arrayBuffer();
  const L = clip.layout;
  const n = L.poses.shape[0];
  const meta = { id: file, label: file, pushes: [], recorded: { fell: false } } as unknown as RunMeta;
  return {
    meta,
    poses: new Float32Array(buf, L.poses.offset, n * bodies * 7),
    metrics: new Float32Array(buf, L.metrics.offset, n * 4),
    contacts: new Uint8Array(buf, L.contacts.offset, n),
    torques: new Int8Array(buf, L.torques.offset, n * 8),
    commands: new Float32Array(buf, L.commands.offset, n),
    track: new Float32Array(buf, L.track.offset, n * 2),
    frames: n,
    stride: bodies * 7,
  };
}

function setPlay(btn: HTMLElement, playing: boolean): void {
  btn.classList.toggle("paused", !playing);
  btn.setAttribute("aria-label", playing ? "Pause" : "Play");
}

export interface NavSection {
  tick(dt: number): void;
  resize(): void;
  applyTheme(): void;
}

export async function initNav(scene: Scene): Promise<NavSection | null> {
  let nav: NavJson;
  try {
    const res = await fetch("data/nav/nav.json");
    if (!res.ok) throw new Error(String(res.status));
    nav = (await res.json()) as NavJson;
  } catch {
    $("steer").hidden = true;
    $("maze").hidden = true;
    return null;
  }
  const theme = readTheme();
  const visible = new Set<string>();
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) e.isIntersecting ? visible.add(e.target.id) : visible.delete(e.target.id);
  });
  io.observe($("steer"));
  io.observe($("maze"));

  // ---------------------------------------------------------------- steering
  const steerA = new AntViewer($<HTMLCanvasElement>("steerA"), scene, theme);
  const steerB = new AntViewer($<HTMLCanvasElement>("steerB"), scene, theme);
  const [clipA, clipB] = await Promise.all([
    fetchClip("steer_forward.bin", nav.steer.pilots.forward, scene.bodies),
    fetchClip("steer_planner.bin", nav.steer.pilots.steerable, scene.bodies),
  ]);
  steerA.setRollout(clipA);
  steerB.setRollout(clipB);
  const steerLen = Math.max(clipA.frames, clipB.frames);
  let steerFrame = 0;
  let steerPlaying = true;
  const sf = nav.steer.pilots.forward.eval;
  const ss = nav.steer.pilots.steerable.eval;
  const steerRow = (name: string, e: typeof sf) =>
    `<tr><td>${name}</td><td>${e.speed_along_command.toFixed(2)} m/s</td><td>${e.sideways_drift.toFixed(2)} m/s</td><td>${e.falls}/${e.episodes}</td></tr>`;
  $("steerStats").innerHTML = `<div class="table-wrap"><table class="results"><thead><tr><th>Walker</th><th>Speed along command</th><th>Sideways drift</th><th>Falls</th></tr></thead>
    <tbody>${steerRow("Stage 2 (forward only)", sf)}${steerRow("Stage 3 (steerable)", ss)}</tbody></table></div>
    <p class="hint">${ss.episodes} episodes, six random headings each, held for 7 s.</p>`;
  $<HTMLInputElement>("steerScrub").max = String(steerLen - 1);
  $<HTMLInputElement>("steerScrub").addEventListener("input", (e) => (steerFrame = Number((e.target as HTMLInputElement).value)));
  $("steerPlay").addEventListener("click", () => {
    steerPlaying = !steerPlaying;
    setPlay($("steerPlay"), steerPlaying);
  });

  // ---------------------------------------------------------------- maze
  const mazeView = new AntViewer($<HTMLCanvasElement>("mazeCanvas"), scene, theme);
  const mazeNames = Object.keys(nav.mazes);
  let mazeName = mazeNames[0];
  let pilot = "planner";
  let clip: Rollout | null = null;
  let mazeFrame = 0;
  let mazePlaying = true;
  let mazeSpeed = 2;
  let holdUntil = 0;
  let token = 0;
  const mapCanvas = $<HTMLCanvasElement>("mazeMap");
  const mapCtx = mapCanvas.getContext("2d")!;

  function drawMap(): void {
    const m = nav.mazes[mazeName];
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = mapCanvas.clientWidth;
    const h = mapCanvas.clientHeight;
    if (!w) return;
    mapCanvas.width = Math.round(w * dpr);
    mapCanvas.height = Math.round(h * dpr);
    mapCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    mapCtx.clearRect(0, 0, w, h);
    const rows = m.grid.length;
    const cols = m.grid[0].length;
    const s = Math.min(w / (cols * m.cell), h / (rows * m.cell));
    const ox = w / 2;
    const oy = h / 2;
    const X = (x: number) => ox + x * s;
    const Y = (y: number) => oy - y * s;
    mapCtx.fillStyle = css("--line-2");
    for (const [x, y, hx, hy] of m.walls) mapCtx.fillRect(X(x - hx), Y(y + hy), hx * 2 * s, hy * 2 * s);
    mapCtx.setLineDash([5, 4]);
    mapCtx.strokeStyle = css("--muted");
    mapCtx.lineWidth = 1.5;
    mapCtx.beginPath();
    m.path.forEach(([x, y], i) => (i ? mapCtx.lineTo(X(x), Y(y)) : mapCtx.moveTo(X(x), Y(y))));
    mapCtx.stroke();
    mapCtx.setLineDash([]);
    mapCtx.fillStyle = css("--ppo");
    mapCtx.beginPath();
    mapCtx.arc(X(m.goal[0]), Y(m.goal[1]), m.goal_radius * s, 0, Math.PI * 2);
    mapCtx.globalAlpha = 0.25;
    mapCtx.fill();
    mapCtx.globalAlpha = 1;
    if (clip?.track) {
      const upto = Math.min(clip.frames - 1, Math.floor(mazeFrame));
      const color = PILOT_COLOR[pilot].startsWith("var(") ? css(PILOT_COLOR[pilot].slice(4, -1)) : PILOT_COLOR[pilot];
      mapCtx.strokeStyle = color;
      mapCtx.lineWidth = 2.5;
      mapCtx.beginPath();
      for (let i = 0; i <= upto; i += 1) {
        const x = X(clip.track[i * 2]);
        const y = Y(clip.track[i * 2 + 1]);
        i ? mapCtx.lineTo(x, y) : mapCtx.moveTo(x, y);
      }
      mapCtx.stroke();
      mapCtx.fillStyle = color;
      mapCtx.beginPath();
      mapCtx.arc(X(clip.track[upto * 2]), Y(clip.track[upto * 2 + 1]), 4.5, 0, Math.PI * 2);
      mapCtx.fill();
    }
  }

  function renderPilots(): void {
    const m = nav.mazes[mazeName];
    $("mazePilots").innerHTML = PILOTS.map((p) => {
      const e = m.pilots[p.id].eval;
      return `<button class="pilot-row" data-pilot="${p.id}" aria-pressed="${p.id === pilot}" style="--swatch:${PILOT_COLOR[p.id]}">
        <span class="lbl">${p.label}</span><span class="val">${e.success}/${e.episodes}</span>
        <span class="sub">${p.detail}${e.seconds ? `, ${e.seconds.toFixed(0)} s` : ""}</span></button>`;
    }).join("");
    $("mazePick").innerHTML = mazeNames
      .map((n) => `<button data-maze="${n}" class="${n === mazeName ? "on" : ""}">${n[0].toUpperCase() + n.slice(1)}</button>`)
      .join("");
  }

  async function loadMaze(): Promise<void> {
    const t = ++token;
    const m = nav.mazes[mazeName];
    const data = m.pilots[pilot];
    const r = await fetchClip(`maze_${mazeName}_${pilot}.bin`, data, scene.bodies);
    if (t !== token) return;
    clip = r;
    mazeView.setMaze(m, css("--ppo"));
    mazeView.setRollout(r);
    mazeFrame = 0;
    holdUntil = 0;
    $<HTMLInputElement>("mazeScrub").max = String(r.frames - 1);
    const label = PILOTS.find((p) => p.id === pilot)!;
    $("mazeName").textContent = label.label;
    $("mazeName").style.setProperty("--swatch", PILOT_COLOR[pilot]);
    $("mazeSub").textContent = `${data.clip.success ? `Goal in ${data.clip.seconds.toFixed(1)} s` : "No goal"}, ${data.eval.success}/${data.eval.episodes} over all seeds`;
    renderPilots();
  }

  function renderTable(): void {
    const head = `<thead><tr><th>Pilot</th>${mazeNames.map((n) => `<th colspan="3">${n[0].toUpperCase() + n.slice(1)}</th>`).join("")}</tr>
      <tr><th></th>${mazeNames.map(() => "<th>Reached goal</th><th>Time</th><th>Falls</th>").join("")}</tr></thead>`;
    const rows = PILOTS.map((p) => {
      const cells = mazeNames
        .map((n) => {
          const e = nav.mazes[n].pilots[p.id].eval;
          return `<td><span class="cellbar"><i style="width:${(e.success / e.episodes) * 100}%"></i></span>${e.success}/${e.episodes}</td><td>${e.seconds ? `${e.seconds.toFixed(1)} s` : "–"}</td><td>${e.falls}</td>`;
        })
        .join("");
      return `<tr><td><i class="sw" style="background:${PILOT_COLOR[p.id]}"></i>${p.label}</td>${cells}</tr>`;
    }).join("");
    $("mazeTable").innerHTML = `${head}<tbody>${rows}</tbody>`;
  }

  $("mazePilots").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>("[data-pilot]");
    if (b?.dataset.pilot) {
      pilot = b.dataset.pilot;
      void loadMaze();
    }
  });
  $("mazePick").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLElement>("[data-maze]");
    if (b?.dataset.maze) {
      mazeName = b.dataset.maze;
      void loadMaze();
    }
  });
  $("mazeView").addEventListener("click", () => {
    mazeView.overheadOn = !mazeView.overheadOn;
    $("mazeView").textContent = mazeView.overheadOn ? "Follow the robot" : "Overhead view";
  });
  $<HTMLInputElement>("mazeScrub").addEventListener("input", (e) => {
    mazeFrame = Number((e.target as HTMLInputElement).value);
    holdUntil = 0;
  });
  $("mazePlay").addEventListener("click", () => {
    mazePlaying = !mazePlaying;
    setPlay($("mazePlay"), mazePlaying);
  });
  document.querySelectorAll<HTMLButtonElement>("[data-mspeed]").forEach((b) =>
    b.addEventListener("click", () => {
      mazeSpeed = Number(b.dataset.mspeed);
      document.querySelectorAll("[data-mspeed]").forEach((x) => x.classList.toggle("on", x === b));
    }),
  );
  document.querySelectorAll("[data-mspeed]").forEach((x) => x.classList.toggle("on", x.getAttribute("data-mspeed") === String(mazeSpeed)));
  mazeView.overheadOn = true;
  $("mazeView").textContent = "Follow the robot";
  renderTable();
  const solved = mazeNames.reduce((s, n) => s + nav.mazes[n].pilots.planner.eval.success, 0);
  const runs = mazeNames.reduce((s, n) => s + nav.mazes[n].pilots.planner.eval.episodes, 0);
  const others = mazeNames.reduce((s, n) => s + ["random", "forward", "beeline"].reduce((t, p) => t + nav.mazes[n].pilots[p].eval.success, 0), 0);
  $("lede").textContent += ` Walker plus planner reaches the goal in ${solved} of ${runs} maze runs; the simpler pilots in ${others}.`;
  await loadMaze();

  let mapTimer = 0;
  return {
    tick(dt: number) {
      if (visible.has("steer")) {
        if (steerPlaying) steerFrame = (steerFrame + dt * FPS) % steerLen;
        steerA.setFrame(Math.min(clipA.frames - 1, steerFrame));
        steerB.setFrame(Math.min(clipB.frames - 1, steerFrame));
        steerA.render(dt);
        steerB.render(dt);
        const i = Math.floor(steerFrame);
        const deg = ((((clipB.commands?.[Math.min(i, clipB.frames - 1)] ?? 0) * 180) / Math.PI) % 360 + 360) % 360;
        $("steerClock").textContent = `${(steerFrame / FPS).toFixed(1).padStart(4, "0")} s`;
        $("steerCmd").textContent = `command: ${Math.round(deg)}°`;
        if (document.activeElement !== $("steerScrub")) $<HTMLInputElement>("steerScrub").value = String(steerFrame);
      }
      if (visible.has("maze") && clip) {
        const n = clip.frames;
        if (mazePlaying) {
          if (mazeFrame >= n - 1) {
            if (!holdUntil) holdUntil = performance.now() + 1800;
            else if (performance.now() > holdUntil) {
              mazeFrame = 0;
              holdUntil = 0;
            }
          } else mazeFrame = Math.min(n - 1, mazeFrame + dt * FPS * mazeSpeed);
        }
        mazeView.setFrame(mazeFrame);
        mazeView.render(dt);
        const done = mazeFrame >= n - 1;
        const data = nav.mazes[mazeName].pilots[pilot];
        $("mazeOutcome").hidden = !done;
        $("mazeOutcome").textContent = data.clip.success ? "Reached the goal" : data.clip.frames >= 2400 ? "Out of time" : "Fell over";
        $("mazeOutcome").dataset.ok = String(data.clip.success);
        $("mazeClock").textContent = `${(mazeFrame / FPS).toFixed(1).padStart(4, "0")} s`;
        if (document.activeElement !== $("mazeScrub")) $<HTMLInputElement>("mazeScrub").value = String(mazeFrame);
        mapTimer -= dt;
        if (mapTimer <= 0) {
          mapTimer = 0.1;
          drawMap();
        }
      }
    },
    resize() {
      steerA.resize();
      steerB.resize();
      mazeView.resize();
      drawMap();
    },
    applyTheme() {
      const t = readTheme();
      steerA.applyTheme(t);
      steerB.applyTheme(t);
      mazeView.applyTheme(t);
      if (clip) mazeView.setMaze(nav.mazes[mazeName], css("--ppo"));
      drawMap();
    },
  };
}
