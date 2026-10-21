import "@fontsource/newsreader/400.css";
import "@fontsource/newsreader/500.css";
import "@fontsource/newsreader/400-italic.css";
import "@fontsource/inter-tight/400.css";
import "@fontsource/inter-tight/500.css";
import "@fontsource/inter-tight/600.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import "./style.css";

import { heatmap, lines, type Series } from "./charts";
import { loadCurves, loadIndex, loadRobustness, loadRollout, loadScene, type Rollout, type RunMeta } from "./data";
import { GaitView } from "./gait";
import { initNav, type NavSection } from "./nav";
import { AntViewer, readTheme } from "./viewer";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const FPS = 20; // Ant-v4 env step is 0.05 s
const fmt = (v: number) => Math.round(v).toLocaleString();
const k = (v: number) => (v >= 1e6 ? `${Number((v / 1e6).toFixed(2))}M` : `${Math.round(v / 1e3)}k`);

// Theme: follow the OS, allow an override that persists for this viewer.
function initialTheme(): "light" | "dark" {
  try {
    const saved = localStorage.getItem("quadruped.theme");
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* storage blocked */
  }
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
document.documentElement.dataset.theme = initialTheme();

function colorFor(id: string): string {
  if (id.startsWith("sac")) return "var(--sac)";
  if (id === "ppo_legacy") return "var(--legacy)";
  if (id.startsWith("ppo")) return "var(--ppo)";
  return "var(--muted)";
}

function hudHtml(r: Rollout, i: number): string {
  const m = r.metrics;
  const o = i * 4;
  const feet = r.contacts[i];
  const down = [0, 1, 2, 3].map((b) => (feet & (1 << b) ? "on" : "")).map((c, j) => `<i class="${c}" title="${["FL", "FR", "BL", "BR"][j]}"></i>`).join("");
  return `<span><em>speed</em>${m[o].toFixed(2)} <small>m/s</small></span>
    <span><em>height</em>${m[o + 1].toFixed(2)} <small>m</small></span>
    <span><em>return</em>${fmt(m[o + 3])}</span>
    <span class="feet"><em>feet</em><span class="feetmap">${down}</span></span>`;
}

async function boot(): Promise<void> {
  const [scene, index, curves, robust] = await Promise.all([loadScene(), loadIndex(), loadCurves(), loadRobustness()]);
  const runs = new Map(index.runs.map((r) => [r.id, r]));
  const theme = readTheme();
  const viewA = new AntViewer($<HTMLCanvasElement>("canvasA"), scene, theme);
  const viewB = new AntViewer($<HTMLCanvasElement>("canvasB"), scene, theme);
  const gait = new GaitView($<HTMLCanvasElement>("gait"));
  $("nEval").textContent = String(index.seeds.length);

  let a: Rollout | null = null;
  let b: Rollout | null = null;
  let frame = 0;
  let playing = true;
  let speed = 1;
  let holdUntil = 0;
  let token = 0;

  const sac = [...runs.values()].filter((r) => r.id.startsWith("sac_") && r.group === "progression");
  const finalSac = sac[sac.length - 1];
  const tunedPpo = index.runs.find((r) => r.id.startsWith("ppo_tuned") && r.group === "progression");
  const legacy = runs.get("ppo_legacy");

  const length = () => Math.max(a?.frames ?? 1, b?.frames ?? 0);

  let navSection: NavSection | null = null;
  function resize(): void {
    viewA.resize();
    viewB.resize();
    gait.resize();
    navSection?.resize();
  }
  window.addEventListener("resize", resize);

  function caption(slot: "A" | "B", meta: RunMeta | null): void {
    $(`name${slot}`).textContent = meta?.label ?? "--";
    const air = meta?.gait.airborne;
    $(`sub${slot}`).textContent = meta
      ? `mean return ${fmt(meta.eval.return_mean)}, fell in ${meta.eval.falls}/${meta.eval.episodes}${air != null ? `, airborne ${Math.round(air * 100)}%` : ""}`
      : "";
    $(`name${slot}`).style.setProperty("--swatch", meta ? colorFor(meta.id) : "transparent");
  }

  async function select(id: string, compareId: string | null = ($<HTMLSelectElement>("compare").value || null)): Promise<void> {
    const t = ++token;
    const meta = runs.get(id);
    if (!meta) return;
    const metaB = compareId ? runs.get(compareId) ?? null : null;
    const [ra, rb] = await Promise.all([loadRollout(meta, scene.bodies), metaB ? loadRollout(metaB, scene.bodies) : null]);
    if (t !== token) return;
    a = ra;
    b = rb;
    viewA.setRollout(ra);
    if (rb) viewB.setRollout(rb);
    $("stageB").hidden = !rb;
    $("stages").classList.toggle("split", !!rb);
    gait.setRollout(ra);
    caption("A", meta);
    caption("B", metaB);
    frame = 0;
    holdUntil = 0;
    $<HTMLInputElement>("scrub").max = String(length() - 1);
    document.querySelectorAll<HTMLButtonElement>(".run").forEach((el) => el.setAttribute("aria-pressed", String(el.dataset.id === id)));
    requestAnimationFrame(resize);
    const url = new URL(location.href);
    url.searchParams.set("run", id);
    if (compareId) url.searchParams.set("vs", compareId);
    else url.searchParams.delete("vs");
    history.replaceState(null, "", url);
  }

  // Picker lists.
  const item = (r: RunMeta) => `<button class="run" data-id="${r.id}" aria-pressed="false" style="--swatch:${colorFor(r.id)}">
      <span class="lbl">${r.label}</span>
      <span class="val">${fmt(r.eval.return_mean)}</span>
      <span class="sub">${r.eval.speed_mean.toFixed(2)} m/s${r.eval.falls ? `, fell ${r.eval.falls}/${r.eval.episodes}` : ""}</span>
    </button>`;
  $("listProgress").innerHTML = index.runs.filter((r) => r.group === "progression").map(item).join("");
  $("listStress").innerHTML = index.runs.filter((r) => r.group === "stress").map(item).join("");
  document.querySelector(".picker")!.addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>(".run");
    if (btn?.dataset.id) void select(btn.dataset.id);
  });
  $("compare").innerHTML += index.runs.map((r) => `<option value="${r.id}">${r.label}</option>`).join("");
  $("compare").addEventListener("change", () => a && void select(a.meta.id));

  // Transport.
  const scrub = $<HTMLInputElement>("scrub");
  scrub.addEventListener("input", () => {
    frame = Number(scrub.value);
    holdUntil = 0;
  });
  gait.onScrub = (f) => {
    frame = f;
    holdUntil = 0;
  };
  const setPlaying = (p: boolean) => {
    playing = p;
    $("play").classList.toggle("paused", !p);
    $("play").setAttribute("aria-label", p ? "Pause" : "Play");
  };
  $("play").addEventListener("click", () => setPlaying(!playing));
  window.addEventListener("keydown", (e) => {
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLSelectElement) return;
    if (e.code === "Space") {
      e.preventDefault();
      setPlaying(!playing);
    } else if (e.code === "ArrowRight") frame = Math.min(length() - 1, frame + FPS);
    else if (e.code === "ArrowLeft") frame = Math.max(0, frame - FPS);
  });
  document.querySelectorAll<HTMLButtonElement>("[data-speed]").forEach((btn) => btn.addEventListener("click", () => {
    speed = Number(btn.dataset.speed);
    document.querySelectorAll("[data-speed]").forEach((x) => x.classList.toggle("on", x === btn));
  }));

  // Theme toggle repaints the 3D scenes from the CSS tokens.
  $("theme").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("quadruped.theme", next);
    } catch {
      /* storage blocked */
    }
    const t = readTheme();
    viewA.applyTheme(t);
    viewB.applyTheme(t);
    navSection?.applyTheme();
    drawCharts();
  });

  // Progression strip.
  const progression = [runs.get("random"), ...sac].filter(Boolean) as RunMeta[];
  const maxRet = Math.max(...progression.map((r) => r.eval.return_mean), 1);
  $("strip").innerHTML = progression.map((r) => `<button class="step" data-id="${r.id}">
      <span class="lbl">${r.id === "random" ? "untrained" : r.label.replace("SAC at ", "")}</span>
      <span class="bar"><i style="height:${Math.max(2, (r.eval.return_mean / maxRet) * 100)}%"></i></span>
      <b>${fmt(r.eval.return_mean)}</b>
      <span class="sub">${r.eval.distance_mean.toFixed(0)} m in ${(r.eval.length_mean / FPS).toFixed(0)} s</span>
    </button>`).join("");
  $("strip").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>(".step");
    if (btn?.dataset.id) {
      void select(btn.dataset.id);
      $("bench").scrollIntoView({ behavior: "smooth" });
    }
  });

  // Stress cards.
  const stress = index.runs.filter((r) => r.group === "stress");
  $("stressCards").innerHTML = `<h3>Stress tests</h3><div class="cards">` + stress.map((r) => {
    const base = r.id.startsWith("sac") ? finalSac : tunedPpo;
    const delta = base ? r.eval.return_mean - base.eval.return_mean : 0;
    return `<button class="card" data-id="${r.id}" style="--swatch:${colorFor(r.id)}">
      <span class="lbl">${r.label}</span>
      <b>${fmt(r.eval.return_mean)}</b>
      <span class="sub ${delta < -500 ? "neg" : ""}">${delta >= 0 ? "+" : "−"}${fmt(Math.abs(delta))} vs nominal, fell ${r.eval.falls}/${r.eval.episodes}</span>
    </button>`;
  }).join("") + `</div>`;
  $("stressCards").addEventListener("click", (e) => {
    const btn = (e.target as HTMLElement).closest<HTMLButtonElement>(".card");
    if (btn?.dataset.id) {
      void select(btn.dataset.id);
      $("bench").scrollIntoView({ behavior: "smooth" });
    }
  });

  function drawCharts(): void {
    const series: Series[] = [];
    const add = (key: string, label: string, color: string) => {
      const c = curves[key];
      if (c) series.push({ id: key, label, color, x: c.timesteps, y: c.mean, sd: c.std });
    };
    add("sac", "SAC", "var(--sac)");
    const tunedKey = Object.keys(curves).find((key) => key.startsWith("ppo_tuned"));
    if (tunedKey) add(tunedKey, "PPO, tuned", "var(--ppo)");
    add("ppo", "PPO, first run", "var(--legacy)");
    if (series.length) {
      lines($("curveChart"), {
        series,
        xFormat: (v) => `${k(v)} steps`,
        yFormat: (v) => fmt(v),
        yLabel: "evaluation return",
        marks: sac.map((r) => ({
          x: Number(r.id.replace("sac_", "").replace("k", "")) * 1000,
          label: `Watch ${r.label}`,
          onClick: () => {
            void select(r.id);
            $("bench").scrollIntoView({ behavior: "smooth" });
          },
        })),
      });
    }
    if (robust) {
      const host = $("heats");
      host.replaceChildren();
      const vmax = Math.max(...Object.values(robust.policies).flatMap((p) => p.cells.flat().map((c) => c.return)));
      for (const [key, p] of Object.entries(robust.policies)) {
        const box = document.createElement("div");
        box.className = "heatbox";
        box.innerHTML = `<h3><i style="background:${key === "sac" ? "var(--sac)" : "var(--ppo)"}"></i>${p.label}</h3>`;
        const grid = document.createElement("div");
        box.append(grid);
        host.append(box);
        heatmap(grid, {
          rows: robust.mass,
          cols: robust.friction,
          values: p.cells.map((row) => row.map((c) => c.return)),
          falls: p.cells.map((row) => row.map((c) => c.falls)),
          episodes: robust.episodes,
          vmax,
        });
      }
      const legend = document.createElement("div");
      legend.className = "heat-legend";
      legend.innerHTML = `<span>0</span><i></i><span>${fmt(vmax)}</span><em>mean return over ${robust.episodes} episodes per cell</em>`;
      host.append(legend);
    }
  }
  drawCharts();

  if (robust) {
    const lowFr = robust.friction.map((f, i) => (f <= 1 ? i : -1)).filter((i) => i >= 0);
    const highFr = robust.friction.map((f, i) => (f > 1.2 ? i : -1)).filter((i) => i >= 0);
    const avg = (cells: { return: number }[][], cols: number[]) =>
      cells.flatMap((row) => cols.map((c) => row[c].return)).reduce((s, v, _, a) => s + v / a.length, 0);
    const rows = Object.entries(robust.policies).map(([, p]) => ({ label: p.label, low: avg(p.cells, lowFr), high: avg(p.cells, highFr) }));
    const push = index.runs.filter((r) => r.id.endsWith("_push"));
    $("finding").innerHTML = `<p>Mass hardly matters; grip does. At nominal friction or below, ${rows.map((r) => `${r.label} averages ${fmt(r.low)}`).join(" and ")}. At 1.3× and above, ${rows.map((r) => fmt(r.high)).join(" and ")}: both gaits rely on the feet sliding.
      ${push.length ? `A 120 N shove: ${push.map((r) => `${r.label.split(":")[0]} fell in ${r.eval.falls}/${r.eval.episodes}`).join(", ")}.` : ""}</p>`;
  }
  let rt = 0;
  window.addEventListener("resize", () => {
    clearTimeout(rt);
    rt = window.setTimeout(drawCharts, 150);
  });

  // Playback loop.
  let last = performance.now();
  let hudTick = 0;
  function tick(now: number): void {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    const n = length();
    if (a && playing) {
      if (frame >= n - 1) {
        if (!holdUntil) holdUntil = now + 1600;
        else if (now > holdUntil) {
          frame = 0;
          holdUntil = 0;
        }
      } else frame = Math.min(n - 1, frame + dt * FPS * speed);
    }
    if (a) {
      viewA.setFrame(frame);
      viewA.render(dt);
      const ia = Math.min(a.frames - 1, Math.floor(frame));
      $("fellA").hidden = !(a.meta.recorded.fell && frame >= a.frames - 1);
      if (b) {
        viewB.setFrame(frame);
        viewB.render(dt);
        $("fellB").hidden = !(b.meta.recorded.fell && frame >= b.frames - 1);
      }
      hudTick -= dt;
      if (hudTick <= 0) {
        hudTick = 0.08;
        $("hudA").innerHTML = hudHtml(a, ia);
        if (b) $("hudB").innerHTML = hudHtml(b, Math.min(b.frames - 1, Math.floor(frame)));
        $("clock").textContent = `${(frame / FPS).toFixed(1).padStart(4, "0")} s`;
        if (document.activeElement !== scrub) scrub.value = String(frame);
      }
      gait.draw(Math.min(a.frames - 1, frame));
    }
    navSection?.tick(dt);
    requestAnimationFrame(tick);
  }

  resize();
  const params = new URLSearchParams(location.search);
  const startId = params.get("run") && runs.has(params.get("run")!) ? params.get("run")! : finalSac?.id ?? index.runs[0].id;
  const vs = params.get("vs");
  if (vs && runs.has(vs)) $<HTMLSelectElement>("compare").value = vs;
  await select(startId);
  document.querySelectorAll<HTMLAnchorElement>("#stageBar [data-stage]").forEach((a) =>
    a.addEventListener("click", () => {
      const id = a.dataset.stage === "random" ? "random" : finalSac?.id;
      if (id) void select(id, null);
    }),
  );
  requestAnimationFrame(tick);
  navSection = await initNav(scene);
  resize();
}

boot().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML("afterbegin", `<div class="fatal">Could not load the recorded rollouts: ${String(err.message ?? err)}</div>`);
});
