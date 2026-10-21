const NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

function ticks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo || 1;
  const raw = span / count;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= count) ?? mag * 10;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(8)));
  return out;
}

export interface Series {
  id: string;
  label: string;
  color: string; // CSS var reference, e.g. var(--sac)
  x: number[];
  y: number[];
  sd?: number[];
}

export interface LinesSpec {
  series: Series[];
  xFormat: (v: number) => string;
  yFormat: (v: number) => string;
  yLabel: string;
  marks?: { x: number; label: string; onClick: () => void }[];
}

/** Multi-series line chart on one y axis: legend, direct end labels, shared crosshair tooltip. */
export function lines(host: HTMLElement, spec: LinesSpec): void {
  host.replaceChildren();
  const width = host.clientWidth || 700;
  const height = host.clientHeight || 300;
  const m = { l: 56, r: 118, t: 12, b: 28 };
  if (width < 520) m.r = 16;
  const iw = width - m.l - m.r;
  const ih = height - m.t - m.b;
  const xsAll = spec.series.flatMap((s) => s.x);
  const ysAll = spec.series.flatMap((s) => s.y.map((v, i) => v + (s.sd?.[i] ?? 0)).concat(s.y.map((v, i) => v - (s.sd?.[i] ?? 0))));
  const x0 = 0;
  const x1 = Math.max(...xsAll);
  const y0 = Math.min(0, ...ysAll);
  const y1 = Math.max(...ysAll);
  const sx = (v: number) => m.l + ((v - x0) / (x1 - x0 || 1)) * iw;
  const sy = (v: number) => m.t + (1 - (v - y0) / (y1 - y0 || 1)) * ih;

  const legend = document.createElement("div");
  legend.className = "legend-row";
  legend.innerHTML = spec.series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join("");

  const svg = el("svg", { width, height, viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": spec.yLabel });
  for (const t of ticks(y0, y1, 5)) {
    svg.append(el("line", { x1: m.l, x2: m.l + iw, y1: sy(t), y2: sy(t), class: t === 0 ? "zero" : "grid" }));
    const tx = el("text", { x: m.l - 8, y: sy(t), class: "tick", "text-anchor": "end", "dominant-baseline": "middle" });
    tx.textContent = spec.yFormat(t);
    svg.append(tx);
  }
  for (const t of ticks(x0, x1, Math.max(2, Math.floor(iw / 120)))) {
    const tx = el("text", { x: sx(t), y: height - 8, class: "tick", "text-anchor": "middle" });
    tx.textContent = spec.xFormat(t);
    svg.append(tx);
  }
  for (const mark of spec.marks ?? []) {
    const g = el("g", { class: "mark", tabindex: 0, role: "button", "aria-label": mark.label });
    g.append(el("line", { x1: sx(mark.x), x2: sx(mark.x), y1: m.t, y2: m.t + ih }));
    g.append(el("rect", { x: sx(mark.x) - 7, y: m.t, width: 14, height: ih, fill: "transparent" }));
    g.addEventListener("click", mark.onClick);
    g.addEventListener("keydown", (e) => (e as KeyboardEvent).key === "Enter" && mark.onClick());
    svg.append(g);
  }
  const ends: { x: number; y: number; s: Series }[] = [];
  for (const s of spec.series) {
    if (s.sd) {
      const top = s.x.map((x, i) => `${sx(x)},${sy(s.y[i] + s.sd![i])}`);
      const bot = s.x.map((x, i) => `${sx(x)},${sy(s.y[i] - s.sd![i])}`).reverse();
      svg.append(el("polygon", { points: [...top, ...bot].join(" "), style: `fill:${s.color}`, opacity: 0.1 }));
    }
    svg.append(el("polyline", {
      points: s.x.map((x, i) => `${sx(x)},${sy(s.y[i])}`).join(" "),
      fill: "none", style: `stroke:${s.color}`, "stroke-width": 2, "stroke-linejoin": "round",
    }));
    ends.push({ x: sx(s.x[s.x.length - 1]), y: sy(s.y[s.y.length - 1]), s });
  }
  // Direct labels at line ends, nudged apart.
  if (m.r > 40) {
    ends.sort((a, b) => a.y - b.y);
    for (let i = 1; i < ends.length; i += 1) if (ends[i].y - ends[i - 1].y < 16) ends[i].y = ends[i - 1].y + 16;
    for (const e of ends) {
      // Label sits where the line ends, so a shorter run is labeled at its own end.
      const tx = el("text", { x: e.x + 8, y: e.y, class: "endlabel", "dominant-baseline": "middle" });
      tx.textContent = e.s.label;
      svg.append(tx);
    }
  }

  const cross = el("line", { y1: m.t, y2: m.t + ih, class: "cross", visibility: "hidden" });
  svg.append(cross);
  const dots = spec.series.map((s) => {
    const d = el("circle", { r: 4, style: `fill:${s.color}`, class: "dot", visibility: "hidden" });
    svg.append(d);
    return d;
  });
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  svg.addEventListener("pointermove", (e) => {
    const rect = svg.getBoundingClientRect();
    const xv = x0 + ((e.clientX - rect.left - m.l) / iw) * (x1 - x0);
    const rows: string[] = [];
    let cx = 0;
    spec.series.forEach((s, k) => {
      let best = 0;
      for (let i = 1; i < s.x.length; i += 1) if (Math.abs(s.x[i] - xv) < Math.abs(s.x[best] - xv)) best = i;
      cx = sx(s.x[best]);
      dots[k].setAttribute("cx", String(cx));
      dots[k].setAttribute("cy", String(sy(s.y[best])));
      dots[k].setAttribute("visibility", "visible");
      rows.push(`<span><i style="background:${s.color}"></i>${s.label}<b>${spec.yFormat(s.y[best])}</b></span>`);
    });
    cross.setAttribute("x1", String(cx));
    cross.setAttribute("x2", String(cx));
    cross.setAttribute("visibility", "visible");
    tip.innerHTML = `<em>${spec.xFormat(xv)}</em>${rows.join("")}`;
    tip.hidden = false;
    tip.style.left = `${Math.min(cx + 14, width - 220)}px`;
    tip.style.top = `${m.t + 26}px`;
  });
  svg.addEventListener("pointerleave", () => {
    tip.hidden = true;
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
  const box = document.createElement("div");
  box.className = "plotbox";
  box.append(svg, tip);
  host.append(legend, box);
}

export interface HeatSpec {
  rows: number[]; // mass
  cols: number[]; // friction
  values: number[][];
  falls: number[][];
  episodes: number;
  vmax: number;
  onCell?: (r: number, c: number) => void;
}

/** Sequential single-hue heatmap: one hue, light to dark, value printed in each cell. */
export function heatmap(host: HTMLElement, spec: HeatSpec): void {
  host.replaceChildren();
  const grid = document.createElement("div");
  grid.className = "heat";
  grid.style.gridTemplateColumns = `56px repeat(${spec.cols.length}, minmax(0, 1fr))`;
  const corner = document.createElement("span");
  corner.className = "axis corner";
  corner.textContent = "mass ↓ / friction →";
  grid.append(corner);
  for (const c of spec.cols) {
    const h = document.createElement("span");
    h.className = "axis";
    h.textContent = `${c.toFixed(1)}×`;
    grid.append(h);
  }
  spec.rows.forEach((r, ri) => {
    const h = document.createElement("span");
    h.className = "axis";
    h.textContent = `${r.toFixed(1)}×`;
    grid.append(h);
    spec.cols.forEach((c, ci) => {
      const v = spec.values[ri][ci];
      const t = Math.max(0, Math.min(1, v / spec.vmax));
      const cell = document.createElement("button");
      cell.className = "cell";
      cell.style.setProperty("--t", t.toFixed(3));
      cell.dataset.dark = String(t > 0.55);
      const isNominal = r === 1 && c === 1;
      if (isNominal) cell.classList.add("nominal");
      const falls = spec.falls[ri][ci];
      cell.innerHTML = `<b>${Math.round(v).toLocaleString()}</b>${falls ? `<small>${falls}/${spec.episodes} fell</small>` : ""}`;
      cell.title = `torso mass ${r}×, friction ${c}×: mean return ${Math.round(v)} over ${spec.episodes} episodes, ${falls} fell`;
      if (spec.onCell) cell.addEventListener("click", () => spec.onCell?.(ri, ci));
      else cell.tabIndex = -1;
      grid.append(cell);
    });
  });
  host.append(grid);
}
