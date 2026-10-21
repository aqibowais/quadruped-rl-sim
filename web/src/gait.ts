import type { Rollout } from "./data";

const FEET = ["Front left", "Front right", "Back left", "Back right"];
// Ant-v4 actuator order: hip_4, ankle_4, hip_1, ankle_1, ... (leg 4 is back right).
const JOINTS = ["BR hip", "BR ankle", "FL hip", "FL ankle", "FR hip", "FR ankle", "BL hip", "BL ankle"];
const WINDOW = 80; // frames shown in the zoomed view (4 s at 20 Hz)

function css(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function mix(a: number[], b: number[], t: number): string {
  return `rgb(${a.map((v, i) => Math.round(v + (b[i] - v) * t)).join(",")})`;
}

function hexRgb(hex: string): number[] {
  const h = hex.replace("#", "");
  return [0, 2, 4].map((i) => parseInt(h.slice(i, i + 2), 16));
}

/**
 * Gait diagram: which feet are on the ground, frame by frame, plus the
 * torque the policy sends to each joint. Click or drag to scrub.
 */
export class GaitView {
  private readonly ctx: CanvasRenderingContext2D;
  private dpr = 1;
  private w = 0;
  private h = 0;
  private rollout: Rollout | null = null;
  onScrub: ((frame: number) => void) | null = null;
  private dragging = false;

  constructor(private readonly canvas: HTMLCanvasElement) {
    this.ctx = canvas.getContext("2d")!;
    const scrub = (e: PointerEvent) => {
      const r = this.rollout;
      if (!r || !this.onScrub) return;
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const lx = this.labelW;
      const frac = Math.max(0, Math.min(1, (x - lx) / (this.w - lx)));
      if (y < this.overviewH + 6) this.onScrub(frac * (r.frames - 1));
      else if (this.zoomStart !== null) this.onScrub(this.zoomStart + frac * WINDOW);
    };
    canvas.addEventListener("pointerdown", (e) => {
      this.dragging = true;
      canvas.setPointerCapture(e.pointerId);
      scrub(e);
    });
    canvas.addEventListener("pointermove", (e) => this.dragging && scrub(e));
    canvas.addEventListener("pointerup", () => (this.dragging = false));
  }

  private readonly labelW = 78;
  private readonly overviewH = 26;
  private zoomStart: number | null = null;

  resize(): void {
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth;
    this.h = this.canvas.clientHeight;
    this.canvas.width = Math.round(this.w * this.dpr);
    this.canvas.height = Math.round(this.h * this.dpr);
  }

  setRollout(r: Rollout): void {
    this.rollout = r;
  }

  draw(frame: number): void {
    const r = this.rollout;
    const { ctx, w, h, labelW } = this;
    if (!r || !w) return;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const ink = css("--ink");
    const muted = css("--muted");
    const line = css("--line");
    const contact = css("--contact");
    const neg = hexRgb(css("--ppo"));
    const pos = hexRgb(css("--sac"));
    const mid = hexRgb(css("--cell-mid"));
    const plotW = w - labelW;
    ctx.font = "500 10.5px 'JetBrains Mono', monospace";
    ctx.textBaseline = "middle";

    // Overview: whole episode, feet down drawn as density.
    const oh = this.overviewH;
    const rowO = oh / 4;
    ctx.fillStyle = muted;
    ctx.textAlign = "left";
    ctx.fillText("Episode", 0, oh / 2);
    for (let f = 0; f < 4; f += 1) {
      ctx.fillStyle = line;
      ctx.fillRect(labelW, f * rowO + 0.5, plotW, rowO - 1);
    }
    ctx.fillStyle = contact;
    const per = plotW / r.frames;
    for (let i = 0; i < r.frames; i += 1) {
      const c = r.contacts[i];
      for (let f = 0; f < 4; f += 1) {
        if (c & (1 << f)) ctx.fillRect(labelW + i * per, f * rowO + 0.5, Math.max(1, per), rowO - 1);
      }
    }
    const start = Math.max(0, Math.min(r.frames - WINDOW, Math.round(frame - WINDOW / 2)));
    this.zoomStart = start;
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1;
    ctx.strokeRect(labelW + start * per + 0.5, 0.5, Math.max(4, WINDOW * per), oh - 1);

    // Zoomed gait diagram.
    const top = oh + 22;
    const span = Math.min(WINDOW, r.frames);
    const fw = plotW / span;
    const gaitRow = 15;
    ctx.fillStyle = muted;
    ctx.fillText("Foot contact", 0, top - 11);
    for (let f = 0; f < 4; f += 1) {
      const y = top + f * (gaitRow + 4);
      ctx.fillStyle = muted;
      ctx.fillText(FEET[f], 0, y + gaitRow / 2);
      ctx.fillStyle = line;
      ctx.fillRect(labelW, y, plotW, gaitRow);
      ctx.fillStyle = contact;
      for (let i = 0; i < span; i += 1) {
        const c = r.contacts[start + i];
        if (c & (1 << f)) ctx.fillRect(labelW + i * fw, y, fw + 0.6, gaitRow);
      }
    }

    // Torque strip.
    const tTop = top + 4 * (gaitRow + 4) + 24;
    const tRow = Math.max(6, Math.min(11, (h - tTop - 8) / 8));
    ctx.fillStyle = muted;
    ctx.fillText("Joint torque", 0, tTop - 11);
    for (let j = 0; j < 8; j += 1) {
      const y = tTop + j * tRow;
      ctx.fillStyle = muted;
      ctx.font = "400 9.5px 'JetBrains Mono', monospace";
      ctx.fillText(JOINTS[j], 0, y + tRow / 2);
      for (let i = 0; i < span; i += 1) {
        const v = r.torques[(start + i) * 8 + j] / 127;
        ctx.fillStyle = mix(mid, v < 0 ? neg : pos, Math.min(1, Math.abs(v)) ** 0.85);
        ctx.fillRect(labelW + i * fw, y, fw + 0.6, tRow - 1);
      }
    }
    ctx.font = "500 10.5px 'JetBrains Mono', monospace";

    // Push markers and playhead in the zoom.
    for (const p of r.meta.pushes) {
      if (p >= start && p < start + span) {
        const x = labelW + (p - start) * fw;
        ctx.fillStyle = css("--sac");
        ctx.fillRect(x, top - 4, 2, tTop + 8 * tRow - top + 4);
        ctx.textAlign = "left";
        ctx.fillText("shove", x + 4, top - 11);
      }
    }
    const px = labelW + (frame - start) * fw;
    ctx.fillStyle = ink;
    ctx.fillRect(px - 0.75, top - 4, 1.5, tTop + 8 * tRow - top + 4);
    const opx = labelW + frame * per;
    ctx.fillRect(opx - 0.75, 0, 1.5, oh);

    // Time axis labels for the zoom window.
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    const dt = 0.05;
    ctx.fillText(`${(start * dt).toFixed(1)} s – ${((start + span) * dt).toFixed(1)} s`, w, top - 11);
  }
}
