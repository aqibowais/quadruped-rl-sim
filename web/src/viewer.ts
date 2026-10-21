import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type { MazeData, Rollout, Scene } from "./data";

export interface ViewerTheme {
  bg: string;
  floorA: string;
  floorB: string;
  body: string;
  leg: string;
  foot: string;
  accent: string;
  trail: string;
}

export function readTheme(): ViewerTheme {
  const css = getComputedStyle(document.documentElement);
  const v = (name: string) => css.getPropertyValue(name).trim() || "#888";
  return {
    bg: v("--stage"),
    floorA: v("--floor-a"),
    floorB: v("--floor-b"),
    body: v("--robot"),
    leg: v("--robot-leg"),
    foot: v("--contact"),
    accent: v("--sac"),
    trail: v("--trail"),
  };
}

function checker(a: string, b: string): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 256;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = a;
  ctx.fillRect(0, 0, 256, 256);
  ctx.fillStyle = b;
  ctx.fillRect(0, 0, 128, 128);
  ctx.fillRect(128, 128, 128, 128);
  ctx.strokeStyle = b;
  ctx.globalAlpha = 0.5;
  ctx.lineWidth = 2;
  ctx.strokeRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

const FLOOR = 160; // meters, follows the robot in 2 m steps so the checker never swims
const TRAIL_MAX = 1200;

/** Replays recorded MuJoCo body poses. Knows nothing about playback timing. */
export class AntViewer {
  private readonly renderer: THREE.WebGLRenderer;
  private readonly scene = new THREE.Scene();
  private readonly camera = new THREE.PerspectiveCamera(38, 1, 0.05, 400);
  private readonly controls: OrbitControls;
  /** MuJoCo is z-up; this group turns it into three.js y-up. */
  private readonly world = new THREE.Group();
  private readonly bodies: THREE.Group[] = [];
  private readonly feet: { mesh: THREE.Mesh; bit: number }[] = [];
  private readonly floor: THREE.Mesh;
  private readonly trail: THREE.Line;
  private readonly trailPos = new Float32Array(TRAIL_MAX * 3);
  private readonly arrow: THREE.ArrowHelper;
  /** Commanded heading of a steerable walker. */
  private readonly command: THREE.ArrowHelper;
  private mazeGroup: THREE.Group | null = null;
  private overhead: { center: THREE.Vector3; span: number } | null = null;
  private readonly mats: Record<"chassis" | "plate" | "joint" | "shin" | "foot" | "contact", THREE.MeshStandardMaterial>;
  private rollout: Rollout | null = null;
  private readonly torso = new THREE.Vector3();
  private follow = new THREE.Vector3();
  private lastFrame = -1;

  constructor(private readonly canvas: HTMLCanvasElement, scene: Scene, theme: ViewerTheme) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: false });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.world.rotation.x = -Math.PI / 2;
    this.scene.add(this.world);

    const hemi = new THREE.HemisphereLight(0xffffff, 0x6b6253, 1.15);
    this.scene.add(hemi);
    const sun = new THREE.DirectionalLight(0xfff4e2, 2.1);
    sun.position.set(4, 9, 3);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -4;
    sun.shadow.camera.right = 4;
    sun.shadow.camera.top = 4;
    sun.shadow.camera.bottom = -4;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun, sun.target);
    this.sun = sun;

    this.mats = {
      chassis: new THREE.MeshStandardMaterial({ color: 0xe9ebef, metalness: 0.55, roughness: 0.38 }),
      plate: new THREE.MeshStandardMaterial({ color: 0x2c3138, metalness: 0.65, roughness: 0.32 }),
      joint: new THREE.MeshStandardMaterial({ color: 0x3a4049, metalness: 0.8, roughness: 0.28 }),
      shin: new THREE.MeshStandardMaterial({ metalness: 0.45, roughness: 0.4 }),
      foot: new THREE.MeshStandardMaterial({ color: 0x1d2025, metalness: 0.2, roughness: 0.85 }),
      contact: new THREE.MeshStandardMaterial({ metalness: 0.2, roughness: 0.6 }),
    };

    for (let b = 0; b < scene.bodies; b += 1) {
      const g = new THREE.Group();
      this.world.add(g);
      this.bodies.push(g);
    }
    for (const spec of scene.geoms) {
      const part = spec.body === 0 && spec.type === "sphere" ? this.torsoMesh(spec.size[0]) : this.legMesh(spec.size[0], spec.size[1], spec.foot !== null);
      part.group.position.set(spec.pos[0], spec.pos[1], spec.pos[2]);
      part.group.quaternion.set(spec.quat[1], spec.quat[2], spec.quat[3], spec.quat[0]);
      part.group.traverse((n) => {
        if (n instanceof THREE.Mesh) n.castShadow = true;
      });
      this.bodies[spec.body].add(part.group);
      if (spec.foot && part.foot) this.feet.push({ mesh: part.foot, bit: scene.feet.indexOf(spec.foot) });
    }

    this.floor = new THREE.Mesh(new THREE.PlaneGeometry(FLOOR, FLOOR), new THREE.MeshStandardMaterial({ roughness: 0.95 }));
    this.floor.receiveShadow = true;
    this.world.add(this.floor);

    const tg = new THREE.BufferGeometry();
    tg.setAttribute("position", new THREE.BufferAttribute(this.trailPos, 3));
    tg.setDrawRange(0, 0);
    this.trail = new THREE.Line(tg, new THREE.LineBasicMaterial({ transparent: true, opacity: 0.85 }));
    this.trail.frustumCulled = false;
    this.world.add(this.trail);

    this.arrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 0.9, 0xc2410c, 0.22, 0.14);
    this.arrow.visible = false;
    this.world.add(this.arrow);
    this.command = new THREE.ArrowHelper(new THREE.Vector3(1, 0, 0), new THREE.Vector3(), 1.4, 0x1f5f99, 0.35, 0.22);
    this.command.visible = false;
    this.world.add(this.command);

    this.camera.position.set(-2.2, 1.6, 3.4);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.2;
    this.controls.maxDistance = 25;
    this.controls.maxPolarAngle = Math.PI * 0.48;
    this.controls.enablePan = false;
    this.applyTheme(theme);
    this.resize();
  }

  private readonly sun: THREE.DirectionalLight;

  // MuJoCo capsules run along local z; three.js cylinders along y.
  private legMesh(radius: number, half: number, shin: boolean): { group: THREE.Group; foot: THREE.Mesh | null } {
    const group = new THREE.Group();
    const taper = shin ? 0.5 : 0.82;
    const body = new THREE.Mesh(new THREE.CylinderGeometry(radius * taper, radius * 0.95, half * 2, 20), shin ? this.mats.shin : this.mats.chassis);
    body.rotation.x = Math.PI / 2;
    group.add(body);
    const hub = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.12, 20, 14), this.mats.joint);
    hub.position.z = -half;
    group.add(hub);
    if (!shin) return { group, foot: null };
    const foot = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.72, 18, 12), this.mats.foot);
    foot.position.z = half;
    foot.scale.set(1.25, 1.25, 0.8);
    group.add(foot);
    return { group, foot };
  }

  private torsoMesh(r: number): { group: THREE.Group; foot: null } {
    const group = new THREE.Group();
    const shell = new THREE.Mesh(new THREE.SphereGeometry(r, 36, 24), this.mats.chassis);
    shell.scale.set(1.18, 1.05, 0.72);
    group.add(shell);
    const deck = new THREE.Mesh(new THREE.BoxGeometry(r * 0.95, r * 1.15, r * 0.34), this.mats.plate);
    deck.position.z = r * 0.5;
    group.add(deck);
    const dome = new THREE.Mesh(new THREE.SphereGeometry(r * 0.3, 20, 14), this.mats.shin);
    dome.position.z = r * 0.72;
    group.add(dome);
    for (const side of [-1, 1]) {
      const eye = new THREE.Mesh(new THREE.SphereGeometry(r * 0.1, 12, 10), this.mats.plate);
      eye.position.set(r * 0.82, side * r * 0.42, r * 0.18);
      group.add(eye);
    }
    return { group, foot: null };
  }

  applyTheme(t: ViewerTheme): void {
    this.scene.background = new THREE.Color(t.bg);
    this.scene.fog = new THREE.Fog(t.bg, 14, 60);
    const mat = this.floor.material as THREE.MeshStandardMaterial;
    mat.map?.dispose();
    mat.map = checker(t.floorA, t.floorB);
    mat.map.repeat.set(FLOOR / 2, FLOOR / 2);
    mat.needsUpdate = true;
    this.mats.shin.color.set(t.accent);
    this.mats.contact.color.set(t.accent);
    this.mats.contact.emissive.set(t.accent).multiplyScalar(0.6);
    (this.trail.material as THREE.LineBasicMaterial).color.set(t.trail);
    this.arrow.setColor(new THREE.Color(t.accent));
  }

  resize(): void {
    const w = this.canvas.clientWidth || 640;
    const h = this.canvas.clientHeight || 400;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Keep the horizontal view roughly constant on portrait screens.
    this.camera.fov = this.camera.aspect < 1 ? 38 / Math.max(0.55, this.camera.aspect) : 38;
    this.camera.updateProjectionMatrix();
  }

  /** Walls, planned path, start and goal of a maze, or null for the open floor. */
  setMaze(m: MazeData | null, accent: string): void {
    if (this.mazeGroup) {
      this.world.remove(this.mazeGroup);
      this.mazeGroup.traverse((o) => (o as THREE.Mesh).geometry?.dispose());
      this.mazeGroup = null;
    }
    this.overhead = null;
    if (!m) return;
    const g = new THREE.Group();
    const wallMat = new THREE.MeshStandardMaterial({ color: this.wallColor, roughness: 0.85 });
    for (const [x, y, hx, hy] of m.walls) {
      const box = new THREE.Mesh(new THREE.BoxGeometry(hx * 2, hy * 2, 1.4), wallMat);
      box.position.set(x, y, 0.7);
      box.castShadow = true;
      box.receiveShadow = true;
      g.add(box);
    }
    const pts = m.path.map(([x, y]) => new THREE.Vector3(x, y, 0.02));
    const path = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineDashedMaterial({ color: accent, dashSize: 0.5, gapSize: 0.35 }));
    path.computeLineDistances();
    g.add(path);
    const ring = (x: number, y: number, r: number, color: string) => {
      const mesh = new THREE.Mesh(new THREE.RingGeometry(r * 0.8, r, 48), new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }));
      mesh.position.set(x, y, 0.015);
      return mesh;
    };
    g.add(ring(m.start[0], m.start[1], 0.9, "#827c70"));
    g.add(ring(m.goal[0], m.goal[1], m.goal_radius, accent));
    const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 2.4, 8), new THREE.MeshStandardMaterial({ color: 0x1b1915 }));
    pole.rotation.x = Math.PI / 2;
    pole.position.set(m.goal[0], m.goal[1], 1.2);
    g.add(pole);
    const flag = new THREE.Mesh(new THREE.PlaneGeometry(0.9, 0.55), new THREE.MeshStandardMaterial({ color: accent, side: THREE.DoubleSide }));
    flag.rotation.x = Math.PI / 2;
    flag.position.set(m.goal[0] + 0.45, m.goal[1], 2.1);
    g.add(flag);
    this.world.add(g);
    this.mazeGroup = g;
    const rows = m.grid.length;
    const cols = m.grid[0].length;
    // Overhead view centred on the maze, in three.js coordinates (x, z, -y).
    this.overhead = { center: new THREE.Vector3(0, 0, 0), span: Math.max(rows, cols) * m.cell };
  }

  private wallColor = "#d9d2c3";
  /** Maze clips can switch between following the robot and looking down on the whole maze. */
  overheadOn = false;

  setRollout(r: Rollout): void {
    this.rollout = r;
    this.lastFrame = -1;
    const p = r.poses;
    const start = new THREE.Vector3(p[0], p[2], -p[1]);
    const delta = start.clone().sub(this.follow);
    this.follow.copy(start);
    this.camera.position.add(delta);
    this.controls.target.copy(start);
  }

  /** Show fractional frame `f`, blending between recorded steps. */
  setFrame(f: number): void {
    const r = this.rollout;
    if (!r) return;
    const n = r.frames;
    const clamped = Math.max(0, Math.min(n - 1, f));
    const i0 = Math.floor(clamped);
    const i1 = Math.min(n - 1, i0 + 1);
    const a = clamped - i0;
    const p = r.poses;
    const q0 = new THREE.Quaternion();
    const q1 = new THREE.Quaternion();
    for (let b = 0; b < this.bodies.length; b += 1) {
      const o0 = i0 * r.stride + b * 7;
      const o1 = i1 * r.stride + b * 7;
      const g = this.bodies[b];
      g.position.set(
        p[o0] + (p[o1] - p[o0]) * a,
        p[o0 + 1] + (p[o1 + 1] - p[o0 + 1]) * a,
        p[o0 + 2] + (p[o1 + 2] - p[o0 + 2]) * a,
      );
      q0.set(p[o0 + 4], p[o0 + 5], p[o0 + 6], p[o0 + 3]);
      q1.set(p[o1 + 4], p[o1 + 5], p[o1 + 6], p[o1 + 3]);
      g.quaternion.slerpQuaternions(q0, q1, a);
    }
    const contact = r.contacts[i0];
    for (const foot of this.feet) foot.mesh.material = contact & (1 << foot.bit) ? this.mats.contact : this.mats.foot;

    const tx = this.bodies[0].position.x;
    const ty = this.bodies[0].position.y;
    this.floor.position.set(Math.round(tx / 2) * 2, Math.round(ty / 2) * 2, 0);

    if (i0 !== this.lastFrame) {
      const count = Math.min(i0 + 1, TRAIL_MAX);
      const from = i0 + 1 - count;
      for (let k = 0; k < count; k += 1) {
        const o = (from + k) * r.stride;
        this.trailPos[k * 3] = p[o];
        this.trailPos[k * 3 + 1] = p[o + 1];
        this.trailPos[k * 3 + 2] = 0.004;
      }
      this.trail.geometry.setDrawRange(0, count);
      this.trail.geometry.attributes.position.needsUpdate = true;
      this.lastFrame = i0;
    }

    if (r.commands) {
      const h = r.commands[i0];
      this.command.visible = true;
      this.command.setDirection(new THREE.Vector3(Math.cos(h), Math.sin(h), 0));
      this.command.position.set(tx, ty, this.bodies[0].position.z + 0.3);
    } else this.command.visible = false;

    const pushing = r.meta.pushes.some((s) => i0 >= s - 2 && i0 < s + 10);
    this.arrow.visible = pushing;
    if (pushing) this.arrow.position.set(tx, ty - 1.15, this.bodies[0].position.z);
  }

  render(dt: number): void {
    if (this.overhead && this.overheadOn) {
      const o = this.overhead;
      const want = new THREE.Vector3(o.center.x, o.span * 0.95, o.center.z + o.span * 0.55);
      this.camera.position.lerp(want, 1 - Math.exp(-dt * 3));
      this.camera.lookAt(o.center);
      this.sun.position.set(4, 30, 3);
      this.sun.target.position.copy(o.center);
      this.renderer.render(this.scene, this.camera);
      return;
    }
    // Torso in three.js coordinates.
    this.bodies[0].getWorldPosition(this.torso);
    const k = 1 - Math.exp(-dt * 4);
    const target = this.follow.clone().lerp(this.torso, k);
    const delta = target.clone().sub(this.follow);
    this.follow.copy(target);
    this.camera.position.add(delta);
    this.controls.target.copy(this.follow);
    this.controls.update();
    this.sun.position.set(this.follow.x + 4, 9, this.follow.z + 3);
    this.sun.target.position.copy(this.follow);
    this.renderer.render(this.scene, this.camera);
  }
}
