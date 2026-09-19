import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js";

const GEOM_SPHERE = 2;
const GEOM_CAPSULE = 3;

const CHASSIS = new THREE.MeshStandardMaterial({ color: 0xe9ebef, metalness: .55, roughness: .38 });
const PLATE = new THREE.MeshStandardMaterial({ color: 0x2c3138, metalness: .65, roughness: .32 });
const JOINT = new THREE.MeshStandardMaterial({ color: 0x3a4049, metalness: .8, roughness: .28 });
const FOOT = new THREE.MeshStandardMaterial({ color: 0x1d2025, metalness: .2, roughness: .85 });

const views = {};
let geometry = [];
let policies = [];

/* ---------- robot construction (driven by the real MuJoCo geom spec) ---------- */

function segmentMesh(shape, kind, accent) {
  const group = new THREE.Group();
  const radius = shape.size[0];
  const half = shape.size[1];

  const taper = kind === "shin" ? 0.5 : 0.82;
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * taper, radius * 0.95, half * 2, 20),
    kind === "shin" ? accent : CHASSIS
  );
  body.rotation.x = Math.PI / 2; // three cylinders run along Y; mujoco capsules along local Z
  group.add(body);

  const hub = new THREE.Mesh(new THREE.SphereGeometry(radius * 1.12, 20, 14), JOINT);
  hub.position.z = -half;
  group.add(hub);

  if (kind === "shin") {
    const foot = new THREE.Mesh(new THREE.SphereGeometry(radius * 0.72, 18, 12), FOOT);
    foot.position.z = half;
    foot.scale.set(1.25, 1.25, .8);
    group.add(foot);
  }
  return group;
}

function torsoMesh(shape, accent) {
  const group = new THREE.Group();
  const r = shape.size[0];

  const shell = new THREE.Mesh(new THREE.SphereGeometry(r, 36, 24), CHASSIS);
  shell.scale.set(1.18, 1.05, .72);
  group.add(shell);

  const deck = new THREE.Mesh(new THREE.BoxGeometry(r * 1.15, r * .95, r * .34), PLATE);
  deck.position.z = r * .5;
  group.add(deck);

  const dome = new THREE.Mesh(new THREE.SphereGeometry(r * .3, 20, 14), accent);
  dome.position.set(0, 0, r * .72);
  group.add(dome);

  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(new THREE.SphereGeometry(r * .1, 12, 10), PLATE);
    eye.position.set(side * r * .42, r * .82, r * .18);
    group.add(eye);
  }
  return group;
}

function buildRobot(accent) {
  const root = new THREE.Group();
  const bodies = new Map();

  for (const shape of geometry) {
    let holder = bodies.get(shape.body);
    if (!holder) {
      holder = new THREE.Group();
      bodies.set(shape.body, holder);
      root.add(holder);
    }
    // body 0 is the torso; legs repeat shoulder -> thigh -> shin
    const kindIndex = shape.body === 0 ? -1 : (shape.body - 1) % 3;
    const kind = kindIndex === 2 ? "shin" : kindIndex === 1 ? "thigh" : "shoulder";

    let mesh;
    if (shape.type === GEOM_SPHERE && shape.body === 0) mesh = torsoMesh(shape, accent);
    else if (shape.type === GEOM_CAPSULE) mesh = segmentMesh(shape, kind, accent);
    else mesh = new THREE.Mesh(new THREE.SphereGeometry(shape.size[0], 18, 12), CHASSIS);

    mesh.position.fromArray(shape.pos);
    const [w, x, y, z] = shape.quat;
    mesh.quaternion.copy(new THREE.Quaternion(x, y, z, w));
    holder.add(mesh);
  }

  root.traverse(node => {
    if (node.isMesh) { node.castShadow = true; node.receiveShadow = true; }
  });
  return { root, bodies };
}

/* ---------- scene ---------- */

function createView(canvasId, accentColor) {
  const canvas = document.getElementById(canvasId);
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf4f5f7);
  scene.fog = new THREE.Fog(0xf4f5f7, 12, 34);

  const camera = new THREE.PerspectiveCamera(42, 1, .1, 200);
  camera.position.set(-1.9, 1.5, 2.6);

  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true;
  controls.enablePan = false;
  controls.minDistance = 1.8;
  controls.maxDistance = 14;
  controls.maxPolarAngle = Math.PI / 2 - .04;

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bec7, 1.5));
  const key = new THREE.DirectionalLight(0xffffff, 2.1);
  key.position.set(4, 8, 5);
  key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  const cam = key.shadow.camera;
  cam.left = -4; cam.right = 4; cam.top = 4; cam.bottom = -4; cam.far = 30;
  scene.add(key, key.target);

  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(400, 400),
    new THREE.ShadowMaterial({ opacity: .22 })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  scene.add(ground);

  const grid = new THREE.GridHelper(400, 400, 0xc3c8d0, 0xdfe3e8);
  grid.material.transparent = true;
  grid.material.opacity = .85;
  scene.add(grid);

  // Lane markers every 5 m so forward progress is obvious under a chase camera.
  const markers = new THREE.Group();
  for (let x = -20; x <= 300; x += 5) {
    const bar = new THREE.Mesh(
      new THREE.BoxGeometry(.06, .004, 3.2),
      new THREE.MeshBasicMaterial({ color: 0xa9b0ba })
    );
    bar.position.set(x, .004, 0);
    markers.add(bar);
  }
  scene.add(markers);

  // MuJoCo is Z-up; rotate the robot container into three.js Y-up.
  const world = new THREE.Group();
  world.rotation.x = -Math.PI / 2;
  scene.add(world);

  const accent = new THREE.MeshStandardMaterial({ color: accentColor, metalness: .5, roughness: .35 });
  const robot = buildRobot(accent);
  world.add(robot.root);

  return {
    renderer, scene, camera, controls, key, world, robot,
    target: { pos: [], quat: [] },
    ready: false,
  };
}

function applyFrame(view, panel) {
  view.target.pos = panel.xpos;
  view.target.quat = panel.xquat;
  view.ready = true;
}

const tmpPos = new THREE.Vector3();
const tmpQuat = new THREE.Quaternion();

function updateView(view, dt) {
  if (!view.ready) return;
  const smooth = 1 - Math.exp(-dt * 22);

  view.robot.bodies.forEach((holder, index) => {
    tmpPos.fromArray(view.target.pos, index * 3);
    const q = view.target.quat;
    tmpQuat.set(q[index * 4 + 1], q[index * 4 + 2], q[index * 4 + 3], q[index * 4]);
    holder.position.lerp(tmpPos, smooth);
    holder.quaternion.slerp(tmpQuat, smooth);
  });

  // chase camera: keep the orbit target on the torso at a constant offset
  const torso = view.robot.bodies.get(0);
  view.world.localToWorld(tmpPos.copy(torso.position));
  const offset = view.camera.position.clone().sub(view.controls.target);
  view.controls.target.copy(tmpPos);
  view.camera.position.copy(tmpPos).add(offset);
  view.key.position.set(tmpPos.x + 4, 8, tmpPos.z + 5);
  view.key.target.position.copy(tmpPos);
  view.key.target.updateMatrixWorld();
  view.controls.update();
}

function resize(view) {
  const canvas = view.renderer.domElement;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w * devicePixelRatio || canvas.height !== h * devicePixelRatio) {
    view.renderer.setSize(w, h, false);
    view.camera.aspect = w / h;
    view.camera.updateProjectionMatrix();
  }
}

/* ---------- stats, chart, controls ---------- */

const PPO_COLOR = "#3b82f6";
const SAC_COLOR = "#0f9d6e";
const fmt = v => Number(v).toFixed(2);
const big = v => Math.round(v).toLocaleString();

let latestTrainers = {};
let history = { ppo: [], sac: [] };
const selects = {};
const serverValue = {};
let unloading = false;
addEventListener("pagehide", () => (unloading = true));

function trainedStepsFor(policyId) {
  if (policyId.startsWith("live_")) {
    const stats = latestTrainers[policyId.slice(5)];
    return stats ? stats.steps : 0;
  }
  if (policyId === "ppo") return 1000000;
  if (policyId.startsWith("sac_")) return Number(policyId.split("_")[1]);
  return 0;
}

function meanRewardFor(policyId) {
  if (!policyId.startsWith("live_")) return null;
  const stats = latestTrainers[policyId.slice(5)];
  return stats ? stats.reward : null;
}

function setStats(side, panel) {
  const select = selects[side];
  serverValue[side] = panel.policy;  // server is the source of truth
  if (select && select.value !== panel.policy && document.activeElement !== select) {
    select.value = panel.policy;
  }
  const mean = meanRewardFor(panel.policy);
  document.getElementById(side + "Trained").textContent = big(trainedStepsFor(panel.policy));
  document.getElementById(side + "Mean").textContent = mean === null ? "—" : big(mean);
  document.getElementById(side + "Return").textContent = big(panel.return);
  document.getElementById(side + "Distance").textContent = fmt(panel.distance);
}

function drawCurve() {
  const canvas = document.getElementById("curve");
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * devicePixelRatio;
  canvas.height = h * devicePixelRatio;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const series = [
    { points: history.ppo || [], color: PPO_COLOR },
    { points: history.sac || [], color: SAC_COLOR },
  ];
  const all = series.flatMap(s => s.points);
  const pad = { l: 52, r: 12, t: 12, b: 24 };

  ctx.font = "11px -apple-system, Segoe UI, Roboto, Arial";
  ctx.fillStyle = "#6b7280";
  if (all.length < 2) {
    ctx.fillText("collecting episodes…", pad.l, h / 2);
    return;
  }

  // both lanes train simultaneously, so wall-clock is the fair shared axis
  const maxMinutes = Math.max(...all.map(p => p.minutes), 0.1);
  let lo = Math.min(...all.map(p => p.reward));
  let hi = Math.max(...all.map(p => p.reward));
  if (hi - lo < 1) { hi += 1; lo -= 1; }
  const x = m => pad.l + (m / maxMinutes) * (w - pad.l - pad.r);
  const y = r => h - pad.b - ((r - lo) / (hi - lo)) * (h - pad.t - pad.b);

  ctx.strokeStyle = "#eceef1";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const value = lo + (hi - lo) * (i / 4);
    const yy = Math.round(y(value)) + .5;
    ctx.beginPath(); ctx.moveTo(pad.l, yy); ctx.lineTo(w - pad.r, yy); ctx.stroke();
    ctx.fillText(Math.round(value).toLocaleString(), 6, yy + 4);
  }
  ctx.fillText("0 min", pad.l, h - 8);
  ctx.fillText(maxMinutes.toFixed(1) + " min", w - pad.r - 44, h - 8);

  ctx.lineWidth = 1.8;
  for (const s of series) {
    if (s.points.length < 2) continue;
    ctx.strokeStyle = s.color;
    ctx.beginPath();
    s.points.forEach((p, i) =>
      i ? ctx.lineTo(x(p.minutes), y(p.reward)) : ctx.moveTo(x(p.minutes), y(p.reward)));
    ctx.stroke();
  }
}

function updateLeader() {
  const ppo = latestTrainers.ppo, sac = latestTrainers.sac;
  const node = document.getElementById("leader");
  if (!ppo || !sac || ppo.reward === null || sac.reward === null) {
    node.textContent = "";
    return;
  }
  const ahead = ppo.reward >= sac.reward ? "PPO" : "SAC";
  node.textContent = `${ahead} ahead by ${big(Math.abs(ppo.reward - sac.reward))} reward`;
}

async function postConfig(payload) {
  await fetch("/api/config", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

function buildSelect(side, selected) {
  const select = document.createElement("select");
  select.className = "policy";
  select.setAttribute("aria-label", side + " policy");
  select.autocomplete = "off";
  for (const policy of policies) {
    const option = document.createElement("option");
    option.value = policy.id;
    option.textContent = policy.label;
    select.appendChild(option);
  }
  select.value = selected;
  serverValue[side] = selected;

  // Chrome fires change on these dropdowns by itself (form restore / autofill),
  // which silently switched policies. Only a real gesture counts.
  let gesture = false;
  select.addEventListener("pointerdown", () => (gesture = true));
  select.addEventListener("keydown", () => (gesture = true));
  select.addEventListener("wheel", () => select.blur());
  select.addEventListener("change", () => {
    // a page being unloaded commits its open dropdown and fires change; that
    // stale value must not reach the server the next page is about to use
    if (!gesture || unloading) {
      select.value = serverValue[side];
      return;
    }
    gesture = false;
    serverValue[side] = select.value;
    postConfig({ [side]: select.value });
    select.blur();
  });

  document.getElementById(side + "PolicyHost").replaceWith(select);
  selects[side] = select;
}

async function refreshHistory() {
  try {
    history = await fetch("/api/history").then(r => r.json());
    drawCurve();
  } catch { /* server restarting */ }
}

async function init() {
  const setup = await fetch("/api/setup").then(r => r.json());
  policies = setup.policies;
  geometry = setup.geometry;
  if (!setup.allowRestart) {
    document.getElementById("restart").hidden = true;
  }

  views.left = createView("leftCanvas", 0x3b82f6);
  views.right = createView("rightCanvas", 0x0f9d6e);

  buildSelect("left", setup.left);
  buildSelect("right", setup.right);
  await postConfig({ speed: 1, paused: false, reset: true });

  document.getElementById("speed").addEventListener("change", e => postConfig({ speed: Number(e.target.value) }));
  document.getElementById("reset").addEventListener("click", () => postConfig({ reset: true }));

  const restart = document.getElementById("restart");
  restart.addEventListener("click", async () => {
    if (!confirm("Throw away both networks and start training from scratch?")) return;
    restart.disabled = true;
    restart.textContent = "Restarting…";
    history = { ppo: [], sac: [] };
    drawCurve();
    await postConfig({ restartTraining: true });
    restart.disabled = false;
    restart.textContent = "Restart training";
  });

  const pause = document.getElementById("pause");
  let paused = false;
  pause.addEventListener("click", () => {
    paused = !paused;
    pause.textContent = paused ? "Resume" : "Pause";
    postConfig({ paused });
  });

  // a focused <select> swallows wheel events and silently changes policy while scrolling
  document.querySelectorAll("select").forEach(select => {
    select.addEventListener("wheel", () => select.blur());
    select.addEventListener("change", () => select.blur());
  });

  const status = document.getElementById("status");
  const source = new EventSource("/api/stream");
  source.onerror = () => status.textContent = "disconnected";
  source.onmessage = event => {
    const frame = JSON.parse(event.data);
    latestTrainers = frame.trainers || {};
    for (const panel of frame.panels) {
      applyFrame(views[panel.id], panel);
      setStats(panel.id, panel);
    }
    updateLeader();
    const ppo = latestTrainers.ppo, sac = latestTrainers.sac;
    status.textContent = frame.paused
      ? "paused"
      : `training · PPO ${big(ppo ? ppo.steps : 0)} steps · SAC ${big(sac ? sac.steps : 0)} steps`;
  };

  refreshHistory();
  setInterval(refreshHistory, 3000);
  addEventListener("resize", drawCurve);

  let last = performance.now();
  (function render(now) {
    const dt = Math.min((now - last) / 1000, .1);
    last = now;
    for (const view of Object.values(views)) {
      resize(view);
      updateView(view, dt);
      view.renderer.render(view.scene, view.camera);
    }
    requestAnimationFrame(render);
  })(last);
}

init().catch(error => {
  console.error(error);
  document.getElementById("status").textContent = "failed to load";
});
