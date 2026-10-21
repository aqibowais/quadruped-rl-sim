"""Record MuJoCo rollouts of the saved policies for the static web app.

    python export_web.py

Every number the site shows comes from this script running the real Ant-v4
simulation with the saved checkpoints. The browser only replays the recorded
body transforms, so the site needs no Python, MuJoCo or GPU at runtime.

Writes web/public/data/:
    scene.json        Ant geometry (geom type, size, local pose, parent body)
    runs.json         one entry per recorded rollout, with evaluation metrics
    runs/<id>.bin     float32 body poses, per-step metrics, foot contacts, torques
    curves.json       evaluation curves from training (logs/*/eval/evaluations.npz)
    robustness.json   mean return over a torso-mass x floor-friction grid
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from randomize_env import make_ant_env

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
OUT = ROOT / "web" / "public" / "data"

FOOT_GEOMS = ("left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom")
FOOT_LABELS = ("FL", "FR", "BL", "BR")
SAC_STEPS = (50_000, 100_000, 200_000, 400_000, 700_000, 1_000_000)
MASS_GRID = (0.6, 0.8, 1.0, 1.2, 1.4)
FRICTION_GRID = (0.4, 0.7, 1.0, 1.3, 1.6)

Agent = Callable[[np.ndarray], np.ndarray]


@dataclass
class Physics:
    mass: float = 1.0
    friction: float = 1.0
    pushes: tuple[float, ...] = ()  # seconds at which a lateral shove is applied
    push_force: float = 0.0


def load_agent(kind: str, path: Path, vecnorm: Path | None = None) -> Agent:
    model = (PPO if kind == "ppo" else SAC).load(str(path), device="cpu")
    norm = None
    if vecnorm is not None and vecnorm.exists():
        norm = VecNormalize.load(str(vecnorm), DummyVecEnv([lambda: make_ant_env()]))
        norm.training = False

    def act(obs: np.ndarray) -> np.ndarray:
        x = norm.normalize_obs(obs) if norm is not None else obs
        action, _ = model.predict(x, deterministic=True)
        return action

    return act


def random_agent(seed: int) -> Agent:
    rng = np.random.default_rng(seed)
    return lambda obs: rng.uniform(-1.0, 1.0, size=8).astype(np.float32)


class Rollout:
    """One Ant-v4 episode with optional physics edits, recording what the site draws."""

    def __init__(self) -> None:
        self.env = make_ant_env(max_episode_steps=1000)
        self.mj = self.env.unwrapped
        self.model = self.mj.model
        self.data = self.mj.data
        self.orig_mass = self.model.body_mass.copy()
        self.orig_inertia = self.model.body_inertia.copy()
        self.orig_friction = self.model.geom_friction.copy()
        self.torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.floor = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.feet = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS]
        # Ant-v4 runs 5 physics substeps per env step. Sampling contacts only
        # after the last one misses short foot strikes, so step one at a time
        # and keep every foot that touched during the env step. mj_step with
        # nstep=n is n single steps, so the physics is unchanged.
        self.step_contacts = 0
        mj = self.mj

        def substep(ctrl, n_frames):
            mj.data.ctrl[:] = ctrl
            for _ in range(n_frames):
                mujoco.mj_step(mj.model, mj.data)
                self.step_contacts |= self.contacts()
            mujoco.mj_rnePostConstraint(mj.model, mj.data)

        mj._step_mujoco_simulation = substep

    def apply(self, physics: Physics) -> None:
        # Same edits as DomainRandomizationWrapper, with one scale for every geom.
        self.model.body_mass[:] = self.orig_mass
        self.model.body_inertia[:] = self.orig_inertia
        self.model.body_mass[self.torso] = self.orig_mass[self.torso] * physics.mass
        self.model.body_inertia[self.torso] = self.orig_inertia[self.torso] * physics.mass
        self.model.geom_friction[:] = self.orig_friction * np.array([physics.friction, physics.friction, physics.friction])

    def contacts(self) -> int:
        bits = 0
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            pair = (c.geom1, c.geom2)
            if self.floor not in pair:
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            if other in self.feet:
                bits |= 1 << self.feet.index(other)
        return bits

    def run(self, agent: Agent, seed: int, physics: Physics, record: bool) -> dict:
        self.apply(physics)
        obs, _ = self.env.reset(seed=seed)
        x0 = float(self.data.qpos[0])
        total = 0.0
        ctrl = 0.0
        poses, metrics, contacts, torques, push_frames = [], [], [], [], []
        push_steps = {int(round(t / self.mj.dt)) for t in physics.pushes}
        step = 0
        done = False
        fell = False
        while not done:
            push_active = any(0 <= step - p < 2 for p in push_steps)
            self.data.xfrc_applied[self.torso, :3] = (0.0, physics.push_force if push_active else 0.0, 0.0)
            if push_active and step in push_steps:
                push_frames.append(step)
            action = np.clip(agent(obs), -1.0, 1.0)
            self.step_contacts = 0
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            ctrl += float(np.sum(np.square(action)))
            done = terminated or truncated
            fell = fell or terminated
            if record:
                poses.append(np.concatenate([self.data.xpos[1:], self.data.xquat[1:]], axis=1).ravel())
                metrics.append([float(info.get("x_velocity", 0.0)), float(self.data.qpos[2]), float(reward), total])
                contacts.append(self.step_contacts)
                torques.append(np.round(action * 127).astype(np.int8))
            step += 1
        self.data.xfrc_applied[:] = 0.0
        result = {
            "return": total,
            "length": step,
            "fell": fell,
            "distance": float(self.data.qpos[0]) - x0,
            "speed": (float(self.data.qpos[0]) - x0) / (step * self.mj.dt),
            "effort": ctrl / step,
        }
        if record:
            result["poses"] = np.asarray(poses, dtype=np.float32)
            result["metrics"] = np.asarray(metrics, dtype=np.float32)
            result["contacts"] = np.asarray(contacts, dtype=np.uint8)
            result["torques"] = np.asarray(torques, dtype=np.int8)
            result["pushes"] = push_frames
        return result


def scene(roll: Rollout) -> dict:
    m = roll.model
    geoms = []
    for i in range(m.ngeom):
        body = int(m.geom_bodyid[i])
        if body == 0:
            continue
        geoms.append(
            {
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i),
                "body": body - 1,
                "type": {2: "sphere", 3: "capsule"}.get(int(m.geom_type[i]), "capsule"),
                "size": np.round(m.geom_size[i], 5).tolist(),
                "pos": np.round(m.geom_pos[i], 5).tolist(),
                "quat": np.round(m.geom_quat[i], 6).tolist(),
                "foot": FOOT_LABELS[FOOT_GEOMS.index(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i))]
                if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) in FOOT_GEOMS
                else None,
            }
        )
    return {"bodies": int(m.nbody - 1), "dt": float(roll.mj.dt), "geoms": geoms, "feet": list(FOOT_LABELS)}


def write_bin(path: Path, rec: dict) -> dict:
    parts = [("poses", rec["poses"]), ("metrics", rec["metrics"]), ("contacts", rec["contacts"]), ("torques", rec["torques"])]
    layout = {}
    offset = 0
    with path.open("wb") as fh:
        for name, arr in parts:
            pad = (-offset) % 4  # keep float32 views aligned
            fh.write(b"\0" * pad)
            offset += pad
            raw = np.ascontiguousarray(arr).tobytes()
            layout[name] = {"offset": offset, "dtype": str(arr.dtype), "shape": list(arr.shape)}
            fh.write(raw)
            offset += len(raw)
    return layout


def gait(contacts: np.ndarray) -> dict:
    if len(contacts) == 0:
        return {}
    duty = {label: float(np.mean((contacts >> i) & 1)) for i, label in enumerate(FOOT_LABELS)}
    stance = np.array([bin(int(c)).count("1") for c in contacts])
    return {"duty": duty, "mean_feet_down": float(stance.mean()), "airborne": float(np.mean(stance == 0))}


def summarize(results: list[dict]) -> dict:
    ret = np.array([r["return"] for r in results])
    return {
        "episodes": len(results),
        "return_mean": float(ret.mean()),
        "return_std": float(ret.std()),
        "returns": [round(float(v), 2) for v in ret],
        "falls": int(sum(r["fell"] for r in results)),
        "length_mean": float(np.mean([r["length"] for r in results])),
        "speed_mean": float(np.mean([r["speed"] for r in results])),
        "distance_mean": float(np.mean([r["distance"] for r in results])),
        "effort_mean": float(np.mean([r["effort"] for r in results])),
    }


def curves() -> dict:
    out = {}
    for run in sorted(LOGS.glob("*/eval/evaluations.npz")):
        name = run.parent.parent.name
        if "smoke" in name or "partial" in name:
            continue
        d = np.load(run)
        res = d["results"]
        out[name] = {
            "timesteps": d["timesteps"].astype(int).tolist(),
            "mean": np.round(res.mean(axis=1), 2).tolist(),
            "std": np.round(res.std(axis=1), 2).tolist(),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="evaluation episodes per run")
    parser.add_argument("--grid-episodes", type=int, default=3)
    parser.add_argument("--skip-grid", action="store_true")
    args = parser.parse_args()

    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    roll = Rollout()
    (OUT / "scene.json").write_text(json.dumps(scene(roll)), encoding="utf-8")

    sac_final = MODELS / "sac_ant.zip"
    ppo_tuned = sorted(MODELS.glob("ppo_tuned_s*_ant.zip"))
    agents: dict[str, tuple[str, Callable[[int], Agent]]] = {
        "random": ("Random actions", random_agent),
        "ppo_legacy": ("PPO, first run", lambda s: load_agent("ppo", MODELS / "ppo_ant.zip", MODELS / "ppo_vecnormalize.pkl")),
    }
    for path in ppo_tuned:
        run = path.stem[: -len("_ant")]
        seed = run.rsplit("_s", 1)[-1]
        agents[run] = (f"PPO, tuned (seed {seed})", lambda s, p=path, r=run: load_agent("ppo", p, MODELS / f"{r}_vecnormalize.pkl"))
    for steps in SAC_STEPS:
        ckpt = MODELS / "checkpoints" / "sac" / f"sac_{steps}_steps.zip"
        if ckpt.exists():
            agents[f"sac_{steps // 1000}k"] = (f"SAC at {steps // 1000}k steps", lambda s, p=ckpt: load_agent("sac", p))

    plan = [(key, label, factory, Physics(), "progression") for key, (label, factory) in agents.items()]
    finals = [("sac", lambda s: load_agent("sac", sac_final), "SAC")]
    if ppo_tuned:
        finals.append(("ppo", agents[ppo_tuned[0].stem[: -len("_ant")]][1], "PPO"))
    stress = [
        ("heavy", "Torso 1.4x mass", Physics(mass=1.4)),
        ("light", "Torso 0.6x mass", Physics(mass=0.6)),
        ("ice", "Floor friction 0.4x", Physics(friction=0.4)),
        ("grip", "Floor friction 1.6x", Physics(friction=1.6)),
        ("push", "Lateral shoves at 10, 20, 30 s", Physics(pushes=(10.0, 20.0, 30.0), push_force=120.0)),
    ]
    for algo, factory, algo_label in finals:
        for key, label, physics in stress:
            plan.append((f"{algo}_{key}", f"{algo_label}: {label}", factory, physics, "stress"))

    entries = []
    seeds = list(range(100, 100 + args.episodes))
    for key, label, factory, physics, group in plan:
        t0 = time.perf_counter()
        results = []
        recorded = None
        for i, seed in enumerate(seeds):
            r = roll.run(factory(seed), seed, physics, record=(i == 0))
            if i == 0:
                recorded = r
            results.append(r)
        layout = write_bin(OUT / "runs" / f"{key}.bin", recorded)
        entry = {
            "id": key,
            "label": label,
            "group": group,
            "physics": {"mass": physics.mass, "friction": physics.friction, "push_force": physics.push_force},
            "frames": int(recorded["length"]),
            "pushes": recorded["pushes"],
            "layout": layout,
            "recorded": {k: recorded[k] for k in ("return", "length", "fell", "distance", "speed")},
            "gait": gait(recorded["contacts"]),
            "eval": summarize(results),
        }
        entries.append(entry)
        s = entry["eval"]
        print(f"{key:22s} return {s['return_mean']:8.1f} ± {s['return_std']:6.1f}  falls {s['falls']}/{s['episodes']}  "
              f"speed {s['speed_mean']:5.2f} m/s  ({time.perf_counter() - t0:.0f}s)", flush=True)

    (OUT / "runs.json").write_text(json.dumps({"seeds": seeds, "runs": entries}, indent=1), encoding="utf-8")
    (OUT / "curves.json").write_text(json.dumps(curves()), encoding="utf-8")

    if not args.skip_grid:
        grid = {"mass": list(MASS_GRID), "friction": list(FRICTION_GRID), "episodes": args.grid_episodes, "policies": {}}
        for algo, factory, algo_label in finals:
            agent = factory(0)
            cells = []
            for mass in MASS_GRID:
                row = []
                for fr in FRICTION_GRID:
                    rs = [roll.run(agent, 500 + k, Physics(mass=mass, friction=fr), record=False) for k in range(args.grid_episodes)]
                    row.append({"return": round(float(np.mean([r["return"] for r in rs])), 1),
                                "falls": int(sum(r["fell"] for r in rs))})
                cells.append(row)
                print(f"grid {algo} mass {mass}: {[c['return'] for c in row]}", flush=True)
            grid["policies"][algo] = {"label": algo_label, "cells": cells}
        (OUT / "robustness.json").write_text(json.dumps(grid, indent=1), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
