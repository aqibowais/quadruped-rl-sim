"""Record the steering and maze stages for the web app.

    python export_nav.py

Writes web/public/data/nav/:
    nav.json              mazes (walls, path, start, goal), stats, curve, file layouts
    steer_<pilot>.bin     open field, the commanded heading changes every 7 s
    maze_<maze>_<pilot>.bin
Each .bin holds float32 body poses, per-step metrics, foot contacts, int8
torques, the commanded heading and the floor track.

Reported numbers use seeds 100-109. The walker checkpoint was chosen with
maze_sweep.py on seeds 0-5, so the reporting seeds were not used to pick it.
"""

from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "web" / "public" / "data" / "nav"
NAV = ROOT / "models" / "nav" / "nav_6000k.zip"
SEEDS = list(range(100, 110))
STEER_SCHEDULE = [0, 90, 180, 270, 45, 225, 135, 315]  # degrees, 7 s each
STEER_STEPS = 140


def write_bin(path: Path, parts: list[tuple[str, np.ndarray]]) -> dict:
    layout = {}
    offset = 0
    with path.open("wb") as fh:
        for name, arr in parts:
            pad = (-offset) % 4
            fh.write(b"\0" * pad)
            offset += pad
            raw = np.ascontiguousarray(arr).tobytes()
            layout[name] = {"offset": offset, "dtype": str(arr.dtype), "shape": list(arr.shape)}
            fh.write(raw)
            offset += len(raw)
    return layout


def pilots():
    from maze_eval import load_forward, load_nav

    nav = load_nav(NAV)
    rng = np.random.default_rng(0)
    return {
        "random": (lambda obs: rng.uniform(-1, 1, 8), "none"),
        "forward": (load_forward(), "none"),
        "beeline": (nav, "beeline"),
        "planner": (nav, "planner"),
    }


# ---------------------------------------------------------------- steering


def steer_episode(act, seed: int, schedule: list[float], record: bool) -> dict:
    from maze_env import foot_bits
    from nav_env import make_directional_env

    env = make_directional_env(seed=seed, max_episode_steps=STEER_STEPS * len(schedule))
    obs, _ = env.reset(seed=seed)
    data = env.mj.data
    frames, metrics, contacts, torques, commands, track = [], [], [], [], [], []
    progress, lateral = [], []
    total = 0.0
    fell = False
    for k in range(STEER_STEPS * len(schedule)):
        env.set_command(np.radians(schedule[k // STEER_STEPS]))
        obs = np.concatenate([obs[:-2], env._command_obs()])
        a = np.clip(act(obs), -1, 1)
        obs, r, te, tu, info = env.step(a)
        total += r
        progress.append(info["progress"])
        lateral.append(info["lateral"])
        if record:
            frames.append(np.concatenate([data.xpos[1:], data.xquat[1:]], axis=1).ravel())
            metrics.append([info["progress"], float(data.qpos[2]), float(r), total])
            contacts.append(foot_bits(env.mj.model, data))
            torques.append(np.round(a * 127).astype(np.int8))
            commands.append(env.command)
            track.append(data.qpos[:2].copy())
        if te:
            fell = True
            break
        if tu:
            break
    out = {"speed": float(np.mean(progress)), "lateral": float(np.mean(lateral)), "fell": fell, "steps": len(progress)}
    if record:
        out["parts"] = [
            ("poses", np.asarray(frames, np.float32)),
            ("metrics", np.asarray(metrics, np.float32)),
            ("contacts", np.asarray(contacts, np.uint8)),
            ("torques", np.asarray(torques, np.int8)),
            ("commands", np.asarray(commands, np.float32)),
            ("track", np.asarray(track, np.float32)),
        ]
    return out


def steer_job(pilot: str) -> tuple[str, dict]:
    act = pilots()[pilot][0]
    rec = steer_episode(act, 100, STEER_SCHEDULE, record=True)
    layout = write_bin(OUT / f"steer_{pilot}.bin", rec.pop("parts"))
    evals = []
    for s in SEEDS:
        rng = np.random.default_rng(s)
        schedule = list(rng.uniform(0, 360, size=6))
        evals.append(steer_episode(act, s, schedule, record=False))
    return pilot, {
        "layout": layout,
        "frames": rec["steps"],
        "recorded": {k: rec[k] for k in ("speed", "lateral", "fell")},
        "eval": {
            "episodes": len(evals),
            "speed_along_command": float(np.mean([e["speed"] for e in evals])),
            "sideways_drift": float(np.mean([e["lateral"] for e in evals])),
            "falls": int(sum(e["fell"] for e in evals)),
        },
    }


# ---------------------------------------------------------------- mazes


def maze_job(args: tuple[str, str]) -> tuple[str, str, dict]:
    from maze_env import MAZES, Maze, make_maze_env, run

    maze_name, pilot = args
    act, steer = pilots()[pilot]
    maze = Maze(maze_name, MAZES[maze_name])
    env = make_maze_env(maze, max_steps=2400)
    results = [run(maze, env, act, steer, seed=s) for s in SEEDS]
    # The clip: the first reporting seed that shows the pilot's typical outcome.
    typical = max(set(r["success"] for r in results), key=lambda v: sum(r["success"] == v for r in results))
    clip_seed = next(s for s, r in zip(SEEDS, results) if r["success"] == typical)
    rec = run(maze, env, act, steer, seed=clip_seed, record=True)
    parts = [
        ("poses", rec["poses"]),
        ("metrics", rec["metrics"]),
        ("contacts", rec["contacts"]),
        ("torques", rec["torques"]),
        ("commands", rec["commands"]),
        ("track", rec["track"]),
    ]
    layout = write_bin(OUT / f"maze_{maze_name}_{pilot}.bin", parts)
    ok = [r for r in results if r["success"]]
    stats = {
        "episodes": len(results),
        "success": len(ok),
        "seconds": float(np.mean([r["seconds"] for r in ok])) if ok else None,
        "falls": int(sum(r["fell"] for r in results)),
        "closest_m": float(np.mean([r["closest_m"] for r in results])),
        "wall_contact_s": float(np.mean([r["wall_contact_s"] for r in results])),
        "recoveries": float(np.mean([r["recoveries"] for r in results])),
    }
    clip = {"seed": clip_seed, "success": rec["success"], "seconds": rec["seconds"], "frames": int(len(rec["poses"]))}
    print(f"{maze_name:10s} {pilot:8s} success {len(ok)}/{len(results)}  clip seed {clip_seed}", flush=True)
    return maze_name, pilot, {"layout": layout, "clip": clip, "eval": stats}


def main() -> None:
    from maze_env import CELL, GOAL_RADIUS, MAZES, Maze

    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(4) as pool:
        steer = dict(pool.map(steer_job, ["forward", "planner"]))
        mazes_out = pool.map(maze_job, [(m, p) for m in MAZES for p in ("random", "forward", "beeline", "planner")])
    mazes = {}
    for name, grid in MAZES.items():
        m = Maze(name, grid)
        mazes[name] = {
            "grid": grid,
            "cell": CELL,
            "goal_radius": GOAL_RADIUS,
            "walls": [list(map(float, b)) for b in m.boxes],
            "path": m.path.tolist(),
            "start": m.start.tolist(),
            "goal": m.goal.tolist(),
            "pilots": {},
        }
    for maze_name, pilot, data in mazes_out:
        mazes[maze_name]["pilots"][pilot] = data
    curve = []
    prog = ROOT / "logs" / "nav" / "progress.jsonl"
    if prog.exists():
        curve = [json.loads(l) for l in prog.read_text(encoding="utf-8").splitlines() if l.strip()]
    nav = {
        "walker": NAV.stem,
        "seeds": SEEDS,
        "steer": {"schedule_deg": STEER_SCHEDULE, "seconds_each": STEER_STEPS * 0.05, "pilots": {"forward": steer["forward"], "steerable": steer["planner"]}},
        "mazes": mazes,
        "curve": curve,
    }
    (OUT / "nav.json").write_text(json.dumps(nav, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
