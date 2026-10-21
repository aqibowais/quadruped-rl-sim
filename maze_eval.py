"""Run every pilot through the mazes and print success, time and wall contact.

    python maze_eval.py --nav models/nav/nav_final.zip --episodes 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from maze_env import MAZES, Maze, make_maze_env, run
from nav_env import make_directional_env

ROOT = Path(__file__).resolve().parent


def load_nav(path: Path):
    model = PPO.load(str(path), device="cpu")
    vn_path = path.with_name(path.stem + "_vecnormalize.pkl")
    norm = VecNormalize.load(str(vn_path), DummyVecEnv([lambda: make_directional_env()]))
    norm.training = False
    return lambda obs: model.predict(norm.normalize_obs(obs), deterministic=True)[0]


def load_forward():
    model = SAC.load(str(ROOT / "models" / "sac_ant.zip"), device="cpu")
    return lambda obs: model.predict(obs[:-2], deterministic=True)[0]


def pilots(nav_path: Path):
    nav = load_nav(nav_path)
    rng = np.random.default_rng(0)
    return {
        "random": (lambda obs: rng.uniform(-1, 1, 8), "none"),
        "forward": (load_forward(), "none"),
        "beeline": (nav, "beeline"),
        "planner": (nav, "planner"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--nav", type=Path, default=ROOT / "models" / "nav" / "nav_final.zip")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=2400)
    args = p.parse_args()
    table = pilots(args.nav)
    for name, grid in MAZES.items():
        maze = Maze(name, grid)
        env = make_maze_env(maze, max_steps=args.max_steps)
        for pilot, (act, steer) in table.items():
            rs = [run(maze, env, act, steer, seed=s) for s in range(args.episodes)]
            ok = [r for r in rs if r["success"]]
            print(
                f"{name:10s} {pilot:8s} success {len(ok)}/{len(rs)}  "
                f"time {np.mean([r['seconds'] for r in ok]) if ok else float('nan'):5.1f}s  "
                f"closest {np.mean([r['closest_m'] for r in rs]):5.1f}m  "
                f"wall contact {np.mean([r['wall_contact_s'] for r in rs]):4.1f}s  falls {sum(r['fell'] for r in rs)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
