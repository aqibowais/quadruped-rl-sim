"""Planner success per walker checkpoint, both mazes. python maze_sweep.py"""
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from maze_env import MAZES, Maze, make_maze_env, run
from maze_eval import load_nav


def score(path: str, episodes: int = 6) -> tuple[str, dict]:
    nav = load_nav(Path(path))
    out = {}
    for name, grid in MAZES.items():
        m = Maze(name, grid)
        env = make_maze_env(m, max_steps=2400)
        rs = [run(m, env, nav, "planner", seed=s) for s in range(episodes)]
        ok = [r for r in rs if r["success"]]
        out[name] = (len(ok), episodes, float(np.mean([r["seconds"] for r in ok])) if ok else None, float(np.mean([r["recoveries"] for r in rs])))
    return Path(path).stem, out


if __name__ == "__main__":
    paths = sorted(str(p) for p in Path("models/nav").glob("nav_*k.zip")) + ["models/nav/nav_final.zip"]
    with Pool(len(paths)) as pool:
        for name, out in pool.map(score, paths):
            print(name, {k: f"{a}/{b} {t and round(t, 1)}s rec {round(r, 1)}" for k, (a, b, t, r) in out.items()}, flush=True)
