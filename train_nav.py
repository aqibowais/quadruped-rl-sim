"""Train the direction-commanded Ant (nav_env.DirectionalAnt) with PPO.

    python train_nav.py --timesteps 8000000

Saves models/nav/nav_<steps>.zip and the matching VecNormalize statistics
every --save-every steps, plus logs/nav/progress.jsonl with the training
return and how well the policy follows commands.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from nav_env import make_directional_env

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models" / "nav"
LOGS = ROOT / "logs" / "nav"


def factory(rank: int, seed: int):
    def _init():
        env = Monitor(make_directional_env(seed=seed + rank), info_keywords=("progress",))
        env.reset(seed=seed + rank)
        return env

    return _init


class Snapshots(BaseCallback):
    """Checkpoint model + normalizer, and log command-following progress."""

    def __init__(self, every: int, venv: VecNormalize) -> None:
        super().__init__()
        self.every = every
        self.venv = venv
        self.next = every
        self.t0 = time.time()
        self.returns: list[float] = []
        self.speeds: list[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self.returns.append(float(ep["r"]))
                self.speeds.append(float(ep.get("progress", 0.0)))
        if self.num_timesteps >= self.next:
            self.next += self.every
            tag = f"nav_{self.num_timesteps // 1000}k"
            self.model.save(str(MODELS / f"{tag}.zip"))
            self.venv.save(str(MODELS / f"{tag}_vecnormalize.pkl"))
            row = {
                "timesteps": int(self.num_timesteps),
                "return": float(np.mean(self.returns[-100:])) if self.returns else None,
                "minutes": round((time.time() - self.t0) / 60, 1),
                "checkpoint": tag,
            }
            with (LOGS / "progress.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timesteps", type=int, default=8_000_000)
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=1_000_000)
    args = p.parse_args()

    MODELS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    fns = [factory(i, args.seed * 100) for i in range(args.envs)]
    venv = SubprocVecEnv(fns) if args.envs > 1 else DummyVecEnv(fns)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    # Same "tuned" PPO preset that fixed forward walking (train.py --ppo-preset tuned).
    model = PPO(
        "MlpPolicy",
        venv,
        seed=args.seed,
        device="cpu",
        verbose=0,
        n_steps=1024,
        batch_size=512,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        policy_kwargs=dict(log_std_init=-1.0, net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )
    model.learn(total_timesteps=args.timesteps, callback=Snapshots(args.save_every, venv))
    model.save(str(MODELS / "nav_final.zip"))
    venv.save(str(MODELS / "nav_final_vecnormalize.pkl"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
