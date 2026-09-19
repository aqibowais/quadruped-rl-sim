"""Load a trained Ant-v4 policy, evaluate it, and record a video/GIF.

Examples
--------
python evaluate.py --model ppo
python evaluate.py --model sac --n-episodes 20
python evaluate.py --model ppo --randomize
python evaluate.py --model random --gif results/random_policy.gif
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from randomize_env import make_ant_env

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ALGO_BY_NAME = {
    "ppo": PPO,
    "sac": SAC,
}


def resolve_model_path(name: str) -> Path:
    if name == "random":
        raise ValueError("random policy has no zip file")
    direct = Path(name)
    if direct.exists():
        return direct
    candidates = [
        MODELS_DIR / f"{name}_ant.zip",
        MODELS_DIR / f"{name}.zip",
        Path(name + ".zip"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No model found for '{name}'. Looked in: " + ", ".join(str(c) for c in candidates)
    )


def infer_algo_cls(path: Path):
    stem = path.stem.lower()
    algo_cls = PPO if "sac" not in stem else SAC
    for key, cls in ALGO_BY_NAME.items():
        if stem.startswith(key) or f"_{key}_" in f"_{stem}_":
            algo_cls = cls
            break
    return algo_cls


def find_vecnormalize(model_path: Path) -> Path | None:
    stem = model_path.stem
    base = stem[:-4] if stem.endswith("_ant") else stem
    for c in (
        model_path.with_name(f"{base}_vecnormalize.pkl"),
        MODELS_DIR / f"{base}_vecnormalize.pkl",
        MODELS_DIR / f"{stem}_vecnormalize.pkl",
    ):
        if c.exists():
            return c
    return None


def make_vec(randomize: bool, seed: int, record: bool):
    def _init():
        env = make_ant_env(
            render_mode="rgb_array" if record else None,
            randomize=randomize,
            seed=seed,
        )
        return Monitor(env)

    return DummyVecEnv([_init])


def rollout_vec(
    vec,
    predict_fn,
    n_episodes: int,
    record: bool,
    max_frames: int,
    frame_skip: int,
) -> tuple[list[float], list[int], list[np.ndarray]]:
    rewards: list[float] = []
    lengths: list[int] = []
    frames: list[np.ndarray] = []
    episodes_done = 0
    obs = vec.reset()
    ep_r = 0.0
    ep_l = 0
    while episodes_done < n_episodes:
        action = predict_fn(obs)
        obs, rew, done, infos = vec.step(action)
        ep_r += float(rew[0])
        ep_l += 1
        if record and len(frames) < max_frames and (ep_l % frame_skip == 0):
            frame = vec.envs[0].render()
            if frame is not None:
                frames.append(np.asarray(frame))
        if done[0]:
            info = infos[0]
            # Prefer Monitor's true undiscounted episode return.
            if "episode" in info:
                rewards.append(float(info["episode"]["r"]))
                lengths.append(int(info["episode"]["l"]))
            else:
                rewards.append(ep_r)
                lengths.append(ep_l)
            episodes_done += 1
            print(
                f"  episode {episodes_done}/{n_episodes}: "
                f"reward={rewards[-1]:.1f} length={lengths[-1]}"
            )
            ep_r = 0.0
            ep_l = 0
    return rewards, lengths, frames


def save_video(frames: list[np.ndarray], mp4_path: Path, gif_path: Path, fps: int) -> None:
    if not frames:
        print("No frames captured; skip video.")
        return
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(mp4_path, frames, fps=fps)
    gif_frames = frames[:: max(1, len(frames) // 180)]
    imageio.mimsave(gif_path, gif_frames, fps=min(fps, 15), loop=0)
    print(f"Wrote {mp4_path} ({len(frames)} frames) and {gif_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained Ant-v4 policy")
    p.add_argument(
        "--model",
        type=str,
        default="ppo",
        help="ppo | sac | random | path/to/model.zip",
    )
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--randomize", action="store_true")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--stochastic", action="store_true", help="Override --deterministic")
    p.add_argument("--record", action="store_true", default=True)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--gif", type=str, default=None)
    p.add_argument("--mp4", type=str, default=None)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=1500)
    p.add_argument("--frame-skip", type=int, default=2)
    p.add_argument("--stats-out", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    record = args.record and not args.no_record
    deterministic = args.deterministic and not args.stochastic
    tag = Path(args.model).stem if args.model not in ("ppo", "sac", "random") else args.model
    if args.randomize:
        tag = f"{tag}_randomized"

    vec = make_vec(randomize=args.randomize, seed=args.seed, record=record)

    if args.model == "random":

        def predict_fn(obs):
            return np.array([vec.action_space.sample()])

    else:
        model_path = resolve_model_path(args.model)
        vn_path = find_vecnormalize(model_path)
        if vn_path is not None:
            vec = VecNormalize.load(str(vn_path), vec)
            vec.training = False
            vec.norm_reward = False
            print(f"Loaded VecNormalize from {vn_path}")
        model = infer_algo_cls(model_path).load(model_path, env=vec)

        def predict_fn(obs):
            action, _ = model.predict(obs, deterministic=deterministic)
            return action

    print(f"Evaluating '{args.model}' | randomize={args.randomize} episodes={args.n_episodes}")
    rewards, lengths, frames = rollout_vec(
        vec,
        predict_fn,
        n_episodes=args.n_episodes,
        record=record,
        max_frames=args.max_frames,
        frame_skip=args.frame_skip,
    )
    vec.close()

    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards))
    mean_l = float(np.mean(lengths))
    print(f"Mean reward: {mean_r:.1f} ± {std_r:.1f}")
    print(f"Mean length: {mean_l:.1f}")

    gif_path = Path(args.gif) if args.gif else RESULTS_DIR / ("demo.gif" if tag == "ppo" else f"{tag}_demo.gif")
    mp4_path = Path(args.mp4) if args.mp4 else RESULTS_DIR / f"{tag}_demo.mp4"
    if record:
        save_video(frames, mp4_path, gif_path, fps=args.fps)

    stats = {
        "model": args.model,
        "randomize": args.randomize,
        "n_episodes": args.n_episodes,
        "mean_reward": mean_r,
        "std_reward": std_r,
        "mean_length": mean_l,
        "rewards": rewards,
        "lengths": lengths,
        "gif": str(gif_path) if record else None,
        "mp4": str(mp4_path) if record else None,
    }
    stats_path = Path(args.stats_out) if args.stats_out else RESULTS_DIR / f"{tag}_eval.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
