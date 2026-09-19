"""Domain-randomization robustness protocol for a trained Ant-v4 policy.

1. Evaluate the frozen policy on the default (nominal) physics.
2. Evaluate the same frozen policy on randomized friction and torso mass.
3. Optionally fine-tune with randomization on, then re-evaluate.

Example
-------
python robustness.py --model ppo --n-episodes 30
python robustness.py --model ppo --retrain-steps 500000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from randomize_env import DEFAULT_FRICTION_RANGE, DEFAULT_MASS_RANGE, make_ant_env

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

ALGO_CLS = {"ppo": PPO, "sac": SAC}


def infer_algo(model_name: str) -> str:
    stem = Path(model_name).stem.lower()
    if "sac" in stem:
        return "sac"
    return "ppo"


def resolve_model(name: str) -> Path:
    direct = Path(name)
    if direct.exists():
        return direct
    for c in (MODELS_DIR / f"{name}_ant.zip", MODELS_DIR / f"{name}.zip"):
        if c.exists():
            return c
    raise FileNotFoundError(f"No model for '{name}'")


def vecnormalize_path(model_path: Path) -> Path | None:
    stem = model_path.stem
    base = stem[:-4] if stem.endswith("_ant") else stem
    for c in (model_path.with_name(f"{base}_vecnormalize.pkl"), MODELS_DIR / f"{base}_vecnormalize.pkl"):
        if c.exists():
            return c
    return None


def make_vec(randomize: bool, seed: int, vn_path: Path | None):
    def _init():
        return Monitor(make_ant_env(randomize=randomize, seed=seed))

    vec = DummyVecEnv([_init])
    if vn_path is not None:
        vec = VecNormalize.load(str(vn_path), vec)
        vec.training = False
        vec.norm_reward = False
    return vec


def evaluate_policy(model, randomize: bool, n_episodes: int, seed: int, vn_path: Path | None) -> dict:
    vec = make_vec(randomize=randomize, seed=seed, vn_path=vn_path)
    rewards = []
    lengths = []
    masses = []
    frictions = []
    obs = vec.reset()
    finished = 0
    while finished < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, infos = vec.step(action)
        if done[0]:
            info = infos[0]
            if "episode" in info:
                rewards.append(float(info["episode"]["r"]))
                lengths.append(int(info["episode"]["l"]))
            dr = info.get("domain_randomization") or {}
            if "torso_mass_scale" in dr:
                masses.append(dr["torso_mass_scale"])
            if "mean_friction_scale" in dr:
                frictions.append(dr["mean_friction_scale"])
            finished += 1
            print(
                f"  [{'DR' if randomize else 'nominal'}] "
                f"ep {finished}/{n_episodes}: {rewards[-1]:.1f}"
            )
    vec.close()
    rewards_a = np.asarray(rewards, dtype=float)
    return {
        "randomize": randomize,
        "n_episodes": n_episodes,
        "mean_reward": float(rewards_a.mean()),
        "std_reward": float(rewards_a.std()),
        "mean_length": float(np.mean(lengths)),
        "rewards": rewards,
        "mass_scales": masses,
        "friction_scales": frictions,
    }


def plot_robustness(nominal: dict, randomized: dict, recovered: dict | None, out_path: Path) -> None:
    labels = ["Nominal physics", "Randomized physics"]
    means = [nominal["mean_reward"], randomized["mean_reward"]]
    stds = [nominal["std_reward"], randomized["std_reward"]]
    if recovered is not None:
        labels.append("After DR fine-tune\n(on randomized env)")
        means.append(recovered["mean_reward"])
        stds.append(recovered["std_reward"])

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=["#1a73e8", "#d93025", "#188038"][: len(labels)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Robustness to friction and torso-mass randomization")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{mean:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Robustness test via domain randomization")
    p.add_argument("--model", type=str, default="ppo")
    p.add_argument("--n-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retrain-steps", type=int, default=0, help="Fine-tune with DR; 0 skips retraining")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = resolve_model(args.model)
    algo = infer_algo(args.model)
    cls = ALGO_CLS[algo]
    print(f"Loading {model_path}")
    vn_path = vecnormalize_path(model_path)
    dummy_vec = make_vec(randomize=False, seed=args.seed, vn_path=vn_path)
    model = cls.load(model_path, env=dummy_vec, device=args.device)

    print("Evaluating on nominal Ant-v4 physics")
    nominal = evaluate_policy(model, randomize=False, n_episodes=args.n_episodes, seed=args.seed, vn_path=vn_path)
    print("Evaluating on randomized friction + torso mass")
    randomized = evaluate_policy(model, randomize=True, n_episodes=args.n_episodes, seed=args.seed + 10_000, vn_path=vn_path)
    dummy_vec.close()

    drop = nominal["mean_reward"] - randomized["mean_reward"]
    drop_pct = 100.0 * drop / abs(nominal["mean_reward"]) if nominal["mean_reward"] else None

    recovered = None
    retrain_stats = None
    if args.retrain_steps > 0:
        print(f"Fine-tuning {algo.upper()} for {args.retrain_steps:,} steps with domain randomization")
        run_name = f"{algo}_dr"
        cmd = [
            sys.executable,
            str(ROOT / "train.py"),
            "--algo",
            algo,
            "--timesteps",
            str(args.retrain_steps),
            "--randomize",
            "--run-name",
            run_name,
            "--resume",
            str(model_path),
            "--seed",
            str(args.seed),
            "--dummy-vec",
        ]
        t0 = time.perf_counter()
        subprocess.check_call(cmd, cwd=str(ROOT))
        retrain_stats = {"wall_clock_seconds": time.perf_counter() - t0, "timesteps": args.retrain_steps}
        dr_path = MODELS_DIR / f"{run_name}_ant.zip"
        dr_vn = MODELS_DIR / f"{run_name}_vecnormalize.pkl"
        dummy2 = make_vec(randomize=True, seed=args.seed, vn_path=dr_vn if dr_vn.exists() else vn_path)
        dr_model = cls.load(dr_path, env=dummy2, device=args.device)
        print("Re-evaluating fine-tuned policy on randomized physics")
        recovered = evaluate_policy(
            dr_model,
            randomize=True,
            n_episodes=args.n_episodes,
            seed=args.seed + 20_000,
            vn_path=dr_vn if dr_vn.exists() else vn_path,
        )
        dummy2.close()

    plot_path = RESULTS_DIR / "robustness_bars.png"
    plot_robustness(nominal, randomized, recovered, plot_path)

    payload = {
        "model": str(model_path),
        "friction_range": list(DEFAULT_FRICTION_RANGE),
        "mass_range": list(DEFAULT_MASS_RANGE),
        "nominal": nominal,
        "randomized": randomized,
        "absolute_drop": drop,
        "percent_drop": drop_pct,
        "recovered": recovered,
        "retrain": retrain_stats,
        "plot": str(plot_path),
    }
    out_path = RESULTS_DIR / "robustness.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Robustness to domain randomization",
        "",
        f"Policy: `{model_path.name}`. Friction multipliers {DEFAULT_FRICTION_RANGE}, "
        f"torso mass multipliers {DEFAULT_MASS_RANGE}.",
        "",
        "| Condition | Mean reward | Std | Mean length |",
        "|-----------|-------------|-----|-------------|",
        f"| Nominal physics | {nominal['mean_reward']:.1f} | {nominal['std_reward']:.1f} | {nominal['mean_length']:.1f} |",
        f"| Randomized physics | {randomized['mean_reward']:.1f} | {randomized['std_reward']:.1f} | {randomized['mean_length']:.1f} |",
    ]
    if recovered is not None:
        md.append(
            f"| Randomized after DR fine-tune | {recovered['mean_reward']:.1f} | "
            f"{recovered['std_reward']:.1f} | {recovered['mean_length']:.1f} |"
        )
    md.append("")
    md.append(f"Performance drop without retraining: **{drop:.1f}** "
              f"({drop_pct:.1f}% relative)." if drop_pct is not None else "")
    (RESULTS_DIR / "robustness.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Drop: {drop:.1f} ({drop_pct:.1f}% )" if drop_pct is not None else f"Drop: {drop:.1f}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
