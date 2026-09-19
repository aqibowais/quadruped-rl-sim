"""End-to-end experiment runner.

Default (definition of done):
  1. Random-policy baseline video
  2. Train PPO for 1M steps, evaluate, record GIF
  3. Train SAC for 1M steps, evaluate
  4. Comparison table + overlay plot
  5. Robustness test + optional DR fine-tune
  6. PDF report

Usage
-----
python run_pipeline.py
python run_pipeline.py --quick          # ~few minutes, not paper-quality
python run_pipeline.py --skip-train     # regenerate plots/report from existing models
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full Ant-v4 RL pipeline")
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--robust-episodes", type=int, default=30)
    p.add_argument("--retrain-steps", type=int, default=300_000)
    p.add_argument("--quick", action="store_true", help="Tiny timestep budget for a smoke test")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-sac", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    p.add_argument("--dummy-vec", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    timesteps = 25_000 if args.quick else args.timesteps
    eval_eps = 3 if args.quick else args.eval_episodes
    robust_eps = 5 if args.quick else args.robust_episodes
    retrain = 8_000 if args.quick else args.retrain_steps
    extra = ["--dummy-vec"] if args.dummy_vec else []

    # Phase 1: random policy baseline
    run(
        [
            PY,
            "evaluate.py",
            "--model",
            "random",
            "--n-episodes",
            str(min(eval_eps, 5)),
            "--gif",
            "results/random_policy.gif",
            "--mp4",
            "results/random_policy.mp4",
            "--stats-out",
            "results/random_eval.json",
            "--seed",
            str(args.seed),
        ]
    )

    if not args.skip_train:
        # Phase 2: PPO
        run(
            [PY, "train.py", "--algo", "ppo", "--timesteps", str(timesteps), "--seed", str(args.seed)]
            + extra
        )
        # Phase 3: SAC
        if not args.skip_sac:
            run(
                [PY, "train.py", "--algo", "sac", "--timesteps", str(timesteps), "--seed", str(args.seed)]
                + extra
            )

    run(
        [
            PY,
            "evaluate.py",
            "--model",
            "ppo",
            "--n-episodes",
            str(eval_eps),
            "--gif",
            "results/demo.gif",
            "--mp4",
            "results/ppo_demo.mp4",
            "--stats-out",
            "results/ppo_eval.json",
        ]
    )
    sac_model = ROOT / "models" / "sac_ant.zip"
    if sac_model.exists() and not args.skip_sac:
        run(
            [
                PY,
                "evaluate.py",
                "--model",
                "sac",
                "--n-episodes",
                str(eval_eps),
                "--gif",
                "results/sac_demo.gif",
                "--mp4",
                "results/sac_demo.mp4",
                "--stats-out",
                "results/sac_eval.json",
            ]
        )

    run([PY, "plot_results.py"])

    if not args.skip_robustness and (ROOT / "models" / "ppo_ant.zip").exists():
        run(
            [
                PY,
                "robustness.py",
                "--model",
                "ppo",
                "--n-episodes",
                str(robust_eps),
                "--retrain-steps",
                str(retrain),
                "--seed",
                str(args.seed),
            ]
        )

    run([PY, "generate_report.py"])
    print("\nPipeline complete. See README.md, report.pdf, and results/.")


if __name__ == "__main__":
    main()
