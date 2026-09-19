"""Train PPO or SAC on Gymnasium Ant-v4.

Examples
--------
python train.py --algo ppo
python train.py --algo sac --timesteps 1000000 --stop-at 150000
python train.py --algo sac --timesteps 1000000 --resume models/checkpoints/sac/sac_150000_steps.zip
python train.py --algo ppo --randomize --run-name ppo_dr
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from plot_results import plot_reward_curve
from randomize_env import make_ant_env

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"
TB_DIR = ROOT / "tb_logs"

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
}


def configure_torch_backend(device: str) -> str:
    """Use the RTX 4060 efficiently without changing SAC update rules.

    Observation/action sizes stay small, so the win is: CUDA for every
    gradient batch, TF32 matmuls on Ampere, and skipping extra CPU
    thread contention around those batches.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # benchmark helps large convnets; for SAC's tiny MLPs it can add
        # autotune stalls around each new kernel shape.
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        # Env stepping is CPU-side MuJoCo; keep a few CPU threads for that
        # and let the GPU take the optimizer step without oversubscription.
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
        name = torch.cuda.get_device_name(0)
        print(f"PyTorch {torch.__version__} | device={device} ({name})")
    else:
        print(f"PyTorch {torch.__version__} | device={device}")
    return device


def replay_buffer_path_for(model_zip: Path) -> Path:
    return model_zip.with_name(f"{model_zip.stem}_replay_buffer.pkl")


class CheckpointAndStopCallback(BaseCallback):
    """Save model (+ SAC replay buffer) on an absolute timestep schedule.

    Uses `model.num_timesteps` rather than callback `n_calls`, so resume
    still checkpoints at 200k/250k/... after a 150k warm start.
    """

    def __init__(
        self,
        ckpt_dir: Path,
        prefix: str,
        save_freq: int = 50_000,
        stop_at: Optional[int] = None,
        save_replay_buffer: bool = True,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose)
        self.ckpt_dir = Path(ckpt_dir)
        self.prefix = prefix
        self.save_freq = int(save_freq)
        self.stop_at = stop_at
        self.save_replay_buffer = save_replay_buffer
        self._saved_buckets: set[int] = set()
        self.stopped_early = False
        self.last_checkpoint: Optional[Path] = None

    def _save(self, ts: int) -> Path:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        zip_path = self.ckpt_dir / f"{self.prefix}_{ts}_steps.zip"
        self.model.save(str(zip_path))
        if self.save_replay_buffer and getattr(self.model, "replay_buffer", None) is not None:
            self.model.save_replay_buffer(str(replay_buffer_path_for(zip_path)))
        self.last_checkpoint = zip_path
        if self.verbose:
            print(f"Checkpoint saved: {zip_path} (num_timesteps={ts})")
        return zip_path

    def _on_step(self) -> bool:
        ts = int(self.model.num_timesteps)
        if self.save_freq > 0:
            bucket = ts // self.save_freq
            if bucket > 0 and bucket not in self._saved_buckets:
                self._save(ts)
                self._saved_buckets.add(bucket)
        if self.stop_at is not None and ts >= self.stop_at:
            if self.last_checkpoint is None:
                self._save(ts)
            self.stopped_early = True
            if self.verbose:
                print(f"Reached --stop-at {self.stop_at:,}; ending this training process.")
            return False
        return True


class EpisodeStatsCallback(BaseCallback):
    """Track rolling episode reward/length for JSON summaries."""

    def __init__(self, window: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self.rewards: list[float] = []
        self.lengths: list[int] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.rewards.append(float(info["episode"]["r"]))
                self.lengths.append(int(info["episode"]["l"]))
        return True

    def summary(self) -> dict:
        if not self.rewards:
            return {"n_episodes": 0, "mean_reward": None, "mean_length": None}
        tail_r = self.rewards[-self.window :]
        tail_l = self.lengths[-self.window :]
        return {
            "n_episodes": len(self.rewards),
            "mean_reward": float(np.mean(tail_r)),
            "std_reward": float(np.std(tail_r)),
            "mean_length": float(np.mean(tail_l)),
            "last_reward": float(self.rewards[-1]),
        }


def _make_env_fn(
    rank: int,
    seed: int,
    randomize: bool,
    monitor_dir: Path,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        env = make_ant_env(randomize=randomize, seed=seed + rank)
        env = Monitor(env, filename=str(monitor_dir / f"worker_{rank}"))
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(
    n_envs: int,
    seed: int,
    randomize: bool,
    monitor_dir: Path,
    use_subproc: bool,
):
    monitor_dir.mkdir(parents=True, exist_ok=True)
    env_fns = [_make_env_fn(i, seed, randomize, monitor_dir) for i in range(n_envs)]
    if n_envs == 1 or not use_subproc:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def default_n_envs(algo: str, override: int) -> int:
    if override > 0:
        return override
    # PPO benefits from parallel rollouts. SAC is typically 1 env.
    return 8 if algo == "ppo" else 1


def wrap_normalize(vec_env, algo: str, *, training: bool) -> VecNormalize:
    """SB3 recommends VecNormalize on MuJoCo; without it Ant-v4 rarely walks.

    PPO can normalize rewards as well. SAC is more sensitive to reward scaling,
    so only observations are normalized for SAC.
    """
    return VecNormalize(
        vec_env,
        training=training,
        norm_obs=True,
        norm_reward=(algo == "ppo" and training),
        clip_obs=10.0,
        gamma=0.99,
    )


def build_model(algo: str, env, seed: int, tb_log: str, device: str = "auto"):
    algo_cls = ALGOS[algo]
    # Keep close to SB3 defaults, with a modest net and (for PPO) a batch that
    # matches 8-env rollouts. VecNormalize is the important extra.
    if algo == "ppo":
        return PPO(
            "MlpPolicy",
            env,
            verbose=1,
            seed=seed,
            tensorboard_log=tb_log,
            device=device,
            n_steps=max(128, 2048 // max(1, env.num_envs)),
            batch_size=64,
            gae_lambda=0.95,
            gamma=0.99,
            n_epochs=10,
            learning_rate=3e-4,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
        )
    return SAC(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=tb_log,
        device=device,
        buffer_size=1_000_000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_rate=3e-4,
        learning_starts=10_000,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO or SAC on Ant-v4")
    p.add_argument("--algo", choices=["ppo", "sac"], required=True)
    p.add_argument("--timesteps", type=int, default=1_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=0, help="0 = algo default (8 PPO / 1 SAC)")
    p.add_argument("--randomize", action="store_true", help="Train with domain randomization")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--progress", action="store_true", default=True)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--dummy-vec",
        action="store_true",
        help="Force DummyVecEnv (more reliable on some Windows setups)",
    )
    p.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable VecNormalize (recommended for SAC; RL Zoo trains SAC on Ant without it)",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a .zip checkpoint. Replay buffer is loaded from the sibling *_replay_buffer.pkl",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Save model (+ SAC replay buffer) every N env steps. 0 disables.",
    )
    p.add_argument(
        "--stop-at",
        type=int,
        default=None,
        help="Stop automatically after this many *absolute* timesteps (e.g. 150000)",
    )
    p.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory for periodic checkpoints (default: models/checkpoints/<run>)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_name = args.run_name or args.algo
    n_envs = default_n_envs(args.algo, args.n_envs)
    seed = args.seed
    set_random_seed(seed)
    device = configure_torch_backend(args.device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TB_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else (MODELS_DIR / "checkpoints" / run_name)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    monitor_dir = LOG_DIR / run_name / "monitor"
    eval_dir = LOG_DIR / run_name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    target_ts = args.timesteps
    stop_at = args.stop_at

    print(
        f"Training {args.algo.upper()} on Ant-v4 | "
        f"target={target_ts:,} n_envs={n_envs} "
        f"randomize={args.randomize} seed={seed} device={device} "
        f"checkpoint_freq={args.checkpoint_freq} stop_at={stop_at}"
    )

    train_env = build_vec_env(
        n_envs=n_envs,
        seed=seed,
        randomize=args.randomize,
        monitor_dir=monitor_dir,
        use_subproc=not args.dummy_vec,
    )
    use_norm = (args.algo == "ppo") and not args.no_normalize
    train_env = wrap_normalize(train_env, args.algo, training=True) if use_norm else train_env

    if args.resume and use_norm:
        resume_path = Path(args.resume)
        stem = resume_path.stem
        base = stem[:-4] if stem.endswith("_ant") else stem
        vn_guess = [
            resume_path.with_name(f"{base}_vecnormalize.pkl"),
            MODELS_DIR / f"{base}_vecnormalize.pkl",
            MODELS_DIR / "ppo_vecnormalize.pkl",
        ]
        for vn in vn_guess:
            if vn.exists():
                inner = train_env.venv
                train_env = VecNormalize.load(str(vn), inner)
                train_env.training = True
                train_env.norm_reward = True
                print(f"Loaded VecNormalize stats from {vn}")
                break

    eval_env = DummyVecEnv(
        [_make_env_fn(0, seed + 10_000, args.randomize, eval_dir / "monitor")]
    )
    if use_norm:
        eval_env = wrap_normalize(eval_env, args.algo, training=False)
        eval_env.obs_rms = train_env.obs_rms
        eval_env.ret_rms = train_env.ret_rms

    tb_log = str(TB_DIR / run_name)
    reset_num_timesteps = True
    if args.resume:
        algo_cls = ALGOS[args.algo]
        resume_zip = Path(args.resume)
        model = algo_cls.load(resume_zip, env=train_env, device=device, tensorboard_log=tb_log)
        buf_path = replay_buffer_path_for(resume_zip)
        if args.algo == "sac":
            if not buf_path.exists():
                raise FileNotFoundError(
                    f"Replay buffer not found at {buf_path}. "
                    "Resume of SAC requires the sibling *_replay_buffer.pkl checkpoint."
                )
            model.load_replay_buffer(str(buf_path))
            print(f"Loaded replay buffer from {buf_path} (size={model.replay_buffer.size()})")
        reset_num_timesteps = False
        print(
            f"Resumed from {resume_zip} at num_timesteps={model.num_timesteps:,} "
            f"toward target {target_ts:,}"
        )
    else:
        model = build_model(args.algo, train_env, seed, tb_log, device=device)

    already = int(model.num_timesteps)
    if stop_at is not None and already >= stop_at:
        print(f"Already at {already:,} >= --stop-at {stop_at:,}; nothing to do.")
        train_env.close()
        eval_env.close()
        return
    if already >= target_ts and stop_at is None:
        print(f"Already at {already:,} >= target {target_ts:,}; nothing to do.")
        train_env.close()
        eval_env.close()
        return

    # SB3: with reset_num_timesteps=False, learn(total_timesteps=N) trains N *additional*
    # steps and internally adds the current count to the stop condition.
    if reset_num_timesteps:
        horizon = stop_at if stop_at is not None else target_ts
        learn_ts = horizon
    else:
        abs_horizon = stop_at if stop_at is not None else target_ts
        learn_ts = max(0, abs_horizon - already)

    stats_cb = EpisodeStatsCallback(window=100)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / f"{run_name}_best"),
        log_path=str(eval_dir),
        eval_freq=max(args.eval_freq // n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )
    ckpt_cb = CheckpointAndStopCallback(
        ckpt_dir=ckpt_dir,
        prefix=run_name,
        save_freq=args.checkpoint_freq,
        stop_at=stop_at,
        save_replay_buffer=(args.algo == "sac"),
    )
    if already > 0 and args.checkpoint_freq > 0:
        # Do not re-save 50k/100k/150k immediately on resume.
        ckpt_cb._saved_buckets = set(range(1, (already // args.checkpoint_freq) + 1))
    callbacks = CallbackList([stats_cb, eval_cb, ckpt_cb])

    t0 = time.perf_counter()
    model.learn(
        total_timesteps=learn_ts,
        callback=callbacks,
        tb_log_name=run_name,
        progress_bar=args.progress and not args.no_progress,
        reset_num_timesteps=reset_num_timesteps,
        log_interval=4,
    )
    wall_s = time.perf_counter() - t0

    model_path = MODELS_DIR / f"{run_name}_ant.zip"
    reached_target = int(model.num_timesteps) >= target_ts and not ckpt_cb.stopped_early
    if reached_target:
        model.save(str(model_path))
        if args.algo == "sac":
            model.save_replay_buffer(str(replay_buffer_path_for(model_path)))
    elif ckpt_cb.last_checkpoint is not None:
        model_path = ckpt_cb.last_checkpoint
        print(f"Stopped early; latest checkpoint is {model_path}")

    vecnorm_path = None
    if use_norm:
        vecnorm_path = MODELS_DIR / f"{run_name}_vecnormalize.pkl"
        train_env.save(str(vecnorm_path))

    curve_path = RESULTS_DIR / f"{run_name}_reward_curve.png"
    # Spec names for the two baseline algorithms.
    if run_name == "ppo":
        curve_path = RESULTS_DIR / "ppo_reward_curve.png"
    elif run_name == "sac":
        curve_path = RESULTS_DIR / "sac_reward_curve.png"

    try:
        plot_reward_curve(monitor_dir, curve_path, title=f"{args.algo.upper()} Ant-v4")
    except FileNotFoundError as exc:
        print(f"Warning: could not plot reward curve ({exc})")
        curve_path = None

    summary = {
        "algo": args.algo,
        "run_name": run_name,
        "timesteps": args.timesteps,
        "num_timesteps": int(model.num_timesteps),
        "remaining_timesteps": max(0, target_ts - int(model.num_timesteps)),
        "stopped_early": bool(ckpt_cb.stopped_early),
        "last_checkpoint": str(ckpt_cb.last_checkpoint) if ckpt_cb.last_checkpoint else None,
        "n_envs": n_envs,
        "seed": seed,
        "device": device,
        "randomize": args.randomize,
        "wall_clock_seconds": wall_s,
        "wall_clock_hours": wall_s / 3600.0,
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnorm_path),
        "curve_path": str(curve_path) if curve_path else None,
        "episode_stats": stats_cb.summary(),
    }
    stats_path = RESULTS_DIR / f"{run_name}_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    train_env.close()
    eval_env.close()
    print(f"Saved model to {model_path}")
    print(f"Wall clock: {wall_s/60:.1f} min")
    print(f"Episode stats: {stats_cb.summary()}")


if __name__ == "__main__":
    main()
