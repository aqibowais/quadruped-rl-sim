"""Plot training curves and write the PPO vs SAC comparison table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
LOG_DIR = ROOT / "logs"


def load_monitor_dir(monitor_dir: Path) -> pd.DataFrame:
    """Concatenate SB3 Monitor CSVs in a directory (skips the JSON header)."""
    files = sorted(monitor_dir.glob("*.csv")) + sorted(monitor_dir.glob("*.monitor.csv"))
    # Monitor writes worker_0.monitor.csv or worker_0.csv depending on filename.
    files += sorted(monitor_dir.glob("*monitor*"))
    files = sorted(set(files))
    if not files:
        # Recursive fallback: VecMonitor + per-worker files.
        files = sorted(monitor_dir.rglob("*.csv"))
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, skiprows=1)
        except Exception:
            continue
        if "r" not in df.columns:
            continue
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No Monitor CSVs found under {monitor_dir}")
    out = pd.concat(frames, ignore_index=True)
    if "t" in out.columns:
        out = out.sort_values("t").reset_index(drop=True)
    return out


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) == 0:
        return values
    window = max(1, min(window, len(values)))
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, values[0]), values])
    return np.convolve(padded, kernel, mode="valid")[: len(values)]


def timesteps_from_monitor(df: pd.DataFrame) -> np.ndarray:
    if "l" in df.columns:
        return np.cumsum(df["l"].to_numpy())
    return np.arange(1, len(df) + 1)


def plot_reward_curve(
    monitor_dir: Path,
    out_path: Path,
    title: str,
    window: int = 50,
) -> Path:
    df = load_monitor_dir(monitor_dir)
    rewards = df["r"].to_numpy(dtype=float)
    steps = timesteps_from_monitor(df)
    smooth = rolling_mean(rewards, window)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.plot(steps, rewards, color="#9aa0a6", alpha=0.35, linewidth=0.8, label="episode reward")
    ax.plot(steps, smooth, color="#1a73e8", linewidth=2.0, label=f"rolling mean ({window})")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_comparison(out_path: Optional[Path] = None, window: int = 50) -> Optional[Path]:
    series = []
    for algo in ("ppo", "sac"):
        monitor_dir = LOG_DIR / algo / "monitor"
        if not monitor_dir.exists():
            continue
        try:
            df = load_monitor_dir(monitor_dir)
        except FileNotFoundError:
            continue
        series.append((algo.upper(), timesteps_from_monitor(df), df["r"].to_numpy(dtype=float)))
    if not series:
        return None

    out_path = out_path or (RESULTS_DIR / "comparison_curves.png")
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    colors = {"PPO": "#1a73e8", "SAC": "#d93025"}
    for name, steps, rewards in series:
        ax.plot(steps, rolling_mean(rewards, window), color=colors.get(name, None), linewidth=2.2, label=name)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode reward (rolling mean)")
    ax.set_title("PPO vs SAC on Ant-v4")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def sample_efficiency(df: pd.DataFrame, threshold: float) -> Optional[int]:
    """First timestep where rolling-100 mean reward crosses `threshold`."""
    rewards = df["r"].to_numpy(dtype=float)
    if len(rewards) == 0:
        return None
    smooth = rolling_mean(rewards, min(100, len(rewards)))
    steps = timesteps_from_monitor(df)
    hits = np.where(smooth >= threshold)[0]
    if len(hits) == 0:
        return None
    return int(steps[hits[0]])


def write_comparison_table(
    threshold: float = 2000.0,
    out_md: Optional[Path] = None,
) -> Path:
    """Build results/comparison_table.md from stats JSON + monitor logs."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = out_md or (RESULTS_DIR / "comparison_table.md")
    rows = []
    for algo in ("ppo", "sac"):
        stats_path = RESULTS_DIR / f"{algo}_stats.json"
        eval_path = RESULTS_DIR / f"{algo}_eval.json"
        if not stats_path.exists():
            continue
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        eval_stats = {}
        if eval_path.exists():
            eval_stats = json.loads(eval_path.read_text(encoding="utf-8"))
        mean_r = eval_stats.get("mean_reward")
        if mean_r is None:
            mean_r = (stats.get("episode_stats") or {}).get("mean_reward")
        std_r = eval_stats.get("std_reward")
        wall = stats.get("wall_clock_seconds")
        steps_to_thr = None
        monitor_dir = LOG_DIR / algo / "monitor"
        if monitor_dir.exists():
            try:
                df = load_monitor_dir(monitor_dir)
                steps_to_thr = sample_efficiency(df, threshold)
            except FileNotFoundError:
                pass
        rows.append(
            {
                "algo": algo.upper(),
                "mean_reward": mean_r,
                "std_reward": std_r,
                "wall_clock_seconds": wall,
                "timesteps": stats.get("timesteps"),
                "steps_to_threshold": steps_to_thr,
                "threshold": threshold,
            }
        )

    lines = [
        "# PPO vs SAC on Ant-v4",
        "",
        f"Sample efficiency is the first timestep where the rolling-100 episode reward reaches {threshold:.0f}.",
        "",
        "| Algorithm | Final avg. reward | Wall-clock | Timesteps | Sample efficiency |",
        "|-----------|-------------------|------------|-----------|-------------------|",
    ]
    for row in rows:
        reward_s = "—"
        if row["mean_reward"] is not None:
            if row["std_reward"] is not None:
                reward_s = f"{row['mean_reward']:.1f} ± {row['std_reward']:.1f}"
            else:
                reward_s = f"{row['mean_reward']:.1f}"
        wall_s = "—"
        if row["wall_clock_seconds"] is not None:
            minutes = row["wall_clock_seconds"] / 60.0
            wall_s = f"{minutes:.1f} min"
        se_s = "not reached"
        if row["steps_to_threshold"] is not None:
            se_s = f"{row['steps_to_threshold']:,} steps"
        ts = f"{row['timesteps']:,}" if row["timesteps"] else "—"
        lines.append(f"| {row['algo']} | {reward_s} | {wall_s} | {ts} | {se_s} |")
    lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")
    (RESULTS_DIR / "comparison_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    plot_comparison()
    return out_md


if __name__ == "__main__":
    write_comparison_table()
    print("Wrote comparison table and overlay plot.")
