"""Live side-by-side RL training viewer.

Trains PPO and SAC on Ant-v4 from scratch, in this process, right now. Each
algorithm's current policy is copied every few thousand steps into a display
simulation that runs at real time and streams body transforms to the browser,
so the 3D views show what the policies can actually do at this moment in
training. Learning curves come from the live episode buffers.

    python live_server.py

Then open http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from randomize_env import make_ant_env

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
DASHBOARD = ROOT / "dashboard"

ENV_DT = 0.05  # Ant-v4 advances 0.05 s of simulated time per env step
SYNC_EVERY = 2_000  # env steps between display-policy refreshes
VERBOSE = False
ALLOW_RESTART = os.getenv("ALLOW_RESTART", "1") == "1"


def clip_normalize(obs: np.ndarray, rms) -> np.ndarray:
    if rms is None:
        return obs
    return np.clip((obs - rms.mean) / np.sqrt(rms.var + 1e-8), -10.0, 10.0)


# --------------------------------------------------------------------------- #
# Live training lanes
# --------------------------------------------------------------------------- #


class _SyncCallback(BaseCallback):
    """Pushes a snapshot of the training policy to the viewer, and stops on request."""

    def __init__(self, lane: "LiveTrainer") -> None:
        super().__init__()
        self.lane = lane
        self.next_sync = SYNC_EVERY

    def _on_step(self) -> bool:
        if self.model.num_timesteps >= self.next_sync:
            self.next_sync = self.model.num_timesteps + SYNC_EVERY
            self.lane.publish(self.model)
        return not self.lane.stop_event.is_set()


class LiveTrainer:
    """One algorithm training in the background with a viewable policy copy."""

    def __init__(self, algo: str, seed: int = 0) -> None:
        self.algo = algo
        self.seed = seed
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.history: list[dict] = []
        self.steps = 0
        self.reward = None
        self.episodes = 0
        self.fps = 0.0
        self.started_at = time.time()
        self._build()

    def _build(self) -> None:
        # PPO already out-collects SAC by ~10x here; 2 envs leaves SAC the CPU
        # it needs for its per-step gradient update.
        n_envs = 2 if self.algo == "ppo" else 1
        venv = DummyVecEnv(
            [
                (lambda i=i: Monitor(make_ant_env(seed=self.seed + i)))
                for i in range(n_envs)
            ]
        )
        if self.algo == "ppo":
            venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
            self.model = PPO(
                "MlpPolicy",
                venv,
                seed=self.seed,
                device="cpu",
                verbose=0,
                n_steps=max(128, 2048 // n_envs),
                batch_size=64,
                learning_rate=3e-4,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
            )
        else:
            self.model = SAC(
                "MlpPolicy",
                venv,
                seed=self.seed,
                device="cuda" if torch.cuda.is_available() else "cpu",
                verbose=0,
                buffer_size=300_000,
                batch_size=256,
                learning_rate=3e-4,
                learning_starts=5_000,
                train_freq=1,
                gradient_steps=1,
                policy_kwargs=dict(net_arch=[256, 256]),
            )
        self.venv = venv
        self.display_policy = copy.deepcopy(self.model.policy).to("cpu")
        self.display_policy.set_training_mode(False)
        self.obs_rms = None

    def start(self) -> None:
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        try:
            self.model.learn(
                total_timesteps=10_000_000,
                callback=_SyncCallback(self),
                log_interval=1_000_000,
            )
        except Exception as exc:  # keep the viewer alive if a lane dies
            print(f"[{self.algo}] training stopped: {exc}")

    def publish(self, model) -> None:
        """Copy current weights (and PPO's obs statistics) to the display policy."""
        state = {k: v.detach().cpu().clone() for k, v in model.policy.state_dict().items()}
        rewards = [info["r"] for info in model.ep_info_buffer] if model.ep_info_buffer else []
        elapsed = max(1e-6, time.time() - self.started_at)
        with self.lock:
            self.display_policy.load_state_dict(state)
            if isinstance(self.venv, VecNormalize):
                self.obs_rms = copy.deepcopy(self.venv.obs_rms)
            self.steps = int(model.num_timesteps)
            self.fps = self.steps / elapsed
            if rewards:
                self.reward = float(safe_mean(rewards))
                self.episodes = int(getattr(model, "_episode_num", len(rewards)))
                self.history.append(
                    {
                        "step": self.steps,
                        "minutes": round(elapsed / 60, 3),
                        "reward": round(self.reward, 1),
                    }
                )

    def act(self, obs: np.ndarray) -> np.ndarray:
        with self.lock:
            obs = clip_normalize(obs, self.obs_rms)
            action, _ = self.display_policy.predict(obs, deterministic=True)
        return action

    def restart(self) -> None:
        self.stop_event.set()
        thread = getattr(self, "thread", None)
        if thread and thread.is_alive():
            thread.join(timeout=10)
        try:
            self.venv.close()
        except Exception:
            pass
        with self.lock:
            self.history = []
            self.steps = 0
            self.reward = None
            self.episodes = 0
            self.fps = 0.0
            self.started_at = time.time()
        self._build()
        self.start()

    def stats(self) -> dict:
        return {
            "algo": self.algo,
            "steps": self.steps,
            "reward": None if self.reward is None else round(self.reward, 1),
            "episodes": self.episodes,
            "fps": round(self.fps, 1),
        }


# --------------------------------------------------------------------------- #
# Saved policies (optional comparison against the finished 1M runs)
# --------------------------------------------------------------------------- #


def discover_policies() -> list[dict]:
    """Live lanes first, then the finished runs worth comparing against."""
    options = [
        {"id": "live_ppo", "label": "PPO — training live"},
        {"id": "live_sac", "label": "SAC — training live"},
        {"id": "random", "label": "Random actions"},
    ]
    if (MODELS / "ppo_ant.zip").exists():
        options.append({"id": "ppo", "label": "PPO — finished 1.0M run"})
    if (MODELS / "sac_ant.zip").exists():
        options.append({"id": "sac_1000000", "label": "SAC — finished 1.0M run"})
    return options


class SavedPolicies:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}
        self._ppo_rms = None
        self._lock = threading.Lock()

    def act(self, policy_id: str, obs: np.ndarray):
        with self._lock:
            model = self._models.get(policy_id)
            if model is None:
                if policy_id == "ppo":
                    model = PPO.load(MODELS / "ppo_ant.zip", device="cpu")
                    path = MODELS / "ppo_vecnormalize.pkl"
                    if path.exists():
                        with path.open("rb") as handle:
                            self._ppo_rms = pickle.load(handle).obs_rms
                else:
                    value = int(policy_id.split("_")[1])
                    file = MODELS / "checkpoints" / "sac" / f"sac_{value}_steps.zip"
                    model = SAC.load(file if file.exists() else MODELS / "sac_ant.zip", device="cpu")
                self._models[policy_id] = model
            if policy_id == "ppo":
                obs = clip_normalize(obs, self._ppo_rms)
            action, _ = model.predict(obs, deterministic=True)
            return action


SAVED = SavedPolicies()


# --------------------------------------------------------------------------- #
# Display simulations
# --------------------------------------------------------------------------- #


class Panel:
    def __init__(self, panel_id: str, policy_id: str, trainers: dict, seed: int) -> None:
        self.id = panel_id
        self.trainers = trainers
        self.env = make_ant_env(seed=seed)
        self.policy_id = policy_id
        self.obs, _ = self.env.reset(seed=seed)
        self.reset_episode()

    def reset_episode(self) -> None:
        self.obs, _ = self.env.reset()
        self.episode_return = 0.0
        self.episode_steps = 0
        self.start_x = float(self.env.unwrapped.data.qpos[0])

    def set_policy(self, policy_id: str) -> None:
        known = {option["id"] for option in discover_policies()}
        if policy_id != self.policy_id and policy_id in known:
            self.policy_id = policy_id
            self.reset_episode()

    def _action(self):
        if self.policy_id == "random":
            return self.env.action_space.sample()
        if self.policy_id.startswith("live_"):
            return self.trainers[self.policy_id[5:]].act(self.obs)
        return SAVED.act(self.policy_id, self.obs)

    def step(self) -> None:
        self.obs, reward, terminated, truncated, _ = self.env.step(self._action())
        self.episode_return += float(reward)
        self.episode_steps += 1
        if terminated or truncated:
            self.reset_episode()

    def snapshot(self) -> dict:
        data = self.env.unwrapped.data
        return {
            "id": self.id,
            "policy": self.policy_id,
            "xpos": np.asarray(data.xpos[1:]).round(4).ravel().tolist(),
            "xquat": np.asarray(data.xquat[1:]).round(4).ravel().tolist(),
            "return": round(self.episode_return, 1),
            "steps": self.episode_steps,
            "speed": round(float(data.qvel[0]), 2),
            "distance": round(float(data.qpos[0]) - self.start_x, 2),
        }


class Simulation:
    def __init__(self, left: str, right: str) -> None:
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.trainers = {"ppo": LiveTrainer("ppo"), "sac": LiveTrainer("sac")}
        self.panels = {
            "left": Panel("left", left, self.trainers, seed=11),
            "right": Panel("right", right, self.trainers, seed=11),
        }
        self.speed = 1.0
        self.paused = False
        self.frame: dict = {}
        self.tick = 0
        self.geometry = self._describe_geometry()
        for trainer in self.trainers.values():
            trainer.start()
        threading.Thread(target=self._loop, daemon=True).start()

    def _describe_geometry(self) -> list[dict]:
        model = self.panels["left"].env.unwrapped.model
        shapes = []
        for i in range(model.ngeom):
            body = int(model.geom_bodyid[i])
            if body == 0:  # world / floor
                continue
            shapes.append(
                {
                    "body": body - 1,
                    "type": int(model.geom_type[i]),
                    "size": np.asarray(model.geom_size[i]).round(4).tolist(),
                    "pos": np.asarray(model.geom_pos[i]).round(4).tolist(),
                    "quat": np.asarray(model.geom_quat[i]).round(4).tolist(),
                }
            )
        return shapes

    def _loop(self) -> None:
        next_step = time.perf_counter()
        while True:
            with self.lock:
                if not self.paused:
                    for panel in self.panels.values():
                        panel.step()
                # frames keep flowing while paused so resets stay visible
                self.tick += 1
                self.frame = {
                    "tick": self.tick,
                    "paused": self.paused,
                    "panels": [p.snapshot() for p in self.panels.values()],
                    "trainers": {k: t.stats() for k, t in self.trainers.items()},
                }
                self.condition.notify_all()
                interval = (ENV_DT / self.speed) if not self.paused else 0.05
            next_step += interval
            delay = next_step - time.perf_counter()
            if delay < -0.25:  # training starved the loop; resync instead of sprinting
                next_step = time.perf_counter()
                delay = 0.0
            time.sleep(max(0.0, delay))

    def wait_frame(self, last_tick: int, timeout: float = 1.0):
        with self.condition:
            if self.tick == last_tick:
                self.condition.wait(timeout)
            return self.tick, self.frame

    def history(self) -> dict:
        return {k: t.history[:] for k, t in self.trainers.items()}

    def configure(self, payload: dict) -> dict:
        restart = bool(payload.get("restartTraining"))
        with self.lock:
            for side in ("left", "right"):
                if side in payload:
                    self.panels[side].set_policy(payload[side])
            if "speed" in payload:
                self.speed = max(0.25, min(4.0, float(payload["speed"])))
            if "paused" in payload:
                self.paused = bool(payload["paused"])
            if payload.get("reset") or restart:
                for panel in self.panels.values():
                    panel.reset_episode()
            state = {
                "left": self.panels["left"].policy_id,
                "right": self.panels["right"].policy_id,
                "speed": self.speed,
                "paused": self.paused,
            }
        if restart:  # rebuilding models must not hold the simulation lock
            for trainer in self.trainers.values():
                trainer.restart()
        return state


SIM: Simulation


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        if self.path.endswith((".js", ".css", ".html")) or self.path in ("/", ""):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/api/setup"):
            self._json(
                {
                    "policies": discover_policies(),
                    "geometry": SIM.geometry,
                    "left": SIM.panels["left"].policy_id,
                    "right": SIM.panels["right"].policy_id,
                    "allowRestart": ALLOW_RESTART,
                }
            )
        elif self.path.startswith("/api/history"):
            self._json(SIM.history())
        elif self.path.startswith("/api/stream"):
            self._stream()
        else:
            super().do_GET()

    def do_POST(self) -> None:
        if not self.path.startswith("/api/config"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not ALLOW_RESTART:
            payload.pop("restartTraining", None)
        if VERBOSE:
            print(f"config {payload} from {self.headers.get('Referer')}", flush=True)
        self._json(SIM.configure(payload))

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        last = -1
        try:
            while True:
                last, frame = SIM.wait_frame(last)
                if frame:
                    self.wfile.write(f"data: {json.dumps(frame)}\n\n".encode())
                    self.wfile.flush()
        except OSError:
            pass  # client closed the tab / navigated away


def main() -> None:
    parser = argparse.ArgumentParser(description="Live RL training viewer")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8080")))
    parser.add_argument("--left", default="live_ppo")
    parser.add_argument("--right", default="live_sac")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="log config requests")
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose
    torch.set_num_threads(2)  # two trainers plus two display sims share this box

    global SIM
    print("Building PPO and SAC from scratch...")
    SIM = Simulation(args.left, args.right)
    print("Training started.")

    handler = partial(Handler, directory=str(DASHBOARD))
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        url = f"http://{display_host}:{args.port}"
        print(f"Live viewer: {url}  (Ctrl+C to stop)")
        if not args.no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
