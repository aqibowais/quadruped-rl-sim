"""Direction-commanded Ant: walk whichever way you are told.

Stock Ant-v4 only rewards +x velocity, so its policies can walk one way and
nothing else. Here every episode samples a commanded heading, switches it now
and then mid-episode, and rewards progress along it. The command is appended
to the observation in the torso frame, so the same network can be steered by
a planner later (see maze_env.py).

Reward per step:
    progress along the command (m/s)
  - 0.5 x sideways drift (m/s)
  - 0.3 x torso vertical speed^2      (discourages the bounding gait)
  - 0.05 x torso roll/pitch rate^2    (keeps the body level)
  + 1.0 survival bonus while healthy
  - 0.5 x |action|^2 control cost      (same weight as Ant-v4)
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from randomize_env import make_ant_env

COMMAND_DIM = 2


def torso_yaw(qpos: np.ndarray) -> float:
    w, x, y, z = qpos[3:7]
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class DirectionalAnt(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        switch_steps: tuple[int, int] = (250, 500),
        lateral_weight: float = 0.5,
        bounce_weight: float = 0.3,
        tilt_weight: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        base = env.observation_space
        self.observation_space = spaces.Box(-np.inf, np.inf, (base.shape[0] + COMMAND_DIM,), np.float64)
        self.switch_steps = switch_steps
        self.lateral_weight = lateral_weight
        self.bounce_weight = bounce_weight
        self.tilt_weight = tilt_weight
        self.rng = np.random.default_rng(seed)
        self.command = 0.0
        self.external_command = False
        self._next_switch = 0
        self._steps = 0

    @property
    def mj(self):
        return self.env.unwrapped

    def set_command(self, heading: float) -> None:
        """Steer from outside (a planner). Disables the random command schedule."""
        self.command = float(heading)
        self.external_command = True

    def _command_obs(self) -> np.ndarray:
        rel = self.command - torso_yaw(self.mj.data.qpos)
        return np.array([np.cos(rel), np.sin(rel)])

    def _obs(self, base: np.ndarray) -> np.ndarray:
        return np.concatenate([base, self._command_obs()])

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._steps = 0
        if not self.external_command:
            self.command = float(self.rng.uniform(-np.pi, np.pi))
            self._next_switch = int(self.rng.integers(*self.switch_steps))
        return self._obs(obs), info

    def step(self, action):
        data = self.mj.data
        before = data.qpos[:2].copy()
        obs, _, terminated, truncated, info = self.env.step(action)
        after = data.qpos[:2].copy()
        vel = (after - before) / self.mj.dt
        d = np.array([np.cos(self.command), np.sin(self.command)])
        progress = float(vel @ d)
        lateral = float(abs(vel[0] * d[1] - vel[1] * d[0]))
        vz = float(data.qvel[2])
        tilt = float(data.qvel[3] ** 2 + data.qvel[4] ** 2)
        healthy = 0.0 if terminated else self.mj.healthy_reward
        ctrl = self.mj.control_cost(action)
        reward = (
            progress
            - self.lateral_weight * lateral
            - self.bounce_weight * vz * vz
            - self.tilt_weight * tilt
            + healthy
            - ctrl
        )
        self._steps += 1
        if not self.external_command and self._steps >= self._next_switch:
            self.command = float(self.rng.uniform(-np.pi, np.pi))
            self._next_switch = self._steps + int(self.rng.integers(*self.switch_steps))
        info = dict(info)
        info.update({"progress": progress, "lateral": lateral, "command": self.command})
        return self._obs(obs), float(reward), terminated, truncated, info


def make_directional_env(seed: Optional[int] = None, xml_file=None, max_episode_steps: int = 1000) -> DirectionalAnt:
    return DirectionalAnt(make_ant_env(seed=seed, xml_file=xml_file, max_episode_steps=max_episode_steps), seed=seed)
