"""Ant-v4 environment factory and domain-randomization wrapper.

Randomizes torso mass and per-leg (and floor) sliding friction at every
reset. This is a lightweight stand-in for the sim-to-real gap: a policy
that only ever saw the default physics can fail when those parameters
move, while a policy trained with randomization is forced to be less
brittle.
"""

from __future__ import annotations

try:
    import windows_mujoco  # noqa: F401  - Windows MuJoCo plugin workaround
except Exception:
    pass

from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import Wrapper

ENV_ID = "Ant-v4"

# Default ranges used throughout the project. Friction is a multiplier on
# the XML sliding friction; mass is a multiplier on the torso body mass.
DEFAULT_FRICTION_RANGE: Tuple[float, float] = (0.5, 1.5)
DEFAULT_MASS_RANGE: Tuple[float, float] = (0.7, 1.3)


def _unwrap_mujoco(env: gym.Env):
    """Return the underlying Gymnasium MuJoCo environment."""
    return env.unwrapped


class DomainRandomizationWrapper(Wrapper):
    """Randomize Ant-v4 torso mass and geom sliding friction on reset.

    Parameters
    ----------
    friction_range:
        Inclusive multiplier range applied independently to each geom's
        sliding friction (MuJoCo `geom_friction[:, 0]`).
    mass_range:
        Inclusive multiplier range applied to the torso body mass and
        matching principal inertia, so mass scaling stays physically
        consistent at first order.
    seed:
        Optional RNG seed for the randomization itself (independent of
        the environment's action/observation seed).
    """

    def __init__(
        self,
        env: gym.Env,
        friction_range: Sequence[float] = DEFAULT_FRICTION_RANGE,
        mass_range: Sequence[float] = DEFAULT_MASS_RANGE,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(env)
        if len(friction_range) != 2 or friction_range[0] <= 0 or friction_range[1] < friction_range[0]:
            raise ValueError(f"Invalid friction_range: {friction_range}")
        if len(mass_range) != 2 or mass_range[0] <= 0 or mass_range[1] < mass_range[0]:
            raise ValueError(f"Invalid mass_range: {mass_range}")

        self.friction_range = (float(friction_range[0]), float(friction_range[1]))
        self.mass_range = (float(mass_range[0]), float(mass_range[1]))
        self._rng = np.random.default_rng(seed)

        mujoco_env = _unwrap_mujoco(env)
        self._model = mujoco_env.model
        self._orig_friction = np.array(self._model.geom_friction, copy=True)
        self._orig_mass = np.array(self._model.body_mass, copy=True)
        self._orig_inertia = np.array(self._model.body_inertia, copy=True)
        self._torso_id = self._find_torso_id()
        self.last_randomization: dict[str, Any] = {}

    def _find_torso_id(self) -> int:
        names = []
        for i in range(self._model.nbody):
            try:
                name = self._model.body(i).name
            except Exception:
                name = ""
            names.append(name)
            if name == "torso":
                return i
        # Body 0 is the world; body 1 is the torso in the stock Ant XML.
        return 1 if self._model.nbody > 1 else 0

    def _randomize_physics(self) -> None:
        mass_scale = float(self._rng.uniform(*self.mass_range))
        self._model.body_mass[:] = self._orig_mass
        self._model.body_inertia[:] = self._orig_inertia
        self._model.body_mass[self._torso_id] = self._orig_mass[self._torso_id] * mass_scale
        self._model.body_inertia[self._torso_id] = self._orig_inertia[self._torso_id] * mass_scale

        friction = self._orig_friction.copy()
        # Independent sliding-friction multiplier per geom (legs + floor).
        # Torsional and rolling friction keep their XML ratio to sliding.
        geom_scales = self._rng.uniform(
            self.friction_range[0], self.friction_range[1], size=(self._model.ngeom,)
        )
        friction[:, 0] = self._orig_friction[:, 0] * geom_scales
        # Keep torsion/roll in the same proportion as the original XML.
        orig_slide = np.maximum(self._orig_friction[:, 0], 1e-8)
        friction[:, 1] = self._orig_friction[:, 1] * (friction[:, 0] / orig_slide)
        friction[:, 2] = self._orig_friction[:, 2] * (friction[:, 0] / orig_slide)
        self._model.geom_friction[:] = friction

        self.last_randomization = {
            "torso_mass_scale": mass_scale,
            "torso_mass": float(self._model.body_mass[self._torso_id]),
            "friction_scales": geom_scales.tolist(),
            "mean_friction_scale": float(np.mean(geom_scales)),
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        obs, info = self.env.reset(seed=seed, options=options)
        self._randomize_physics()
        info = dict(info)
        info["domain_randomization"] = dict(self.last_randomization)
        return obs, info

    def restore_defaults(self) -> None:
        """Write the original XML mass/friction back into the model."""
        self._model.body_mass[:] = self._orig_mass
        self._model.body_inertia[:] = self._orig_inertia
        self._model.geom_friction[:] = self._orig_friction


def make_ant_env(
    *,
    render_mode: Optional[str] = None,
    randomize: bool = False,
    friction_range: Sequence[float] = DEFAULT_FRICTION_RANGE,
    mass_range: Sequence[float] = DEFAULT_MASS_RANGE,
    max_episode_steps: int = 1000,
    seed: Optional[int] = None,
    xml_file: Optional[str | Path] = None,
    width: int = 480,
    height: int = 480,
    camera_name: Optional[str] = None,
) -> gym.Env:
    """Create a monitored-ready Ant-v4 env, optionally with DR."""
    kwargs: dict[str, Any] = {
        "render_mode": render_mode,
        "max_episode_steps": max_episode_steps,
        "width": width,
        "height": height,
    }
    if xml_file is not None:
        kwargs["xml_file"] = str(Path(xml_file).resolve())
    if camera_name is not None:
        kwargs["camera_name"] = camera_name
    env = gym.make(ENV_ID, **kwargs)
    if randomize:
        env = DomainRandomizationWrapper(
            env,
            friction_range=friction_range,
            mass_range=mass_range,
            seed=seed,
        )
    return env
