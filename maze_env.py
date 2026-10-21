"""Ant in a maze: a learned walker plus a planner.

The low level is the direction-commanded walker from nav_env.py. It never saw
a wall during training. The high level plans on the maze grid with A*,
smooths the path, and steers by pure pursuit: every step it points the
walker at a spot a couple of meters further along the path.

Comparison pilots, all in the same MuJoCo maze:
    random      random torques
    forward     the original SAC walker; it can only walk toward +x
    beeline     commanded walker aimed straight at the goal, no planning
    planner     commanded walker following the A* path
"""

from __future__ import annotations

import heapq
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import gymnasium.envs.mujoco as gym_mujoco
import mujoco
import numpy as np

from nav_env import DirectionalAnt, make_directional_env

CELL = 4.0
WALL_HEIGHT = 1.4
CLEARANCE = 1.3
LOOKAHEAD = 2.2
GOAL_RADIUS = 1.5
# Recovery: a learned gait can freeze in a stance that still earns the survival
# bonus. If the robot moves less than STUCK_METERS in STUCK_STEPS, the planner
# commands a sideways shuffle for RECOVER_STEPS, then resumes the path.
STUCK_STEPS = 60
STUCK_METERS = 0.3
RECOVER_STEPS = 25

MAZES: dict[str, list[str]] = {
    # The goal is close as the crow flies but behind a wall; the way round is long.
    "switchback": [
        "#########",
        "#S..#...#",
        "###.#.#.#",
        "#...#.#.#",
        "#.###.#.#",
        "#.....#G#",
        "#########",
    ],
    "spiral": [
        "#########",
        "#.......#",
        "#.#####.#",
        "#.#G..#.#",
        "#.###.#.#",
        "#S....#.#",
        "#########",
    ],
}


def cell_center(r: int, c: int, rows: int, cols: int) -> np.ndarray:
    return np.array([(c - cols / 2 + 0.5) * CELL, (rows / 2 - r - 0.5) * CELL])


def find(grid: list[str], ch: str) -> tuple[int, int]:
    for r, row in enumerate(grid):
        c = row.find(ch)
        if c >= 0:
            return r, c
    raise ValueError(f"{ch} not in maze")


def wall_boxes(grid: list[str]) -> list[tuple[float, float, float, float]]:
    """Merge runs of '#' in each row into boxes: (cx, cy, half_x, half_y)."""
    rows, cols = len(grid), len(grid[0])
    boxes = []
    for r, row in enumerate(grid):
        c = 0
        while c < cols:
            if row[c] != "#":
                c += 1
                continue
            start = c
            while c < cols and row[c] == "#":
                c += 1
            a = cell_center(r, start, rows, cols)
            b = cell_center(r, c - 1, rows, cols)
            boxes.append(((a[0] + b[0]) / 2, a[1], (c - start) * CELL / 2, CELL / 2))
    return boxes


def maze_xml(grid: list[str]) -> Path:
    """Stock ant.xml with the maze walls added and the torso moved to the start cell."""
    src = Path(gym_mujoco.__file__).parent / "assets" / "ant.xml"
    xml = src.read_text(encoding="utf-8")
    rows, cols = len(grid), len(grid[0])
    start = cell_center(*find(grid, "S"), rows, cols)
    walls = "\n".join(
        f'    <geom name="wall_{i}" type="box" pos="{x:.3f} {y:.3f} {WALL_HEIGHT / 2}" size="{hx:.3f} {hy:.3f} {WALL_HEIGHT / 2}" '
        f'contype="1" conaffinity="1" rgba="0.85 0.82 0.76 1"/>'
        for i, (x, y, hx, hy) in enumerate(wall_boxes(grid))
    )
    xml = xml.replace('<body name="torso" pos="0 0 0.75">', f'{walls}\n    <body name="torso" pos="{start[0]:.3f} {start[1]:.3f} 0.75">')
    out = Path(tempfile.gettempdir()) / f"ant_maze_{abs(hash(tuple(grid)))}.xml"
    out.write_text(xml, encoding="utf-8")
    return out


def astar(grid: list[str], start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    rows, cols = len(grid), len(grid[0])
    frontier = [(0, start)]
    came: dict[tuple[int, int], Optional[tuple[int, int]]] = {start: None}
    cost = {start: 0}
    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            break
        r, c = cur
        for nr, nc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if not (0 <= nr < rows and 0 <= nc < cols) or grid[nr][nc] == "#":
                continue
            g = cost[cur] + 1
            if g < cost.get((nr, nc), 1 << 30):
                cost[(nr, nc)] = g
                came[(nr, nc)] = cur
                heapq.heappush(frontier, (g + abs(goal[0] - nr) + abs(goal[1] - nc), (nr, nc)))
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = came.get(node)
    return path[::-1]


@dataclass
class Maze:
    name: str
    grid: list[str]
    rows: int = field(init=False)
    cols: int = field(init=False)
    start: np.ndarray = field(init=False)
    goal: np.ndarray = field(init=False)
    boxes: list = field(init=False)
    path: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rows, self.cols = len(self.grid), len(self.grid[0])
        s, g = find(self.grid, "S"), find(self.grid, "G")
        self.start = cell_center(*s, self.rows, self.cols)
        self.goal = cell_center(*g, self.rows, self.cols)
        self.boxes = wall_boxes(self.grid)
        cells = astar(self.grid, s, g)
        self.path = self._smooth(np.array([cell_center(r, c, self.rows, self.cols) for r, c in cells]))

    def clear(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Segment a-b keeps CLEARANCE from every wall box."""
        for t in np.linspace(0.0, 1.0, max(2, int(np.linalg.norm(b - a) / 0.25))):
            p = a + t * (b - a)
            for x, y, hx, hy in self.boxes:
                dx = max(abs(p[0] - x) - hx, 0.0)
                dy = max(abs(p[1] - y) - hy, 0.0)
                if np.hypot(dx, dy) < CLEARANCE:
                    return False
        return True

    def _smooth(self, pts: np.ndarray) -> np.ndarray:
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.clear(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return np.array(out)

    def pursuit(self, pos: np.ndarray) -> float:
        """Heading toward the point LOOKAHEAD meters past the closest point on the path."""
        best_d, best_s, s_acc = 1e9, 0.0, 0.0
        seg_len = []
        for a, b in zip(self.path[:-1], self.path[1:]):
            ab = b - a
            L = float(np.linalg.norm(ab))
            t = float(np.clip((pos - a) @ ab / max(L * L, 1e-9), 0, 1))
            d = float(np.linalg.norm(a + t * ab - pos))
            if d < best_d:
                best_d, best_s = d, s_acc + t * L
            s_acc += L
            seg_len.append(L)
        target_s = min(best_s + LOOKAHEAD, s_acc)
        acc = 0.0
        target = self.path[-1]
        for (a, b), L in zip(zip(self.path[:-1], self.path[1:]), seg_len):
            if acc + L >= target_s:
                target = a + (b - a) * (target_s - acc) / max(L, 1e-9)
                break
            acc += L
        d = target - pos
        return float(np.arctan2(d[1], d[0]))


def make_maze_env(maze: Maze, seed: Optional[int] = None, max_steps: int = 3000) -> DirectionalAnt:
    env = make_directional_env(seed=seed, xml_file=maze_xml(maze.grid), max_episode_steps=max_steps)
    env.set_command(0.0)
    return env


FEET = ("left_ankle_geom", "right_ankle_geom", "third_ankle_geom", "fourth_ankle_geom")


def foot_bits(model, data) -> int:
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FEET]
    bits = 0
    for c in data.contact[: data.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            if other in feet:
                bits |= 1 << feet.index(other)
    return bits


def wall_geom_ids(model) -> set[int]:
    return {i for i in range(model.ngeom) if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("wall_")}


Pilot = Callable[[np.ndarray, "DirectionalAnt", Maze], np.ndarray]


def run(
    maze: Maze,
    env: DirectionalAnt,
    act: Callable[[np.ndarray], np.ndarray],
    steer: str,
    seed: int,
    record: bool = False,
    recover: bool = True,
) -> dict:
    """steer: 'planner' (pure pursuit on the A* path), 'beeline' (straight at the goal), or 'none'."""
    obs, _ = env.reset(seed=seed)
    data = env.mj.data
    walls = wall_geom_ids(env.mj.model)
    frames, commands, positions = [], [], []
    contacts, torques, metrics = [], [], []
    total = 0.0
    touching = 0
    steps = 0
    reached = False
    fell = False
    done = False
    recovering = 0
    recoveries = 0
    window_from = 0
    side = 1.0
    while not done:
        pos = data.qpos[:2].copy()
        if steer == "planner" and recover and recovering == 0 and len(positions) - window_from > STUCK_STEPS:
            if np.linalg.norm(positions[-1] - positions[-STUCK_STEPS]) < STUCK_METERS:
                recovering = RECOVER_STEPS
                recoveries += 1
                side = -side
        if steer == "planner" and recovering > 0:
            env.set_command(maze.pursuit(pos) + side * np.pi / 2)
            recovering -= 1
            if recovering == 0:
                window_from = len(positions)  # restart the stuck window from here
        elif steer == "planner":
            env.set_command(maze.pursuit(pos))
        elif steer == "beeline":
            d = maze.goal - pos
            env.set_command(float(np.arctan2(d[1], d[0])))
        obs = np.concatenate([obs[:-2], env._command_obs()])
        action = np.clip(act(obs), -1.0, 1.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        steps += 1
        hit = any((c.geom1 in walls) or (c.geom2 in walls) for c in data.contact[: data.ncon])
        touching += int(hit)
        positions.append(data.qpos[:2].copy())
        if record:
            frames.append(np.concatenate([data.xpos[1:], data.xquat[1:]], axis=1).ravel())
            commands.append(env.command)
            contacts.append(foot_bits(env.mj.model, data))
            torques.append(np.round(action * 127).astype(np.int8))
            metrics.append([float(info.get("progress", 0.0)), float(data.qpos[2]), float(reward), total])
        if np.linalg.norm(data.qpos[:2] - maze.goal) < GOAL_RADIUS:
            reached = True
            done = True
        fell = fell or terminated
        done = done or terminated or truncated
    positions = np.array(positions)
    out = {
        "success": reached,
        "fell": fell and not reached,
        "steps": steps,
        "seconds": steps * env.mj.dt,
        "path_m": float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))) if len(positions) > 1 else 0.0,
        "closest_m": float(np.min(np.linalg.norm(positions - maze.goal, axis=1))),
        "wall_contact_s": touching * env.mj.dt,
        "recoveries": recoveries,
    }
    if record:
        out["poses"] = np.asarray(frames, dtype=np.float32)
        out["commands"] = np.asarray(commands, dtype=np.float32)
        out["track"] = positions.astype(np.float32)
        out["contacts"] = np.asarray(contacts, dtype=np.uint8)
        out["torques"] = np.asarray(torques, dtype=np.int8)
        out["metrics"] = np.asarray(metrics, dtype=np.float32)
    return out
