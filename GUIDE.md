# Quadruped RL: the full story

This is the long version of the README: what the project does, how each piece works, why it was built that way, and how to answer the questions you are likely to get. Read it top to bottom once, then use the section headings to revise.

---

## 1. The project in 30 seconds

> I trained a simulated four-legged robot in stages. First I compared two RL algorithms, SAC and PPO, on learning to walk. Then I changed the task so one network can walk in any commanded direction. Finally I put that walker inside mazes and drove it with a classic A* planner. The learned walker plus the planner reached the goal in 19 of 20 runs; every simpler pilot reached it 0 times. The website replays real recorded episodes of every stage.

The point it makes: **learning and planning do different jobs**. RL is good at the low-level skill (coordinating eight motors so the body moves without falling), and a planner is good at the high-level decision (which way to go around walls). Splitting the problem this way is a standard design in robotics, and the results show why.

---

## 2. Stages at a glance

| Stage | What it is | Algorithm | Key result |
| --- | --- | --- | --- |
| 1. Untrained | Random torques | none | Return −43, falls every time |
| 2. Walking | Run forward as fast as possible | SAC and PPO | SAC 5,100, tuned PPO 4,111 |
| 3. Steering | Walk in whatever heading is commanded | PPO | 1.89 m/s along the command vs 0.13 for the stage 2 walker |
| 4. Maze | Stage 3 walker steered by A* + pure pursuit | planning on top of RL | 19/20 goals; random, forward-only and beeline pilots: 0/20 |

Each stage exists because the previous one can't solve the next problem. That is how the website and README are organised.

---

## 3. RL in five ideas

You only need these to explain everything else.

1. **MDP.** At each step the agent sees an observation *s*, picks an action *a*, gets a reward *r*, and the world moves on. The goal is to maximise the discounted sum of future rewards (the *return*), with discount γ = 0.99 here.
2. **Policy.** A neural network π(a | s). For continuous control it outputs the mean and standard deviation of a Gaussian over actions. At evaluation time we take the mean (deterministic).
3. **Value function.** A second network that predicts how much return to expect from a state. It is used to judge whether an action was better or worse than usual (the *advantage*).
4. **Actor-critic.** The policy is the actor, the value network is the critic. Both SAC and PPO are actor-critic methods.
5. **On-policy vs off-policy.** On-policy (PPO) learns only from data collected by the current policy and then throws it away. Off-policy (SAC) stores every transition in a replay buffer and reuses it many times.

---

## 4. The robot and simulator

- **MuJoCo** is a physics engine used across robotics research: rigid bodies, joints, contacts, friction.
- **Gymnasium `Ant-v4`** is the standard quadruped benchmark built on it: a torso and four legs, each with a hip and an ankle, so **8 motors**.
- **Observation (27 numbers):** torso height and orientation, joint angles, and all velocities. The x/y position is left out, so the policy can't memorise locations.
- **Action (8 numbers in [−1, 1]):** a torque for each motor.
- **Control rate:** 20 Hz (MuJoCo steps at 100 Hz, and each action is held for 5 physics steps).
- **Reward:** forward velocity + 1 per step for staying healthy − 0.5 × the squared action size (a cost on effort).
- **Episode end:** 1,000 steps (50 s), or earlier if the torso height leaves 0.2–1.0 m (it fell, or flipped).

Why Ant: it is hard enough that the gait has to be learned (eight coupled joints, contacts), it is the benchmark everyone knows, and it runs on a laptop.

---

## 5. Stage 1: the random baseline

Uniform random torques. It scores −43 and falls in all 5 episodes. A baseline matters because a "return of 1,000" sounds good until you know that standing still for 1,000 steps earns exactly that from the survival bonus. Every later number should be read against it.

---

## 6. Stage 2: learning to walk with SAC and PPO

### How SAC works
Soft Actor-Critic is off-policy and *maximum entropy*. Besides return, it is rewarded for keeping its policy random (high entropy), with a weight α tuned automatically to hit a target entropy.

- It keeps a replay buffer (1M transitions) and does one gradient update per environment step, sampling batches of 256.
- It has two Q-networks and uses the smaller estimate, which reduces the overestimation that breaks many value-based methods.
- Target networks are slow-moving copies (updated with τ = 0.005) that keep the learning targets stable.

Why it suits Ant: sample efficiency. Reusing every transition many times matters when simulating is the slow part.

### How PPO works
Proximal Policy Optimization is on-policy.

- It collects a fresh batch of experience (here 8 environments × 1,024 steps), computes advantages with GAE (λ = 0.95), and then takes several passes over the batch (10 epochs, minibatches of 256).
- The key trick is the **clipped objective**: if an update would change the probability of an action by more than ±20%, the gradient is cut off. This keeps each update small and safe without the second-order maths of TRPO.

Why include it: PPO is the most widely used on-policy method, it parallelises well, and comparing it with SAC is a fair question about sample efficiency versus stability.

### Settings and the PPO fix

| | SAC | PPO (tuned) |
| --- | --- | --- |
| Networks | 256×256 | 256×256, separate actor and critic |
| Steps | 1M | 2M |
| Hardware | RTX 4060, 2.7 h | CPU, 35 min |
| Normalisation | none | observations and rewards (VecNormalize) |

The first PPO run (default settings, initial action std 1.0) scored 218 and fell every time.

- **Why it failed:** with an action std of 1.0, the random torques are so large that the torso is thrown above 1.0 m within a few dozen steps, which ends the episode. The policy never saw long enough episodes to discover walking.
- **The fix:**
  - start with a smaller action noise (log σ = −1, about 0.37);
  - use separate 256×256 networks for actor and critic;
  - use longer rollouts.
- **Result:** at the same 1M steps, the return went up about 15×.

This is a good story to tell: the problem was diagnosed from the episode lengths and termination reasons, not by blind hyperparameter search.

**Why VecNormalize for PPO only:** PPO's advantage estimates and value loss are sensitive to the scale of observations and rewards, and the SB3/RL-Zoo setups for MuJoCo normalise for PPO. SAC trains well on Ant without it.

### What the results mean

| Policy | Return | Speed | Falls |
| --- | --- | --- | --- |
| SAC | 5,100 ± 165 | 5.2 m/s | 0/5 |
| PPO tuned | 4,111 ± 978 | 4.0 m/s | 1/5 |

- SAC is higher and more consistent here. **One seed each, so this is a result of this run, not a ranking of the algorithms.** Say that before anyone else does.
- Compare at equal *environment steps* for sample efficiency, and at equal *wall-clock time* for practicality. PPO's steps are much cheaper.
- **Gait:** early SAC checkpoints (50k–200k steps) stand still and collect the survival bonus. That is a local optimum the reward allows. Forward motion appears between 200k and 400k steps.
- The final gait is a bound: all four feet are off the ground 66% of the time. Contacts are counted over every physics sub-step, not just the 20 Hz control steps, so short touches aren't missed.

### Robustness
`export_web.py` scales the torso mass (0.6–1.4×) and floor friction (0.4–1.6×) inside the loaded MuJoCo model and re-runs both policies. Neither saw these changes in training.

- **Mass barely matters.**
- **More friction breaks both gaits:** at 1.3× or more, returns fall to about 1,400–1,700. Both gaits rely on the feet *sliding*, so with more grip the feet catch and the body trips. On low-friction "ice" both actually go faster.
- **A 120 N sideways shove** for 0.1 s knocks both over often.

Why it matters: this is the sim-to-real question in miniature. A policy can be excellent in nominal conditions and exploit quirks of the simulator.

---

## 7. Stage 3: steering on command

**Problem:** the stage 2 reward only pays for +x velocity, so those policies can go one way only. In the steering test the SAC walker moves 0.13 m/s along random commanded headings while drifting 3.38 m/s sideways (it just runs in its own direction).

**Change the task, not the algorithm.** `nav_env.py` wraps Ant:

- Each episode samples a heading, and it changes every 250–500 steps (12–25 s), so the policy must learn to turn mid-stride, not just to pick a direction at the start.
- The command is added to the observation as (cos, sin) of the heading **relative to the torso's yaw**. Relative, so the same network works whichever way it is facing. Cos/sin, so there is no jump at ±180°.

The reward, term by term, and why each term is there:

| Term | Why |
| --- | --- |
| + progress along the command (m/s) | The actual goal |
| − 0.5 × sideways speed | Without it, moving diagonally still earns partial credit |
| − 0.3 × vertical torso speed² | Removes the bounding gait, which is fast but wild and hard to steer |
| − 0.05 × roll/pitch rate² | Keeps the body level, so turns don't tip it |
| + 1 survival, − 0.5 × action² | Same as Ant-v4 |

**Training:** PPO with the tuned preset, 12 parallel environments, 8M steps, 42 min on CPU. Checkpoints were saved with their normalisation statistics, because a PPO policy trained with VecNormalize is useless without the matching observation scaling.

**Why PPO here and not SAC:** commands change during the episode, which makes the problem more non-stationary for replay-based learning. PPO parallelises cheaply on CPU (no GPU was needed), and its tuned preset was already known to work on Ant. This is a practical choice, not a claim that SAC can't do it.

**Result:**

| | Along command | Drift | Falls |
| --- | --- | --- | --- |
| Stage 2 SAC | 0.13 m/s | 3.38 m/s | 1/10 |
| Stage 3 PPO | 1.89 m/s | 0.47 m/s | 0/10 |

Slower than the 5.2 m/s sprinter. That is the price of a controllable gait, and it is the right trade for navigation.

---

## 8. Stage 4: mazes with a planner

### Why not train RL end to end?
You could reward reaching the goal and hope the policy learns both walking and wall-avoidance. It would need far more training, the reward is sparse, and the result would only work in the mazes it trained on. A hierarchy reuses a skill that already works and adds reasoning on top. The walker **never saw a wall** during training.

### The pieces (`maze_env.py`)

1. **Maze geometry.**
   - Mazes are text grids with 4 m cells.
   - Walls are box geoms (1.4 m tall) injected into the Ant's MuJoCo XML, so they are real physics, not drawn on top.
2. **A\***.
   - Shortest path on the grid of free cells (4-connected, with a Manhattan heuristic, which is admissible on that grid, so the path is optimal).
3. **Path smoothing.**
   - Grid paths have sharp corners.
   - From each kept point, the planner jumps to the furthest later grid point it can reach in a straight line that stays at least 1.3 m from every wall (a line-of-sight check). This gives fewer, gentler turns.
4. **Pure pursuit.**
   - At every control step, find the closest point on the path and look 2.2 m further along it.
   - The commanded heading is the direction to that point.
   - This is the classic path-tracking method from autonomous vehicles: smooth, and robust to the walker drifting off the line.
5. **Stuck recovery.**
   - Some walker checkpoints occasionally freeze in a stance that still earns the survival bonus (a local optimum again).
   - If the robot moves less than 0.3 m in 60 steps (3 s), the planner commands a sideways shuffle for 25 steps, then resumes the path.
   - This is the standard "recovery behaviour" pattern in mobile robotics (ROS navigation has the same idea).

### Fair comparison
Every pilot runs in the same mazes, from the same starts, with the same seeds (100–109), with up to 3,000 steps (150 s) to reach within 1.5 m of the goal:

| Pilot | What it shows |
| --- | --- |
| Random torques | Floor |
| Stage 2 forward walker | Can't turn, walks into the first wall |
| Stage 3 walker aimed straight at the goal ("beeline") | Steering alone is not enough; it pins itself against a wall |
| Stage 3 walker + planner | The full system |

**Checkpoint selection:** `maze_sweep.py` tried the stage 3 checkpoints on seeds 0–5 and picked the one at 6M steps, the best on those seeds (the final 8M checkpoint was not). The reported results use seeds 100–109. Picking and reporting on the same seeds would overstate the result; this is the same idea as a validation/test split.

**Result:** switchback (88 m route) 10/10 in 49 s on average; spiral (32 m route) 9/10 in 20 s; every other pilot 0/20.

---

## 9. The website

- **Why replays instead of running MuJoCo live:** a static site is free to host, has no server or cold starts, and shows exactly the episodes the numbers come from.
- **How the recordings are made:** `export_web.py` and `export_nav.py` load each checkpoint, run it in MuJoCo with deterministic actions, and record at 20 Hz:
  - every body's position and orientation (float32);
  - foot contacts as bits;
  - joint torques (int8);
  - for navigation, the command and the maze.
- **How the site plays them:** three.js poses the same MuJoCo bodies, so the robot you see is the real recorded motion.
- **Tests:** `web/src/*.test.ts` check the contract between exporter and viewer (binary sizes, unit quaternions, contact bits, and that the steerable walker out-steers the forward one). Render runs them before every build, so a bad export can't be deployed.
- **What's on the page:** the gait diagram, which shows feet on the ground over time, and the torque strip, which shows effort per joint. These are the standard ways locomotion people look at a gait.

---

## 10. Questions you should be ready for

**Why compare only one seed per algorithm?**
Compute budget. I state it as a limit and don't claim SAC is better than PPO in general. The right next step is 5+ seeds with confidence intervals.

**Isn't the planner doing all the work?**
No. The "beeline" pilot uses the same walker without the planner and gets 0/20. The forward walker with any planner would get 0, because it can't turn. You need both.

**Why is the stage 3 walker slower than stage 2?**
Its reward penalises bouncing, sideways drift and tilt, which trades top speed for control. A navigation stack needs a walker that goes where it is told.

**What does the recovery behaviour hide?**
Without it, some runs freeze. I report it openly; recovery behaviours are standard in real navigation stacks. A better fix would be to penalise standing still during training.

**Why does the robot fail on high friction?**
Both stage 2 gaits learned to slide their feet, which only works with the default friction. That's simulator exploitation; domain randomisation during training (varying friction and mass) is the usual cure.

**How would this transfer to a real robot?**
- *What it doesn't have:* actuator latency, sensor noise, rough terrain or perception, and the planner is given the map.
- *What you'd need:* domain randomisation, a state estimator, and a mapping/perception layer (e.g. SLAM).
- *What already maps across:* the hierarchical structure is exactly what real legged-robot navigation uses.

**Why A\* and not RRT or a learned planner?**
The maze is a known grid, and A* is optimal and simple there. RRT is for continuous, high-dimensional spaces; a learned planner would need training data and would generalise worse.

---

## 11. Limits and next steps

- One seed per configuration; more seeds with confidence intervals.
- Domain randomisation during training, to fix the friction weakness.
- Perception instead of a known map (a local occupancy grid from simulated lidar).
- Rough terrain (MuJoCo heightfields).
- Train the walker with a penalty for standing still, so recovery is rarely needed.

---

## 12. Words to know

- **Return:** sum of (discounted) rewards in an episode.
- **Deterministic evaluation:** using the policy's mean action, no sampling.
- **GAE:** Generalized Advantage Estimation, a low-variance estimate of how much better an action was than expected.
- **Entropy (SAC):** how random the policy is; SAC is rewarded for keeping it high.
- **VecNormalize:** running mean/std normalisation of observations and rewards.
- **Pure pursuit:** steer toward a point a fixed distance ahead on the path.
- **Hierarchical control:** a high-level planner gives goals to a low-level controller.
- **Sim-to-real gap:** the difference between simulated and real physics that breaks policies on hardware.
