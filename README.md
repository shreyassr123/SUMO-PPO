# SUMO-PPO

A small, readable framework for reinforcement learning experiments in the
[SUMO](https://eclipse.dev/sumo/) traffic simulator, built around a
from-scratch implementation of Proximal Policy Optimisation (PPO).

It is intended as a starting point for student projects in traffic-signal
control, vehicle control and related traffic-management problems. 

---

## Contents

1. [Project overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Quick start](#4-quick-start)
5. [Understanding PPO](#5-understanding-ppo)
6. [Understanding SUMO and TraCI](#6-understanding-sumo-and-traci)
7. [Traffic-signal example](#7-traffic-signal-example)
8. [Vehicle-control example](#8-vehicle-control-example)
9. [Training and evaluation](#9-training-and-evaluation)
10. [How to modify the project](#10-how-to-modify-the-project)
11. [Results and reproducibility](#11-results-and-reproducibility)
12. [Troubleshooting](#12-troubleshooting)
13. [Scope and limitations](#13-scope-and-limitations)
14. [Origin, attribution and licence](#14-origin-attribution-and-licence)

---

## 1. Project overview

**What the repository does.** It connects a PPO learning agent to traffic
simulations run by SUMO, and provides two complete examples:

| Example | The agent controls | Actions | Default reward |
|---|---|---|---|
| A. Traffic-signal control | one traffic light at a four-way intersection | keep the current green, or switch to the next green | minus the number of queued vehicles |
| B. Vehicle speed control | one car on a single-lane road with other traffic | decelerate, maintain speed, accelerate | minus delay, unsafe following and harsh acceleration |

Both examples are trained by the *same* PPO code, together with baselines
(fixed-time signals, random actions, SUMO's own driver model) for comparison.

**Why PPO.** PPO is a widely used policy-gradient method. It learns directly
from interaction, works with discrete actions, is fairly robust to its
hyperparameters, and its core fits in a few hundred lines that can be read in
an afternoon.

**What SUMO contributes.** SUMO (Simulation of Urban MObility) is an
open-source *microscopic* traffic simulator: every vehicle is simulated
individually, with car-following, lane-changing and right-of-way rules on a
road network. It supplies the realistic environment in which the agent acts.

**What TraCI does.** TraCI (Traffic Control Interface) is SUMO's remote
control. A Python program connects to a running SUMO process, advances the
simulation one step at a time, reads its state (queues, speeds, positions)
and changes it (signal states, vehicle speeds).

**What you can learn from it.** How a reinforcement learning problem is
formulated (observations, actions, rewards, episodes); how PPO collects data
and updates its networks; how a traffic simulator is controlled from Python;
and how to evaluate a learned controller fairly against baselines.

---

## 2. Architecture

```text
SUMO simulation
      |
      v
Environment observation        envs/*_env.py : get_observation()
      |
      v
PPO policy                     ppo/agent.py  : PPOAgent.act()
      |
      v
Selected action (an integer)
      |
      v
TraCI command                  envs/*_env.py : step() / apply_action()
      |
      v
SUMO advances                  sumo.simulationStep()
      |
      v
Reward and next observation    envs/*_env.py : calculate_reward(), get_observation()
      |
      v
PPO rollout and update         ppo/rollout.py : collect_rollout()
                               ppo/agent.py   : update_policy()
```

The learning algorithm, the environments and the scenarios are kept
separate: PPO never sees SUMO, and the environments never see PPO.

```text
SUMO-PPO/
├── README.md
├── requirements.txt
├── pytest.ini
├── ppo/                       the learning algorithm (no SUMO code)
│   ├── networks.py            actor pi(a|s) and critic V(s) networks
│   ├── buffer.py              rollout storage and GAE (NumPy only)
│   ├── agent.py               PPOAgent: act, update_policy, checkpoints; PPOConfig
│   └── rollout.py             collect_rollout() for training, evaluate_policy() for testing
├── envs/                      Gymnasium environments (no PPO code)
│   ├── __init__.py            make_env(): choose an environment by name
│   ├── sumo_base.py           starting/stopping SUMO, seeding, finding SUMO
│   ├── traffic_signal_env.py  Example A
│   └── vehicle_control_env.py Example B
├── scenarios/                 SUMO input files
│   ├── build_networks.py      builds network.net.xml from the node/edge files
│   ├── simple_intersection/   nodes, edges, traffic demand, configuration
│   └── vehicle_control/
├── scripts/                   command-line programs
│   ├── train.py               train PPO
│   ├── evaluate.py            evaluate a saved checkpoint
│   ├── run_baseline.py        run a non-learning controller
│   └── common.py              CSV files, summaries and plots shared by the scripts
├── tests/                     automated tests (pytest)
└── docs/
    ├── ppo_explained.md       the PPO implementation, equation by equation
    ├── sumo_explained.md      SUMO, TraCI and the scenario files
    └── extending_the_framework.md
```

| Module | Responsibility |
|---|---|
| `ppo/networks.py` | Two small neural networks: the actor outputs one logit per action; the critic outputs one number, the estimated return. |
| `ppo/buffer.py` | `RolloutBuffer` stores one rollout and computes advantages with GAE. |
| `ppo/agent.py` | `PPOAgent` chooses actions, runs the PPO update and saves/loads checkpoints. `PPOConfig` holds the hyperparameters. |
| `ppo/rollout.py` | `RolloutCollector.collect_rollout()` runs the policy in the environment; `evaluate_policy()` runs complete test episodes for any controller. |
| `envs/sumo_base.py` | `SumoEnvironment` launches SUMO for each episode, closes it again, and hands each episode a reproducible random seed. |
| `envs/*_env.py` | The two examples: observation, action, reward and episode-ending rules. |
| `scripts/*.py` | Command-line entry points that connect the pieces. |

---

## 3. Installation

You need three things: Python packages, the SUMO program itself, and SUMO's
Python interface (TraCI). Installing the SUMO *program* (the simulator,
`sumo` and `sumo-gui`) is separate from making its *Python interface*
importable; both are needed.

### 3.1 Python and the Python packages

Python 3.9 or newer is required. From the project folder:

**Linux / macOS**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pip install torch` (pulled in by `requirements.txt`) installs a CPU build of
PyTorch, which is all these small networks need. If you want a specific
CUDA build, install PyTorch first with the command from
<https://pytorch.org/get-started/locally/>.

### 3.2 SUMO

Choose **one** of the two routes.

**Route 1: official installer or package manager (recommended; includes `sumo-gui`).**

* **Windows:** download the installer from <https://sumo.dlr.de/docs/Downloads.php>
  and run it. Then set `SUMO_HOME` to the installation folder and add its
  `bin` folder to `PATH`. The folder is usually
  `C:\Program Files (x86)\Eclipse\Sumo` or `C:\Program Files\Eclipse\Sumo`;
  check where your installer put it. In PowerShell:
  ```powershell
  setx SUMO_HOME "C:\Program Files (x86)\Eclipse\Sumo"
  ```
  Add `%SUMO_HOME%\bin` to `PATH` via *Start → "Edit the system environment
  variables" → Environment Variables → Path → Edit → New*. Open a **new**
  terminal afterwards: `setx` and PATH changes do not affect terminals that
  are already open.
* **Ubuntu/Debian:**
  ```bash
  sudo add-apt-repository ppa:sumo/stable
  sudo apt-get update
  sudo apt-get install sumo sumo-tools sumo-doc
  echo 'export SUMO_HOME=/usr/share/sumo' >> ~/.bashrc
  source ~/.bashrc
  ```

With this route the Python interface lives in `$SUMO_HOME/tools`. You do not
need to install it separately: the framework adds that folder to Python's
search path when `SUMO_HOME` is set (see `import_traci()` in
`envs/sumo_base.py`).

**Route 2: everything through pip.**
```bash
pip install eclipse-sumo traci
```
This installs the SUMO programs and TraCI into your virtual environment, and
`SUMO_HOME` is not needed. Depending on your platform the pip package may not
provide `sumo-gui`; use Route 1 if you need the graphical interface. To keep
TraCI and SUMO at the same version you can pin both, e.g.
`pip install eclipse-sumo==<version> traci==<version>`.

### 3.3 Check the installation

```bash
sumo --version
python -c "from envs.sumo_base import import_traci; print(import_traci().__file__)"
```

The first command prints SUMO's version; the second (run from the project
folder) prints where TraCI was found. The original research code used SUMO
1.28.0 on Windows. This framework only uses long-standing TraCI functions,
but it has **not yet been run against a particular SUMO version** (see
[Section 11](#11-results-and-reproducibility)); please record the version you
use with your results.

### 3.4 Build the road networks

The repository stores short, human-readable node and edge files; SUMO's
`netconvert` tool turns them into the full network files that the simulator
loads:

```bash
python scenarios/build_networks.py
```

This writes `scenarios/simple_intersection/network.net.xml` and
`scenarios/vehicle_control/network.net.xml`. Run it again whenever you edit
a `.nod.xml` or `.edg.xml` file. Once built, you may commit the network files
so that others can skip this step.

### 3.5 Run the tests

```bash
pytest
```

Tests that need PyTorch, Gymnasium or SUMO are skipped automatically, with a
reason, if those are missing. `pytest -rs` lists the reasons.

---

## 4. Quick start

All commands are run from the project folder with the virtual environment
active. `<run>` stands for the folder that `train.py` creates and prints, for
example `results/traffic_signal_seed0_20260110-142501`.

```bash
# 0. (Optional) check PPO on its own, without SUMO. Random actions keep the
#    CartPole pole up for about 20 steps; the maximum is 500. A working PPO
#    should show returns rising well above 20 over the run.
python scripts/train.py --env cartpole --episodes 300

# 1. Watch a basic SUMO simulation (sumo-gui, signal runs its built-in fixed program)
sumo-gui -c scenarios/simple_intersection/simulation.sumocfg

# 2. Run a baseline controller (fixed-time signal, 10 evaluation episodes)
python scripts/run_baseline.py --env traffic_signal --controller fixed_time

# 3. Train PPO for traffic-signal control
python scripts/train.py --env traffic_signal --episodes 200

# 4. Evaluate the trained signal controller on new traffic
python scripts/evaluate.py --checkpoint <run>/checkpoints/final.pt

# 5. Train PPO for vehicle control (episodes are short, so use more of them)
python scripts/train.py --env vehicle_control --episodes 1000

# 6. Evaluate the trained vehicle controller and compare with SUMO's driver
python scripts/evaluate.py --checkpoint <run>/checkpoints/final.pt
python scripts/run_baseline.py --env vehicle_control --controller sumo

# 7. With and without the graphical interface: add --gui to any script
python scripts/run_baseline.py --env vehicle_control --controller sumo --episodes 1 --gui
python scripts/evaluate.py --checkpoint <run>/checkpoints/final.pt --episodes 1 --gui
```

Without `--gui`, SUMO runs headless (no window), which is much faster and is
what you want for training. With `--gui`, a SUMO window opens, the simulation
starts by itself and the window closes at the end of each episode; use the
delay control in the toolbar to slow it down.

The episode counts above are starting points, not tuned values: these
experiments have not been run yet (see [Section 11](#11-results-and-reproducibility)).

All options of a script are listed with `--help`:

| Script | Main options |
|---|---|
| `train.py` | `--env`, `--episodes`, `--seed`, `--lr`, `--critic-lr`, `--gamma`, `--rollout-steps`, `--eval-episodes`, `--output-dir`, `--gui` |
| `evaluate.py` | `--checkpoint`, `--episodes`, `--seed`, `--stochastic`, `--output`, `--gui`, `--env` |
| `run_baseline.py` | `--env`, `--controller`, `--episodes`, `--seed`, `--fixed-green`, `--output-dir`, `--gui` |

---

## 5. Understanding PPO

This section gives the ideas; [docs/ppo_explained.md](docs/ppo_explained.md)
maps every equation to the code.

* **Observation** *s*: the numbers the agent sees at a decision point, e.g.
  the queue length on each approach.
* **Action** *a*: the decision it takes, here an integer such as 0 = keep the
  current green, 1 = switch.
* **Reward** *r*: a single number returned after every action that says how
  good the outcome was. The agent maximises the (discounted) sum of rewards,
  the **return** *G = r₀ + γr₁ + γ²r₂ + …*, where the discount factor γ (0.99)
  makes near-term rewards count slightly more.
* **Episode**: one run from `reset()` until the task ends (*terminated*, e.g.
  the car arrives) or is cut off by a time limit (*truncated*). Here one
  episode is one SUMO simulation.
* **Policy** π(a | s): the agent's strategy, a probability for each action
  given the observation. During training actions are *sampled* from it, which
  provides exploration.
* **Value function** V(s): the expected return from state *s* when following
  the current policy.
* **Actor and critic**: two neural networks. The *actor* represents the
  policy; the *critic* estimates the value function. The critic is only used
  during training, to judge the actor's actions.
* **Rollout**: a batch of experience (by default 2048 steps, possibly
  spanning several episodes) collected with the current policy before each
  update.
* **Advantage** A(s, a): how much better an action turned out than the critic
  expected. Positive advantage: make that action more likely; negative: less
  likely.
* **GAE** (Generalised Advantage Estimation) computes advantages from the
  one-step errors δₜ = rₜ + γV(sₜ₊₁) − V(sₜ):
  Âₜ = δₜ + γλδₜ₊₁ + (γλ)²δₜ₊₂ + …; λ = 0.95 trades bias against variance.
* **PPO clipping**: with rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ), PPO maximises
  min(rₜÂₜ, clip(rₜ, 1−ε, 1+ε)Âₜ). Once the new policy differs from the one
  that collected the data by more than ε = 0.2 in probability ratio, that
  sample stops pushing the policy further, which keeps each update small and
  stable.
* **Policy update**: after each rollout, the actor and critic are trained for
  10 passes ("epochs") over the rollout in shuffled minibatches; then the old
  data are discarded, because they no longer describe the current policy.

---

## 6. Understanding SUMO and TraCI

More detail, including how to view the scenarios, is in
[docs/sumo_explained.md](docs/sumo_explained.md).

* **Road network** (`network.net.xml`): junctions, roads (*edges*), their
  lanes, the allowed movements between lanes (*connections*) and the default
  traffic-light programs. It is generated by `netconvert` from simple node
  (`*.nod.xml`) and edge (`*.edg.xml`) files.
* **Route file** (`traffic.rou.xml`): vehicle types, routes (lists of edges)
  and the vehicles or *flows* (streams of vehicles) that drive them.
* **Configuration file** (`simulation.sumocfg`): which network and route files
  to load, the simulation step length and other options. Launching SUMO with
  `-c simulation.sumocfg` loads everything.
* **Time steps**: SUMO advances in discrete steps; both scenarios use 1 s. An
  RL decision can cover several steps (5 s in the signal example).
* **Traffic demand**: here, flows with an insertion *probability* per second,
  so the number and timing of vehicles change with SUMO's random seed.
* **TraCI communication**: each environment starts its own SUMO process with
  `traci.start(...)`, then repeatedly sends commands and calls
  `simulationStep()`. SUMO waits for the client between steps, so Python
  fully controls the pace.
* **Traffic-light control**: the signal environment writes complete signal
  states with `trafficlight.setRedYellowGreenState()`, one character per
  controlled movement (`G` green with priority, `g` green that must yield,
  `y` yellow, `r` red).
* **Vehicle control**: the vehicle environment sets a desired speed with
  `vehicle.setSpeed()`; SUMO still applies its safety checks.

How Python and SUMO interact during one `env.step(action)`:

```text
Python (environment)                      SUMO process
--------------------                      ------------
send the action (signal state / speed) -->
simulationStep()                       --> advance 1 s, move every vehicle
read queues, speeds, positions        <--  current state
... repeat for the decision interval ...
compute reward, build observation
return (observation, reward, terminated, truncated, info) to PPO
```

---

## 7. Traffic-signal example

File: `envs/traffic_signal_env.py`. Scenario: `scenarios/simple_intersection/`.

**Network.** One signalised four-way intersection (`center`) with four
200 m approaches, one lane per direction, speed limit 50 km/h (13.89 m/s).
Incoming edges are `north_in`, `east_in`, `south_in`, `west_in`; outgoing
edges are `*_out`. U-turns are not built.

**Traffic demand.** Twelve flows, one per origin–destination pair, inserted
with a fixed probability per second between t = 0 and t = 3600 s.
North–south traffic is about twice as heavy as east–west:

| Approach | Straight | Left | Right | Total |
|---|---|---|---|---|
| north, south | 0.10 veh/s | 0.015 | 0.015 | 468 veh/h each |
| east, west | 0.05 veh/s | 0.01 | 0.01 | 252 veh/h each |

**Observation** (all values in [0, 1], `float32`):

| Index | Meaning | Normalisation |
|---|---|---|
| 0–3 | halted vehicles on `east_in`, `north_in`, `south_in`, `west_in` | ÷ approach capacity (lane length ÷ 7.5 m), clipped to 1 |
| 4–5 | current green phase, one-hot, in the order of the network's signal program (expected: one phase for the north–south movements, one for east–west) | — |
| 6 | time since the current green started | ÷ 60 s, clipped to 1 |

A vehicle counts as halted when its speed is below 0.1 m/s (SUMO's
definition). The green phases are read from the network's signal program, so
the observation size is 4 + *P* + 1, where *P* is the number of green phases
(2 expected for this network; `train.py` prints the size it found).

**Actions.** `0` = keep the current green; `1` = request the next green
phase (phases are used in a fixed cycle).

**Signal transitions.** PPO only *requests* a change. The environment
enforces the safety rules, and the agent cannot skip them:

1. A request is ignored until the current green has lasted at least
   **10 s** (minimum green).
2. Movements that lose their green get **3 s yellow**.
3. Then all signals are red for **2 s** (all-red clearance).
4. Then the next green phase starts.

**Decision frequency.** The agent decides once per **5 s** of green. A step in
which a switch happens therefore covers 3 + 2 + 5 = 10 simulated seconds.

**Reward.** For every simulated second *k* of a step,
the reward rate is −Σᵢ qᵢ(k), where qᵢ is the number of halted vehicles on
approach *i* (`calculate_reward()`). The step reward is the sum of these
rates over all seconds of the step, i.e. minus the vehicle-seconds spent
queueing during that step. The return over an episode is therefore minus the
total queueing time of all vehicles at the intersection.

**Episode.** 3600 simulated seconds, then the episode is **truncated** (the
traffic would have continued, so this is a time limit rather than an end of
the task). It is **terminated** early only if SUMO reports no vehicles left
or still to come.

**Evaluation metrics** (in `info`, written to the CSV files):

| Metric | Meaning |
|---|---|
| `mean_queue_length` | average number of halted vehicles on all approaches (veh) |
| `mean_waiting_time` | average total time spent halted per vehicle, over vehicles that completed their trip (s) — SUMO's *accumulated waiting time*, reported separately from the reward |
| `throughput` | vehicles that completed their trip during the episode |
| `phase_switches` | number of green-to-green transitions |
| `teleports` | vehicles SUMO removed after being stuck for 300 s (should be 0; non-zero values indicate gridlock) |

**Example of one complete step.** Suppose that at t = 40 s green phase 0 has
been on for 15 s and the agent chooses action `1`:

```text
t = 40-43 s   yellow for the movements that lose their green  (reward accumulates -sum(q) each second)
t = 43-45 s   all red
t = 45-50 s   green phase 1
t = 50 s      new observation: queues, phase one-hot = [0, 1], elapsed = 5/60
              reward = -(sum of halted vehicles over the 10 seconds)
```

Had the agent chosen action `1` after only 5 s of green, the request would
have been ignored and the step would simply have been 5 more seconds of
phase 0.

In code:

```python
from envs import make_env

env = make_env("traffic_signal")            # or TrafficSignalEnv(min_green=15, ...)
observation, info = env.reset(seed=0)
observation, reward, terminated, truncated, info = env.step(1)
print(observation, reward, info["time_in_green"])
env.close()
```

**Baselines.** `fixed_time` requests a switch whenever the green has lasted
`--fixed-green` seconds (default 30 s), using the same yellow and all-red
rules; `random` chooses keep or switch with equal probability.

---

## 8. Vehicle-control example

File: `envs/vehicle_control_env.py`. Scenario: `scenarios/vehicle_control/`.

**Road network.** A straight 1.5 km road made of two 750 m edges
(`road_1`, `road_2`), one lane, one direction, 50 km/h. There is no room to
overtake.

**Controlled vehicle.** The vehicle `ego` (red in the GUI) departs at
t = 60 s and drives to the end of `road_2`. Its vehicle type has no random
driver imperfection and a speed factor of exactly 1, so its allowed speed is
the speed limit.

**Background traffic.** Ordinary cars (0.06 veh/s, speeds spread around the
limit) and slow vans (0.015 veh/s, at most 30 km/h) are inserted from
t = 0 and driven by SUMO's default car-following model (Krauss). The ego
vehicle therefore usually catches up with slower traffic.

**Observation** (5 values, `float32`):

| Index | Meaning | Range |
|---|---|---|
| 0 | own speed ÷ allowed speed | [0, 1] |
| 1 | gap to the leader ÷ 100 m | [0, 1] |
| 2 | (leader speed − own speed) ÷ allowed speed | [−1, 1] |
| 3 | 1 if a leader is within 100 m, else 0 | {0, 1} |
| 4 | remaining route distance ÷ route length | [0, 1] |

**Missing leader.** If no vehicle is within 100 m ahead, the observation
contains gap = 1 (100 m), relative speed = 0 and leader flag = 0. The flag lets
the network distinguish "no leader" from "a leader exactly 100 m away".

**Actions** and **speed commands.** Every simulation step (1 s):

| Action | Meaning | Desired speed sent to SUMO |
|---|---|---|
| 0 | decelerate | current speed − 2.0 m/s |
| 1 | maintain | current speed |
| 2 | accelerate | current speed + 1.0 m/s |

The desired speed is clipped to [0, allowed speed] and sent with
`vehicle.setSpeed()`.

**Safety constraints.** The ego vehicle runs with speed mode 31, SUMO's
default, which keeps all of SUMO's checks active: safe following speed,
maximum acceleration and deceleration, right of way and braking at red
lights. What PPO overrides is only the *desired* speed of the ego vehicle;
SUMO still reduces the speed when the desired speed would cause a rear-end
collision. All other vehicles are entirely under SUMO's control. Do not lower
the speed mode unless you deliberately want to study collisions.

**Reward** (`calculate_reward()`), per step:

r = −(w_d · delay + w_s · safety + w_c · comfort)

| Term | Definition | Weight |
|---|---|---|
| delay | 1 − speed ÷ allowed speed | w_d = 1.0 |
| safety | max(0, 1 − headway ÷ 1.5 s), headway = gap ÷ speed; 0 without leader | w_s = 1.0 |
| comfort | (acceleration ÷ 2 m/s²)² using SUMO's actual acceleration | w_c = 0.1 |

A collision adds −10 and ends the episode. Progress is expressed as *less
delay* on purpose: a positive reward per step would make arriving early
*end* a stream of positive rewards, so the agent would learn to dawdle (see
[docs/extending_the_framework.md](docs/extending_the_framework.md)). The step
in which the vehicle arrives gives reward 0, since there is nothing left to
measure. With these definitions, driving at the speed limit with at least
1.5 s headway and smooth acceleration is the best possible behaviour.

**Termination rules.**

* **terminated** when the vehicle reaches the end of its route, collides,
  or disappears from the simulation for any other reason. The environment
  checks SUMO's arrived and collision lists *before* making any further TraCI
  call about the vehicle, so it never queries a vehicle ID that no longer
  exists.
* **truncated** after 300 s of driving.

**Evaluation metrics**: `travel_time` (s), `mean_speed` (m/s),
`mean_abs_acceleration` (m/s²), `min_time_headway` (s; NaN if there was never a
leader in view), `short_headway_steps` (steps with headway below 1.5 s),
`arrived`, `collided`.

**Baselines.** `sumo` lets SUMO's own driver model drive the ego vehicle
(the agent's actions are ignored, but observations, rewards and metrics are
computed in exactly the same way); `random` picks actions uniformly.

---

## 9. Training and evaluation

**Starting training.** `python scripts/train.py --env traffic_signal`.
Each run creates `results/<env>_seed<seed>_<date-time>/` containing:

| File | Contents |
|---|---|
| `config.json` | command, seed and all PPO hyperparameters |
| `training_episodes.csv` | return, length and environment metrics for every training episode |
| `updates.csv` | per update: policy loss, value loss, entropy, approximate KL, clip fraction, explained variance |
| `training_curves.png` | episode return (with moving average), critic loss and policy entropy |
| `checkpoints/final.pt` | the final networks; `update_0010.pt`, `update_0020.pt`, … every 10 updates |
| `evaluation.csv` | greedy evaluation of the final policy on seeds 10000+ (unless `--eval-episodes 0`) |

**What an episode represents.** One SUMO simulation from start to time limit
(one hour of traffic for the signal example, one trip for the vehicle).

**How PPO collects rollouts and when updates happen.**
`RolloutCollector.collect_rollout()` runs the current policy for
`--rollout-steps` decisions (default 2048), starting new SUMO simulations as
episodes end. Then `PPOAgent.update_policy()` trains on that data and the
cycle repeats until `--episodes` episodes have finished. An episode may
start in one rollout and end in the next. The final rollout is cut short at
the last episode, so exactly `--episodes` episodes are run.

**What the hyperparameters control.** See `PPOConfig` in `ppo/agent.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `gamma` | 0.99 | discount factor; higher values look further ahead |
| `gae_lambda` | 0.95 | GAE parameter; lower values reduce variance but add bias |
| `clip_range` | 0.2 | ε in the clipped objective; how far one update can move the policy |
| `actor_lr`, `critic_lr` | 3e-4, 1e-3 | Adam learning rates |
| `rollout_steps` | 2048 | decisions per update |
| `update_epochs` | 10 | passes over each rollout |
| `minibatch_size` | 256 | samples per gradient step |
| `entropy_coef` | 0.01 | bonus for keeping the policy uncertain (exploration) |
| `max_grad_norm` | 0.5 | gradient clipping threshold |
| `target_kl` | 0.02 | stop actor updates on a rollout once the approximate KL exceeds this |

`--lr`, `--critic-lr`, `--gamma` and `--rollout-steps` can be set on the
command line; change the others in `PPOConfig`.

**Loading a model.**

```python
from ppo.agent import PPOAgent
agent, metadata = PPOAgent.load_checkpoint("results/<run>/checkpoints/final.pt")
action, log_prob, value = agent.act(observation, deterministic=True)
```

The checkpoint stores the network sizes and hyperparameters, so no other
information is needed. `evaluate.py` refuses a checkpoint whose
observation or action size no longer matches the environment.

**Evaluating on a new seed or traffic demand.** `evaluate.py --seed 20000`
uses different random traffic. To test a different demand, edit or copy
`traffic.rou.xml`, point a copy of `simulation.sumocfg` at it, and create the
environment in Python with `config_file=` set to that copy. Evaluation is greedy (most probable
action) unless `--stochastic` is given.

**Interpreting training curves.** The return should trend upwards, with a lot
of noise, because traffic differs between episodes. Policy entropy should
decrease slowly as the policy becomes more decided; if it collapses to near 0
early, the agent may have stopped exploring. A critic loss that keeps growing,
or a KL that triggers early stopping on every update, suggests that the
learning rates are too high. `updates.csv` also has `explained_variance`:
values near 1 mean the critic predicts returns well, values near or below 0
mean it does not.

**Training versus evaluation performance.** Training returns are measured
while the policy is still changing and still sampling random actions, on
traffic that it may have adapted to. The meaningful comparison is an
*independent* evaluation: a fixed policy, deterministic actions, and seeds
not used for training. `evaluate.py` and `run_baseline.py` use the same
seeds by default (10000, 10001, …), so PPO and the baselines are evaluated
on identical traffic.

---

## 10. How to modify the project

[docs/extending_the_framework.md](docs/extending_the_framework.md) works
through these changes in more detail.

**Changing observations.** Edit `get_observation()` in the environment and
update `observation_space` in its `__init__` to the new length and bounds.
The PPO networks read the size from `env.observation_space.shape[0]` when
`train.py` creates the agent, so nothing in `ppo/` changes. Old checkpoints
no longer fit and must be retrained. Keep values roughly in [−1, 1] and
always finite.

**Changing actions.** Three places: the action constants and their
interpretation (`KEEP_PHASE`/`SWITCH_PHASE`, or `ACTION_ACCELERATIONS`), the
`action_space` in `__init__`, and the code that turns an action into TraCI
commands (`step()`/`change_to_next_green()` for signals, `apply_action()` for
the vehicle). PPO adapts to any number of discrete actions.

**Changing rewards.** Edit `calculate_reward()`. It must return one finite
number. Every change alters what "good" means, so check the new reward
against the behaviour you want (e.g. a waiting-time reward favours long-waiting
vehicles, a throughput reward ignores fairness), compare controllers on the
evaluation *metrics*, not just on the return, and remember that returns from
different reward definitions are not comparable.

**Changing traffic demand.** Edit the `probability` of the flows in
`traffic.rou.xml` (vehicles per second), add flows, or change their
`begin`/`end` times. Keep vehicles and flows ordered by departure time. Larger
demand files can be generated with SUMO's `randomTrips.py` tool
(`$SUMO_HOME/tools/randomTrips.py`).

**Changing the road network.** Edit the `.nod.xml` and `.edg.xml` files and
rerun `python scenarios/build_networks.py`, or create a new scenario folder
with its own files and add it to `SCENARIOS` in `build_networks.py`. You can
also draw networks in SUMO's `netedit`. Then check the identifiers used by
the environment: the traffic light ID (`tls_id="center"`), the edge names
used in routes, and the controlled vehicle ID (`vehicle_id="ego"`). The
signal environment discovers the approaches and green phases of the light
itself, so a different intersection layout usually works without code
changes.

**Changing PPO hyperparameters.** Use the command-line options of
`train.py` or edit the defaults in `PPOConfig` (table in Section 9).

**Adding a new environment.** Subclass `SumoEnvironment`, define the spaces
and four methods, and register the class in `envs/__init__.py`:

```python
# envs/my_env.py
import numpy as np
from gymnasium import spaces
from .sumo_base import SCENARIO_DIR, SumoEnvironment

class MyEnv(SumoEnvironment):
    def __init__(self, gui=False, config_file=SCENARIO_DIR / "my_scenario" / "simulation.sumocfg"):
        super().__init__(config_file, gui=gui)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def on_reset(self):
        self.start_time = self.simulation_time

    def get_observation(self):
        return np.zeros(3, dtype=np.float32)          # read SUMO state via self.sumo

    def get_info(self):
        return {}

    def step(self, action):
        action = self.check_action(action)
        # ... send TraCI commands for the action via self.sumo ...
        self.sumo.simulationStep()
        reward = 0.0                                  # your calculate_reward()
        truncated = self.simulation_time - self.start_time >= 600
        return self.get_observation(), reward, False, truncated, self.get_info()
```

```python
# envs/__init__.py
from .my_env import MyEnv
SUMO_ENVIRONMENTS = {..., "my_env": MyEnv}
```

After that, `python scripts/train.py --env my_env` works.

---

## 11. Results and reproducibility

**No experimental results are reported in this repository.** It provides
the implementation and the tools to train, evaluate and compare
controllers; it makes no claim that PPO outperforms the baselines.

**Verification status at the time of writing.** The code was developed in
an environment without SUMO and without PyTorch. Verified there:

* all Python files compile, and pass `ruff`/`flake8` checks;
* the GAE and return calculations (`tests/test_buffer.py`) and the rollout
  bookkeeping (`tests/test_rollout.py`);
* the environments' own logic, against scripted TraCI stand-ins
  (`tests/test_environments.py`): minimum green, yellow and all-red sequencing,
  reward accounting, speed commands and bounds, arrival, collision and
  time-limit handling, repeated resets;
* consistency of the scenario XML files (`tests/test_scenarios.py`);
* `run_baseline.py` end to end against the scripted stand-ins.

**Not yet run:** the PyTorch tests and the CartPole check
(`tests/test_ppo.py`), the real-SUMO integration tests (`tests/test_sumo.py`),
`build_networks.py`, and any training or evaluation with SUMO. Run
`pytest -rs` after installation and update this section with what you
observe.

**Practices for your own experiments.**

* **Random seeds.** `--seed` seeds Python, NumPy and PyTorch and the
  environment's random generator, which in turn chooses SUMO's `--seed` for
  every episode. The same command on the same software versions should give
  the same results (PyTorch does not guarantee bit-identical results across
  platforms or versions).
* **Repeated experiments.** RL results vary a lot between seeds. Train with
  several seeds (e.g. `--seed 0`, `1`, `2`, `3`, `4`) and report the mean and
  spread, not the best run.
* **Independent evaluation seeds.** Evaluate on seeds not used in training
  (the default 10000+ is far from the training seeds), and use the same
  evaluation seeds for every controller.
* **Metrics.** Report traffic metrics (queue length, waiting time,
  throughput; travel time, headways) as well as the return.
* **Checkpoint selection.** `final.pt` is the policy at the end of training.
  If you choose among the periodic checkpoints, do so on a *separate*
  validation seed range, and report the final numbers on yet another one;
  choosing the checkpoint with the best training return favours lucky
  episodes.
* **Training plots** (`training_curves.png`) show learning progress, not
  final performance.
* **Result directories.** Each run has its own folder; `results/` is ignored
  by git.
* **Limitations of training reward.** A higher training return does not by
  itself mean a better controller: training episodes include exploration,
  the reward may not capture everything you care about, and different reward
  definitions give incomparable returns.

---

## 12. Troubleshooting

| Problem | What to do |
|---|---|
| `Could not find the SUMO program 'sumo'` | SUMO is not installed or its `bin` folder is not on `PATH`. Check `sumo --version` in a *new* terminal; set `SUMO_HOME` (Section 3.2). |
| `SUMO_HOME` not set | Needed only for Route 1. Windows: `setx SUMO_HOME "C:\…\Eclipse\Sumo"` then open a new terminal. Linux: add `export SUMO_HOME=/usr/share/sumo` to `~/.bashrc`. Check with `echo $SUMO_HOME` / `echo %SUMO_HOME%`. |
| `Could not import TraCI` | Either set `SUMO_HOME` so that `$SUMO_HOME/tools` exists, or `pip install traci`. The error message shows the current value of `SUMO_HOME`. |
| `SUMO network file not found … network.net.xml` | Run `python scenarios/build_networks.py`. |
| `SUMO configuration file not found` / invalid paths | Paths in `simulation.sumocfg` are relative to that file's folder. Run scripts from the project folder. |
| `SUMO failed to start` | The error shows the exact command. Run it in a terminal to see SUMO's own message (often a mistake in an XML file). |
| Port or connection problems (`Could not connect to TraCI server`, connection refused) | `traci.start` picks a free port automatically. A firewall or security software can block local connections; allow Python and SUMO. Close stray SUMO processes from crashed runs (Task Manager / `pkill sumo`). |
| SUMO closes unexpectedly (`connection closed by SUMO`) | Usually an error in the scenario or closing the GUI window during an episode. Run the command from the error message by hand; check `sumo -c <config>` alone. |
| Missing vehicle identifiers (`Vehicle 'x' is not known`) | A vehicle left the simulation before being queried. Check `getIDList()` or the arrived list before querying it, as `VehicleControlEnv.step()` does. If `ego` never appears, check its `depart` time and route in `traffic.rou.xml`. |
| No vehicles appear | Check that `simulation.sumocfg` lists the route file, that flows have a `probability` or `period` and an `end` time after `begin`, and that route edges exist. Vehicles that are not sorted by departure time are skipped. |
| Traffic-light actions do not change phases | A switch is only carried out after the minimum green (10 s); `info["phase_switches"]` counts actual changes. Check that `tls_id` matches the junction ID in the network. |
| NaN losses during PPO training | `PPOAgent` stops with a `FloatingPointError`. Check observations and rewards for NaN/inf (the collector checks observations), scale very large values, or lower the learning rates. |
| `Observation has N values but the agent expects M` / checkpoint does not match | The environment's observation or action definition changed after training. Retrain, or restore the definition used for the checkpoint. |
| GUI versus headless operation | Without `--gui` nothing is shown, by design. With `--gui`, `sumo-gui` must be installed (the pip package may lack it). On a remote Linux machine without a display, use headless mode. |
| Training is slow | Make sure `--gui` is not set. Most of the time is spent in SUMO and TraCI, not in PyTorch. |

---

## 13. Scope and limitations

What the examples do **not** provide:

* no multi-agent PPO (one agent controls one signal or one vehicle);
* no continuous-action PPO (actions are discrete);
* no recurrent policies; the agent sees only the current observation;
* no automatic optimisation of arbitrary traffic networks;
* no guarantee of collision-free learned control if SUMO's safety checks are
  changed or the environment differs from the one used in training;
* no claim that PPO outperforms a carefully designed baseline;
* the signal example uses a single intersection with simple demand; vehicles
  that cannot be inserted because a queue has reached the start of an approach
  wait outside the network and are not counted in the queue observation.

Possible extensions (not implemented): waiting-time or pressure rewards,
richer observations, several intersections, continuous accelerations,
multi-agent learning, lane changing. See
[docs/extending_the_framework.md](docs/extending_the_framework.md).

---

## 14. Origin, attribution and licence

The PPO learner is adapted from the PPO baseline (`22_09_PPO_baseline.py`)
of the Mobility_PPO research project, which studied multimodal travel
recommendations for event visitors. The event-specific parts of that project
(traveller decisions, arrival-time rewards, mode choice, public-transport
and parking logic, action masks) were removed; the learning algorithm was
restructured into a general, episode-based PPO.

