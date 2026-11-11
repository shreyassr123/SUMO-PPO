# The example environments

Exact definitions of the observations, actions, rewards, safety rules,
episodes and metrics of the environments in `envs/`.

## 1. Traffic-signal control

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
| 0–3 | halted vehicles on `east_in`, `north_in`, `south_in`, `west_in` | ÷ approach capacity (length of its traffic lanes ÷ 7.5 m), clipped to 1 |
| 4–5 | current green stage, one-hot, in the order of the network's signal program (one stage for the north–south movements, one for east–west) | — |
| 6 | time since the current green started | ÷ maximum green (60 s), clipped to 1 |

A vehicle counts as halted when its speed is below 0.1 m/s (SUMO's
definition). The green stages are read from the network's signal program
(`read_signal_layout()`), so the observation size is *A* + *P* + 1 for *A*
approaches and *P* green stages; here 4 + 2 + 1 = 7, which `train.py` prints
at the start. Two program phases that differ only in their pedestrian signals
count as one stage (this matters on the city map, Section 1.1).

**Actions.** `0` = keep the current green; `1` = request the next green
stage (stages are used in a fixed cycle).

**Signal transitions.** PPO only *requests* a change. The environment
enforces the safety rules, and the agent cannot skip them:

1. A request is ignored until the current green has lasted at least
   **10 s** (minimum green).
2. A green that has lasted **60 s** (maximum green) is ended whatever the
   agent chooses, so that no approach can be starved. Set `max_green=None`
   to remove this rule.
3. Movements that lose their green get **3 s yellow**.
4. Then **2 s all-red** clearance: every movement is red, except movements
   that are green in both the old and the new stage, which keep their green.
5. Then the next green stage starts.

When an episode starts, the environment waits until SUMO's own program shows
a green stage and takes over in that stage, so control never begins with an
abrupt signal change.

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
| `mean_queue_length` | average number of halted vehicles on all approaches (veh) — the quantity the reward penalises |
| `mean_delay` | average delay per vehicle at the junction (s): time on the approach minus the time needed to drive it at the speed limit. Vehicles still waiting at the end count with their delay so far |
| `max_delay` | the largest such delay (s); a large value reveals an approach that is being starved |
| `throughput` | vehicles that left the approaches, i.e. crossed the junction, during the episode |
| `phase_switches` | number of green-to-green changes |
| `teleports` | vehicles SUMO removed after being stuck for 300 s, anywhere in the network (should be 0 on the small intersection; non-zero values indicate gridlock) |

All metrics are measured at the controlled junction, so they mean the same
on the small intersection and on the city map.

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

**Baselines** (`scripts/run_baseline.py`):

* `fixed_time` requests a switch whenever the green has lasted
  `--fixed-green` seconds (default 30 s), using the same yellow and all-red
  rules as PPO;
* `sumo_program` leaves the junction to the fixed program stored in the
  network file, with that program's own timings (the environment's
  `sumo_controls_signal=True`);
* `random` chooses keep or switch with equal probability.

### 1.1 The same task on the city map

Environment name: `city_traffic_signal`. Scenario: `scenarios/city_grid/`
(details and file origins in `scenarios/city_grid/README.md`).

This is the road network and traffic of the original Mobility_PPO project:
a 3.8 km × 3.6 km grid with **200 signalised junctions**, two traffic lanes
plus a pavement per direction, and pedestrian crossings. Traffic comes from
an activity-based model of **20,000 people** whose cars, motorbikes and
commercial vehicles leave home with them, park between activities and
reroute around congestion, plus 6 bus lines and 4 metro lines. The event
travellers of the original project are not included.

PPO controls **junction 517**, the busiest four-way junction in the morning;
the other 199 signals run their fixed programs. The task definition is
unchanged: same observation layout (4 approaches + 2 green stages + 1 = 7
values), actions, safety rules, reward and metrics.

| Setting | Value |
|---|---|
| Approaches (observation order) | `-1259` from the south, `-1263` from the west, `-1267` from the north, `518` from the east; two traffic lanes, about 180 m each |
| Green stages | 0 = east–west, 1 = north–south (each with pedestrians crossing in parallel) |
| Simulated day starts | 06:30 (set by `begin` in `simulation.sumocfg`) |
| Warm-up | 30 minutes under the junction's own program, not recorded |
| Episode | 07:00–08:00 under the agent's control, then truncated |

Things to know before using it:

* **It is much slower.** Every person, parked car and public-transport vehicle
  is simulated, so an episode takes far longer than on the small
  intersection. Measure the time of one episode first
  (`run_baseline.py --env city_traffic_signal --controller sumo_program --episodes 1`)
  and size your training runs accordingly. `scenarios/city_grid/README.md`
  lists the settings to remove for a faster, less detailed simulation.
* **The network starts empty at 06:30** because people whose trips began
  earlier are skipped; the warm-up fills it. Change `warmup_seconds`,
  `episode_seconds` or the `begin` time to study other periods, e.g.
  `make_env("city_traffic_signal", warmup_seconds=600, episode_seconds=1800)`.
* **Other junctions** can be controlled with `tls_id`, e.g.
  `make_env("city_traffic_signal", tls_id="783")`. 159 of the 200 signals
  have the two green stages this requires; T-junctions give an observation
  with 3 approaches, so their checkpoints are not interchangeable.
* `sumo_program` is the natural baseline here: it is how the junction
  operates in the rest of the simulated city.
* **Junction 517 is lightly loaded between 07:00 and 08:00.** Under its own
  program the average queue was about 3 vehicles and the average delay about
  20 s, so there is limited room for improvement. In a 2-episode smoke test
  the barely trained policy switched as often as the minimum green allows
  (240 changes per hour). That gave a slightly shorter queue but a longer
  worst-case delay, and such frequent switching would not be acceptable in
  practice. If you study this junction, consider a longer `min_green`, a
  switching penalty in `calculate_reward()`, a busier time window or more
  demand.
* Teleports are counted over the whole city, so they may occur even under
  the junction's own program; compare them between controllers rather than
  expecting zero.

---

## 2. Vehicle speed control

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
[extending_the_framework.md](extending_the_framework.md)). The step
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
