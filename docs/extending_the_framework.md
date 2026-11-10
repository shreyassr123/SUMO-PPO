# Extending the framework

This guide is for starting your own project on top of the two examples. The
PPO code in `ppo/` should not need changes for any of the steps below; all
problem-specific decisions live in an environment class.

## 1. A workflow for a new project

1. **Define the decision problem in words.** Who decides, how often, what
   can they observe, what can they change, and what counts as success?
2. **Build or adapt a scenario** in `scenarios/` and look at it in
   `sumo-gui` before writing any RL code. Check that traffic actually flows
   and that the situation you want to control occurs.
3. **Write the environment** (Section 2) and run it with a simple
   rule-based controller and with random actions. If a sensible rule does
   not beat random actions on your metrics, the reward or the scenario is
   probably not measuring what you think.
4. **Validate the environment** (Section 6).
5. **Train PPO** with a few seeds, then evaluate it and the baselines on the
   same independent seeds.
6. **Change one thing at a time** (reward, observation, demand) and keep a
   note of every run's `config.json`.

## 2. Creating a new environment class

Start from the template in README Section 10, or copy the example closest to
your problem. A subclass of `SumoEnvironment` provides:

| Method / attribute | Purpose |
|---|---|
| `__init__` | call `super().__init__(config_file, gui=gui)`; define `observation_space` (a `gymnasium.spaces.Box`) and `action_space` (a `gymnasium.spaces.Discrete`) |
| `on_reset()` | set up episode state after SUMO has started (counters, IDs, warm-up steps) |
| `get_observation()` | return a `float32` NumPy array matching `observation_space` |
| `step(action)` | validate the action (`check_action`), send TraCI commands, advance SUMO, compute reward and episode end, return the usual 5-tuple |
| `get_info()` | return a dict of metrics; numeric entries are saved to the CSV files automatically |

Register the class in `SUMO_ENVIRONMENTS` in `envs/__init__.py`. If its
baselines need extra settings, add them to `scripts/run_baseline.py`.

Useful base-class helpers: `self.sumo` (the TraCI connection),
`self.simulation_time`, `self.step_length`, `self.steps_for(seconds)`.

## 3. Modifying observations

Add the new quantity in `get_observation()` and extend `observation_space`.
For example, to give the signal agent the waiting time on each approach:

```python
waiting = np.array([self.sumo.edge.getWaitingTime(e) for e in self.incoming_edges])
waiting = np.minimum(waiting / 600.0, 1.0)          # scale: 600 vehicle-seconds -> 1
return np.concatenate([queues, phase, [elapsed], waiting]).astype(np.float32)
```

and in `__init__` add `len(self.incoming_edges)` to `n_values`.

Guidelines:

* scale every value to roughly [−1, 1] or [0, 1] with a fixed, documented
  constant (not with statistics of the current episode);
* include what the agent needs to choose well, and avoid information it could
  not have in reality unless you are studying an idealised controller;
* when a quantity can be missing (no leader, no vehicle on a lane), encode
  that explicitly, e.g. with an indicator value, instead of an arbitrary
  number;
* the agent has no memory: if timing matters, put it in the observation (as
  with the elapsed green time).

## 4. Defining actions

Actions are integers `0 … n−1`. Keep the meaning of each in named constants,
translate them to TraCI commands in one place, and let the environment
enforce physical and safety constraints, so that no action can produce an
unsafe or impossible state. Examples:

* **signal control with phase selection:** `Discrete(len(self.green_states))`
  where action *k* means "the next green should be phase *k*"; the
  transition code must then build the yellow state from the current to the
  chosen phase (`yellow_state(current, chosen)` already does this);
* **vehicle control with finer steps:** more entries in `ACTION_ACCELERATIONS`,
  e.g. {−3, −1.5, −0.5, 0, 0.5, 1.5}; update `action_space` to match;
* **lane changing:** an extra action calling
  `self.sumo.vehicle.changeLane(vehicle_id, lane_index, duration)` on a
  multi-lane road.

Continuous actions (e.g. an acceleration in m/s²) would need a Gaussian policy
in `ppo/networks.py`; this is a possible extension, not part of the current
code.

## 5. Implementing rewards

`calculate_reward()` must return one finite number. Some options for the
signal example:

| Objective | Reward rate per second | Comment |
|---|---|---|
| queue length (default) | −Σᵢ qᵢ | simple; ignores how long individual vehicles wait |
| waiting time | −Σᵢ `edge.getWaitingTime(eᵢ)` | favours vehicles that have waited long; note SUMO's waiting time is *accumulated*, so it grows while a queue persists |
| change in waiting time | (W_before − W_after) per step | rewards reductions instead of levels |
| throughput | + vehicles that crossed the stop line | can ignore fairness between approaches |
| pressure | −Σ (queue in − queue out) per movement | used in max-pressure control |

Points to check whenever you design a reward:

* **Does the best behaviour under the reward match what you want?** For
  example, a *positive* reward per step for progress in an episode that ends
  on arrival is a trap: arriving early ends the stream of positive rewards,
  so a slow agent can collect as much, or more, reward. This is why the
  vehicle example penalises *delay* (each step costs something unless the
  vehicle drives at the limit) instead of rewarding progress.
* **Do terms conflict?** If one term rewards speed and another penalises
  short headways, the optimum is a compromise; make sure the weights express
  the trade-off you intend and document it.
* **Is the scale reasonable?** Value normalisation in `PPOAgent` copes with
  large rewards, but a term that is 1000 times larger than the others will
  dominate.
* **Are all quantities defined in every state?** Handle missing vehicles,
  empty lanes and division by zero.
* **Report metrics, not just the reward.** Keep the evaluation metrics in
  `get_info()` independent of the reward definition, so that runs with
  different rewards remain comparable.

## 6. Validating a modified environment

Before training:

* run `pytest`; add tests for your own logic. `tests/fake_sumo.py` shows how
  to replace SUMO with a scripted stand-in to test, for instance, that a
  safety rule is never violated;
* check the Gymnasium API:
  ```python
  from gymnasium.utils.env_checker import check_env
  from envs import make_env
  env = make_env("my_env")
  check_env(env, skip_render_check=True)
  env.close()
  ```
* run an episode with random actions and one with a sensible rule
  (`scripts/run_baseline.py`), and look at it with `--gui`;
* print a few observations and rewards and check them by hand: signs, ranges,
  units, and what happens at the start and end of an episode;
* make sure episodes end (terminated or truncated) and that the time limit is
  reported as `truncated`, not `terminated`.

After training, look at `training_curves.png` and `updates.csv`, then
evaluate on new seeds and compare with the baselines on the same seeds.

## 7. Example research directions

* **Reward design for signals:** compare queue, waiting-time and pressure
  rewards on the same evaluation seeds; study fairness between approaches.
* **Demand robustness:** train on one demand level and evaluate on heavier,
  lighter or time-varying demand.
* **Decision frequency:** how do the decision interval and minimum green
  affect learning and performance?
* **Observation design:** what is the smallest observation that still works?
  Does adding upstream detectors help?
* **Several intersections:** extend the scenario to a corridor and control one
  light with PPO while the others run fixed time; then consider one agent per
  light (multi-agent learning is not implemented).
* **Eco-driving:** add an energy or emissions term to the vehicle reward
  (SUMO provides `vehicle.getCO2Emission` and `getFuelConsumption`) and study
  the trade-off with travel time.
* **Approaching a traffic light:** combine the two examples, giving the
  controlled vehicle the distance to a signal and its state.
* **Hyperparameter sensitivity:** how stable are the results across seeds and
  settings such as γ, the rollout length and the entropy coefficient?
