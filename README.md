# SUMO-PPO

A Python framework for training and evaluating Proximal Policy Optimisation (PPO) agents in [SUMO](https://eclipse.dev/sumo/). It provides traffic-signal and vehicle-control examples that share the same PPO implementation. The code is intended as a starting point for student projects: observations, actions, rewards and scenarios can be changed without editing the learning algorithm.

## Environments

| Environment | Scenario | Agent actions | Reward | Baselines |
| --- | --- | --- | --- | --- |
| `traffic_signal` | Single four-way intersection | Keep the current green or request the next phase | Negative queueing time | `fixed_time`, `sumo_program`, `random` |
| `city_traffic_signal` | Larger city-grid network | Traffic-signal control | See [environment definitions](docs/environments.md) | See [environment definitions](docs/environments.md) |
| `vehicle_control` | Single-lane road with background traffic | Decelerate, maintain speed or accelerate | Penalties for delay, short headway and harsh acceleration | `sumo`, `random` |

The `city_traffic_signal` example uses the larger scenario in `scenarios/city_grid/` (approximately 30 MB), which includes a road network, activity demand and public-transport files. It is separate from the small intersection example. Its exact observations, rewards and control rules are documented in [docs/environments.md](docs/environments.md).

## Repository structure

```text
SUMO-PPO/
├── ppo/              PPO networks, rollout buffer, optimisation and evaluation
├── envs/             SUMO environments and TraCI connection management
├── scenarios/
│   ├── simple_intersection/
│   ├── vehicle_control/
│   ├── city_grid/
│   └── build_networks.py
├── scripts/          Training, evaluation and baseline commands
├── tests/            Automated tests
├── docs/             Environment definitions and extension guide
├── requirements.txt
└── README.md
```

Each environment defines its observation and action spaces, reward calculation and episode-ending rules. The `ppo/` package is independent of SUMO-specific control logic.

## Installation

**Requirements:** Python 3.9 or newer, SUMO with `netconvert`, and the Python dependencies in `requirements.txt`.

1. Clone the repository:

   ```bash
   git clone https://github.com/shreyassr123/SUMO-PPO.git
   cd SUMO-PPO
   ```

2. Create a virtual environment and install the Python dependencies.

   **Windows (PowerShell):**

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   **Linux / macOS:**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Install [SUMO](https://sumo.dlr.de/docs/Downloads.php). Set `SUMO_HOME` to its installation directory and add the `bin` directory to `PATH`; the code loads TraCI from `SUMO_HOME/tools`. Where supported, you can instead install the simulator and TraCI using `python -m pip install eclipse-sumo traci`.

4. Check the installation and build the two small networks:

   ```bash
   sumo --version
   netconvert --version
   python scenarios/build_networks.py
   ```

   The builder generates `network.net.xml` for `simple_intersection` and `vehicle_control`. The city-grid network and its supporting files are supplied separately in `scenarios/city_grid/`; the small-network builder does not regenerate the city grid.

## Running the examples

Run commands from the repository root with your Python environment activated. Commands run SUMO without a graphical window by default; use `--gui` where supported to inspect a simulation.

### Single-intersection signal control

```bash
# View the intersection using the signal programme stored in its network
sumo-gui -c scenarios/simple_intersection/simulation.sumocfg

# Compare against baseline controllers
python scripts/run_baseline.py --env traffic_signal --controller fixed_time
python scripts/run_baseline.py --env traffic_signal --controller sumo_program

# Train PPO
python scripts/train.py --env traffic_signal --episodes 200
```

The observation contains incoming queues, the current green phase and time in that phase. The agent requests to keep or switch the green. Signal changes are constrained by **a 10 s minimum green, a 60 s maximum green, 3 s of yellow and 2 s of all-red clearance**. The reward is negative queueing time accumulated over the simulation steps advanced by an action. The default episode lasts up to 3,600 simulated seconds. Reported metrics include queue length, `mean_delay`, `max_delay`, throughput and phase switches.

The `sumo_program` baseline follows the traffic-light programme embedded in the SUMO network; `fixed_time` uses a fixed green duration with the environment's transition rules.

### City-grid signal control

```bash
# View the supplied city-grid network and traffic
sumo-gui -c scenarios/city_grid/simulation.sumocfg

# Train a PPO controller for the city environment
python scripts/train.py --env city_traffic_signal --episodes 200
```

The city scenario is larger and includes activity-based demand, buses and metro services. Consult [docs/environments.md](docs/environments.md) for the controlled signals, observations, action timing, reward and evaluation metrics. Because it uses a more complex network, start by checking the SUMO configuration and running a short simulation before training.

### Vehicle speed control

```bash
# SUMO's built-in driver baseline
python scripts/run_baseline.py --env vehicle_control --controller sumo

# Train PPO
python scripts/train.py --env vehicle_control --episodes 1000

# View one baseline episode
python scripts/run_baseline.py --env vehicle_control --controller sumo --episodes 1 --gui
```

One vehicle travels along a 1.5 km single-lane road among background traffic. Its observation includes speed, leader gap, relative speed, a leader-presence flag and remaining distance. The agent chooses a speed change once per second, with SUMO's normal safety checks enabled. An episode ends on **arrival, collision, other removal from the simulation, or the 300 s driving limit**. Evaluation records travel time, acceleration, headway, arrival and collision outcomes.

### Evaluate a trained checkpoint

Each training run writes its settings, episode metrics, training curves, evaluation results and checkpoints under a timestamped `results/` directory. Find the actual run directory printed by `train.py`, then replace `YOUR_RUN_FOLDER` here:

```bash
python scripts/evaluate.py --checkpoint "results/YOUR_RUN_FOLDER/checkpoints/final.pt" --episodes 10
```

For a different evaluation seed range, add `--seed 20000`. Use the same evaluation seeds for PPO and baseline comparisons. For a PPO-only check without SUMO:

```bash
python scripts/train.py --env cartpole --episodes 300
```

Use `--help` with any script to see its supported options.

## Adapting the project

| Change | Where to start |
| --- | --- |
| Observation variables | `get_observation()` and `observation_space` in the relevant `envs/` module |
| Control actions | `action_space` and the action-to-TraCI mapping in the environment |
| Reward | `calculate_reward()` in the environment |
| Traffic demand or road network | Files in `scenarios/`; rebuild the small networks after editing nodes or edges |
| PPO hyperparameters | Options in `scripts/train.py` or `PPOConfig` in `ppo/agent.py` |
| Another environment | Create an environment module and register it in `envs/__init__.py` |

Changing observation or action dimensions requires retraining, because existing checkpoints will no longer match. See [Environment definitions](docs/environments.md) for the exact observations, rewards and safety rules, and [Extending the framework](docs/extending_the_framework.md) for examples of modifications. Further implementation details are in [PPO notes](docs/ppo_explained.md) and [SUMO/TraCI notes](docs/sumo_explained.md).

## Verification and reproducibility

**Local verification reported for 10 November 2026:** all **64 tests passed** on Windows with **SUMO 1.26.0** and PyTorch installed. Baseline runs, PPO training, checkpoint loading and GUI execution were also checked. This result refers to the locally tested version; newly pushed files or changes should be retested before distribution. The exact PyTorch version should be confirmed from the local installation before recording it here.

To run the checks on another machine:

```bash
pytest -rs
```

Use independent random seeds for evaluation and compare traffic metrics, not only PPO episode returns. The repository does not claim that PPO always outperforms its baselines.

## Scope, attribution and licence

This is a discrete-action, single-agent starting point. It does not provide multi-agent PPO or continuous-action vehicle control.

The PPO learner was adapted from the `Mobility_PPO` research baseline and reorganised for general episode-based control. SUMO is developed by DLR and contributors and is not distributed with this repository.

**Licence:** [MIT](LICENSE)