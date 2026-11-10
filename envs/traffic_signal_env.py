"""Example A: PPO control of one traffic signal.

The agent decides, every few seconds, whether to keep the current green phase
or to move to the next one. The environment carries out every signal change
safely: it enforces a minimum green time and inserts yellow and all-red
clearance intervals, which the agent cannot skip.

See the README section "Traffic-signal example" for the full specification.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .sumo_base import SCENARIO_DIR, SumoEnvironment, import_traci

KEEP_PHASE = 0
SWITCH_PHASE = 1

# Queue lengths are normalised by how many cars fit on an approach: the
# scenario's cars are 5 m long and keep a 2.5 m gap when stopped.
VEHICLE_SPACING_M = 7.5
# Elapsed green time is divided by this value and clipped to 1.
ELAPSED_GREEN_SCALE_S = 60.0


def read_signal_layout(network_file: Path, tls_id: str) -> tuple[list[str], dict[str, float]]:
    """Read a traffic light's green phases and incoming approaches from a .net.xml file.

    Returns:
        green_states: SUMO signal-state strings (one character per controlled
            link) of the green phases in the light's default program, in order.
        approach_lengths: total lane length in metres of each incoming edge,
            keyed by edge ID and sorted alphabetically.
    """
    root = ET.parse(network_file).getroot()

    program = next((tl for tl in root.iter("tlLogic") if tl.get("id") == tls_id), None)
    if program is None:
        available = sorted({tl.get("id") for tl in root.iter("tlLogic")})
        raise ValueError(f"No traffic light '{tls_id}' in {network_file}. Available: {available}")

    # A green phase gives right of way to some movements (G or g) and contains
    # no yellow; yellow phases from the file are rebuilt by the environment.
    green_states = [
        phase.get("state")
        for phase in program.iter("phase")
        if any(c in "Gg" for c in phase.get("state")) and "y" not in phase.get("state")
    ]
    if len(green_states) < 2:
        raise ValueError(f"Traffic light '{tls_id}' needs at least two green phases; found {green_states}.")

    incoming = {c.get("from") for c in root.iter("connection") if c.get("tl") == tls_id}
    approach_lengths = {}
    for edge in root.iter("edge"):
        if edge.get("id") in incoming:
            approach_lengths[edge.get("id")] = sum(float(lane.get("length")) for lane in edge.iter("lane"))
    return green_states, dict(sorted(approach_lengths.items()))


def yellow_state(current: str, following: str) -> str:
    """Signal state used to clear ``current`` before ``following`` starts.

    Movements that are green now but not green in the next phase show yellow;
    every other movement keeps its current signal.
    """
    return "".join(
        "y" if now in "Gg" and nxt not in "Gg" else now
        for now, nxt in zip(current, following)
    )


class TrafficSignalEnv(SumoEnvironment):
    """Single-intersection signal control with actions {keep, switch}.

    Observation (all values in [0, 1]):
        [0 : A]          queued (halted) vehicles on each incoming approach,
                         divided by the approach's storage capacity
        [A : A + P]      one-hot encoding of the current green phase
        [A + P]          time since the current green started / 60 s, clipped

    where A is the number of approaches (4 here, ordered alphabetically by edge
    ID: east_in, north_in, south_in, west_in) and P the number of green phases
    found in the network's signal program (2 expected for the provided
    network: north-south and east-west).

    Reward for one decision step: minus the number of halted vehicles on all
    approaches, summed over every simulated second of the step, i.e. minus
    the vehicle-seconds spent queueing.
    """

    def __init__(
        self,
        config_file: str | Path = SCENARIO_DIR / "simple_intersection" / "simulation.sumocfg",
        gui: bool = False,
        tls_id: str = "center",
        episode_seconds: float = 3600.0,
        decision_interval: float = 5.0,
        min_green: float = 10.0,
        yellow_time: float = 3.0,
        all_red_time: float = 2.0,
        sumo_options: list[str] | None = None,
    ):
        super().__init__(config_file, gui=gui, sumo_options=sumo_options)
        self.tls_id = tls_id
        self.episode_seconds = episode_seconds
        self.decision_interval = decision_interval
        self.min_green = min_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time

        self.green_states, approach_lengths = read_signal_layout(self.network_file, tls_id)
        self.incoming_edges = list(approach_lengths)
        self.approach_capacity = np.array(
            [length / VEHICLE_SPACING_M for length in approach_lengths.values()], dtype=np.float32
        )

        n_values = len(self.incoming_edges) + len(self.green_states) + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n_values,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    # ------------------------------------------------------------- episode set-up

    def on_reset(self) -> None:
        if self.tls_id not in self.sumo.trafficlight.getIDList():
            raise ValueError(f"Traffic light '{self.tls_id}' is not in the running simulation.")
        n_links = len(self.sumo.trafficlight.getRedYellowGreenState(self.tls_id))
        if n_links != len(self.green_states[0]):
            raise ValueError(
                f"Traffic light '{self.tls_id}' controls {n_links} links in SUMO but the "
                f"phases read from {self.network_file.name} have {len(self.green_states[0])}."
            )
        self._waiting_time_var = import_traci().constants.VAR_ACCUMULATED_WAITING_TIME

        self.start_time = self.simulation_time
        self.green_index = 0
        self.set_signal(self.green_states[0])
        self.green_start_time = self.simulation_time

        self.queue_lengths = self.read_queue_lengths()
        self.total_queue_seconds = 0.0
        self.phase_switches = 0
        self.teleports = 0
        self._waiting_so_far: dict[str, float] = {}
        self.completed_waiting_times: list[float] = []

    # ---------------------------------------------------------------- RL interface

    def step(self, action):
        action = self.check_action(action)
        reward = 0.0

        # PPO only *requests* a change. The request is ignored until the
        # current green has lasted at least min_green seconds.
        if action == SWITCH_PHASE and self.time_in_green >= self.min_green:
            reward += self.change_to_next_green()

        reward += self.simulate(self.decision_interval)

        terminated = self.sumo.simulation.getMinExpectedNumber() == 0  # no traffic left or to come
        truncated = not terminated and self.simulation_time - self.start_time >= self.episode_seconds
        return self.get_observation(), reward, terminated, truncated, self.get_info()

    def get_observation(self) -> np.ndarray:
        queues = np.minimum(self.queue_lengths / self.approach_capacity, 1.0)
        phase = np.zeros(len(self.green_states), dtype=np.float32)
        phase[self.green_index] = 1.0
        elapsed = min(self.time_in_green / ELAPSED_GREEN_SCALE_S, 1.0)
        return np.concatenate([queues, phase, [elapsed]]).astype(np.float32)

    def calculate_reward(self) -> float:
        """Reward rate for the current second: minus the total queue length."""
        # Students can replace queue length with another traffic objective,
        # e.g. waiting time, throughput or a pressure term. Keep the returned
        # reward a scalar; step() multiplies it by the step length and sums it
        # over the decision interval.
        return -float(np.sum(self.queue_lengths))

    def get_info(self) -> dict:
        elapsed = self.simulation_time - self.start_time
        waits = self.completed_waiting_times
        return {
            "simulation_time": self.simulation_time,
            "green_phase": self.green_index,
            "time_in_green": self.time_in_green,
            "mean_queue_length": self.total_queue_seconds / elapsed if elapsed > 0 else 0.0,
            "mean_waiting_time": float(np.mean(waits)) if waits else 0.0,
            "throughput": len(waits),
            "phase_switches": self.phase_switches,
            "teleports": self.teleports,
        }

    # ------------------------------------------------------------ signal control

    @property
    def time_in_green(self) -> float:
        return self.simulation_time - self.green_start_time

    def set_signal(self, state: str) -> None:
        # Setting the state directly replaces SUMO's own signal program; the
        # state then stays in force until the environment changes it again.
        self.sumo.trafficlight.setRedYellowGreenState(self.tls_id, state)

    def change_to_next_green(self) -> float:
        """Run yellow, then all-red, then start the next green phase.

        Returns the reward accumulated while the transition was simulated.
        """
        current = self.green_states[self.green_index]
        next_index = (self.green_index + 1) % len(self.green_states)
        following = self.green_states[next_index]

        self.set_signal(yellow_state(current, following))
        reward = self.simulate(self.yellow_time)
        if self.all_red_time > 0:
            self.set_signal("r" * len(current))
            reward += self.simulate(self.all_red_time)

        self.green_index = next_index
        self.set_signal(following)
        self.green_start_time = self.simulation_time
        self.phase_switches += 1
        return reward

    # -------------------------------------------------------- simulation stepping

    def simulate(self, seconds: float) -> float:
        """Advance SUMO by ``seconds`` and return the reward accumulated meanwhile."""
        reward = 0.0
        for _ in range(self.steps_for(seconds)):
            self.sumo.simulationStep()
            self.record_traffic()
            reward += self.calculate_reward() * self.step_length
        return reward

    def read_queue_lengths(self) -> np.ndarray:
        # SUMO counts a vehicle as halting when its speed is below 0.1 m/s.
        return np.array(
            [self.sumo.edge.getLastStepHaltingNumber(edge) for edge in self.incoming_edges],
            dtype=np.float32,
        )

    def record_traffic(self) -> None:
        """Update queue lengths and episode metrics after one SUMO step."""
        self.queue_lengths = self.read_queue_lengths()
        self.total_queue_seconds += float(np.sum(self.queue_lengths)) * self.step_length
        self.teleports += self.sumo.simulation.getStartingTeleportNumber()

        # Waiting time per vehicle is an evaluation metric only. Subscriptions
        # ask SUMO to send every vehicle's accumulated waiting time in a single
        # message per step instead of one request per vehicle.
        for vehicle in self.sumo.simulation.getDepartedIDList():
            self.sumo.vehicle.subscribe(vehicle, [self._waiting_time_var])
        for vehicle, values in self.sumo.vehicle.getAllSubscriptionResults().items():
            self._waiting_so_far[vehicle] = values[self._waiting_time_var]
        for vehicle in self.sumo.simulation.getArrivedIDList():
            self.completed_waiting_times.append(self._waiting_so_far.pop(vehicle, 0.0))
