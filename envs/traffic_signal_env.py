"""Example A: PPO control of one traffic signal.

The agent decides, every few seconds, whether to keep the current green phase
or to move to the next one. The environment carries out every signal change
safely: it enforces minimum and maximum green times and inserts yellow and
all-red clearance intervals, which the agent cannot skip.

The same class runs the small teaching intersection and a single junction
inside the larger city network (see envs/__init__.py). Full specification:
docs/environments.md.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gymnasium import spaces

from .sumo_base import SCENARIO_DIR, SumoEnvironment

KEEP_PHASE = 0
SWITCH_PHASE = 1

# Queue lengths are normalised by how many cars fit on an approach: a car
# 5 m long plus the 2.5 m gap it keeps when stopped.
VEHICLE_SPACING_M = 7.5
# Longest time to wait, at reset, for SUMO's own program to show a green
# phase before the environment takes over the signal.
MAX_TAKEOVER_WAIT_S = 300.0


@dataclass
class SignalLayout:
    """What the environment needs to know about one signalised junction."""

    green_states: list[str]          # signal state of each green stage, in program order
    approach_lengths: dict[str, float]  # incoming edge -> total length of its vehicle lanes (m)
    free_flow_times: dict[str, float]   # incoming edge -> time to drive it at the speed limit (s)


def read_signal_layout(network_file: Path, tls_id: str) -> SignalLayout:
    """Read a traffic light's green stages and incoming approaches from a .net.xml file.

    Pedestrian crossings are ignored when deciding what counts as a green
    stage, and walking areas and pavements are not counted as approaches.
    """
    root = ET.parse(network_file).getroot()

    program = next((tl for tl in root.iter("tlLogic") if tl.get("id") == tls_id), None)
    if program is None:
        available = sorted({tl.get("id") for tl in root.iter("tlLogic")})
        shown = ", ".join(available[:20]) + (" ..." if len(available) > 20 else "")
        raise ValueError(f"No traffic light '{tls_id}' in {network_file}. Available: {shown}")

    connections = [c for c in root.iter("connection") if c.get("tl") == tls_id]
    # Controlled links that start on a walking area (IDs beginning with ':')
    # are pedestrian crossings.
    crossing_links = {int(c.get("linkIndex")) for c in connections if c.get("from").startswith(":")}

    green_states: list[str] = []
    previous_vehicle_signals = None
    for phase in program.iter("phase"):
        state = phase.get("state")
        vehicle_signals = "".join(s for i, s in enumerate(state) if i not in crossing_links)
        if "y" in vehicle_signals or not any(s in "Gg" for s in vehicle_signals):
            previous_vehicle_signals = None
            continue
        # Programs often split one green into two phases that differ only in
        # the pedestrian signals ("walk", then "don't walk"). For the agent
        # they are one green stage; the first of the two is kept.
        if vehicle_signals != previous_vehicle_signals:
            green_states.append(state)
        previous_vehicle_signals = vehicle_signals
    if len(green_states) < 2:
        raise ValueError(f"Traffic light '{tls_id}' needs at least two green stages; found {green_states}.")

    incoming = {c.get("from") for c in connections if not c.get("from").startswith(":")}
    approach_lengths, free_flow_times = {}, {}
    for edge in root.iter("edge"):
        if edge.get("id") not in incoming or edge.get("function"):
            continue
        lanes = [lane for lane in edge.iter("lane") if lane.get("allow") != "pedestrian"]
        approach_lengths[edge.get("id")] = sum(float(lane.get("length")) for lane in lanes)
        free_flow_times[edge.get("id")] = float(lanes[0].get("length")) / float(lanes[0].get("speed"))
    approach_lengths = dict(sorted(approach_lengths.items()))
    return SignalLayout(green_states, approach_lengths, free_flow_times)


def yellow_state(current: str, following: str) -> str:
    """Signal state used to clear ``current`` before ``following`` starts.

    Movements that are green now but not green in the next phase show yellow;
    every other movement keeps its current signal.
    """
    return "".join(
        "y" if now in "Gg" and nxt not in "Gg" else now
        for now, nxt in zip(current, following)
    )


def all_red_state(current: str, following: str) -> str:
    """Signal state for the all-red clearance between two green phases.

    Every movement is red except those that are green in both phases; those
    have no conflict to clear and keep their green.
    """
    return "".join(
        now if now in "Gg" and nxt in "Gg" else "r"
        for now, nxt in zip(current, following)
    )


class TrafficSignalEnv(SumoEnvironment):
    """Single-junction signal control with actions {keep, switch}.

    Observation (all values in [0, 1]):
        [0 : A]          queued (halted) vehicles on each incoming approach,
                         divided by the approach's storage capacity
        [A : A + P]      one-hot encoding of the current green stage
        [A + P]          time since the current green started / max_green, clipped

    where A is the number of approaches (sorted alphabetically by edge ID)
    and P the number of green stages found in the junction's signal program
    (2 for both provided scenarios: one per traffic direction).

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
        max_green: float | None = 60.0,
        yellow_time: float = 3.0,
        all_red_time: float = 2.0,
        warmup_seconds: float = 0.0,
        sumo_controls_signal: bool = False,
        sumo_options: list[str] | None = None,
    ):
        """
        Args:
            max_green: a green stage is ended automatically after this many
                seconds, so no approach can be starved. None disables it.
            warmup_seconds: simulated time at the start of each episode during
                which SUMO's own program runs the signal and nothing is
                recorded, so that the episode starts with traffic in the network.
            sumo_controls_signal: if True, the agent's actions are ignored and
                the signal keeps running SUMO's own program from the network
                file. Used for the "sumo_program" baseline; observations,
                rewards and metrics are computed as for the agent.
        """
        super().__init__(config_file, gui=gui, sumo_options=sumo_options)
        self.tls_id = tls_id
        self.episode_seconds = episode_seconds
        self.decision_interval = decision_interval
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.all_red_time = all_red_time
        self.warmup_seconds = warmup_seconds
        self.sumo_controls_signal = sumo_controls_signal

        layout = read_signal_layout(self.network_file, tls_id)
        self.green_states = layout.green_states
        self.incoming_edges = list(layout.approach_lengths)
        self.free_flow_times = layout.free_flow_times
        self.approach_capacity = np.array(
            [length / VEHICLE_SPACING_M for length in layout.approach_lengths.values()], dtype=np.float32
        )
        # Elapsed green time is reported relative to the maximum green (or 60 s without one).
        self.elapsed_scale = max_green or 60.0

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

        for _ in range(int(round(self.warmup_seconds / self.step_length))):
            self.sumo.simulationStep()

        self.green_index = self.wait_for_program_green()
        if not self.sumo_controls_signal:
            self.set_signal(self.green_states[self.green_index])
        self.start_time = self.green_start_time = self.simulation_time

        self.queue_lengths = self.read_queue_lengths()
        self.total_queue_seconds = 0.0
        self.phase_switches = 0
        self.teleports = 0
        self.throughput = 0
        self._delay_sum, self._delay_max, self._delay_count = 0.0, 0.0, 0
        # Vehicle -> (time it entered an approach, edge). Vehicles already
        # waiting when the episode starts are marked None: their entry time is
        # unknown, so they count towards throughput but not towards delay.
        self._on_approach: dict[str, tuple[float, str] | None] = {
            vehicle: None for vehicle in self.vehicles_on_approaches()
        }

    def wait_for_program_green(self) -> int:
        """Advance until SUMO's own program shows one of the green stages.

        Taking over only at such a moment means control never begins with an
        abrupt change of the signals shown to drivers.
        """
        for _ in range(self.steps_for(MAX_TAKEOVER_WAIT_S)):
            state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id)
            if state in self.green_states:
                return self.green_states.index(state)
            self.sumo.simulationStep()
        raise RuntimeError(f"Traffic light '{self.tls_id}' showed none of its green stages "
                           f"within {MAX_TAKEOVER_WAIT_S:.0f} s.")

    # ---------------------------------------------------------------- RL interface

    def step(self, action):
        action = self.check_action(action)
        reward = 0.0

        if not self.sumo_controls_signal:
            # PPO only *requests* a change. The request is carried out once the
            # current green has lasted min_green seconds; after max_green
            # seconds the change happens whatever the agent chooses.
            switch_requested = action == SWITCH_PHASE and self.time_in_green >= self.min_green
            max_green_reached = self.max_green is not None and self.time_in_green >= self.max_green
            if switch_requested or max_green_reached:
                reward += self.change_to_next_green()

        reward += self.simulate(self.decision_interval)

        terminated = self.sumo.simulation.getMinExpectedNumber() == 0  # no traffic left or to come
        truncated = not terminated and self.simulation_time - self.start_time >= self.episode_seconds
        return self.get_observation(), reward, terminated, truncated, self.get_info()

    def get_observation(self) -> np.ndarray:
        queues = np.minimum(self.queue_lengths / self.approach_capacity, 1.0)
        phase = np.zeros(len(self.green_states), dtype=np.float32)
        phase[self.green_index] = 1.0
        elapsed = min(self.time_in_green / self.elapsed_scale, 1.0)
        return np.concatenate([queues, phase, [elapsed]]).astype(np.float32)

    def calculate_reward(self) -> float:
        """Reward rate for the current second: minus the total queue length."""
        # Students can replace queue length with another traffic objective,
        # e.g. waiting time, throughput or a pressure term. Keep the returned
        # reward a scalar; step() multiplies it by the step length and sums it
        # over the decision interval.
        return -float(np.sum(self.queue_lengths))

    def get_info(self) -> dict:
        now = self.simulation_time
        elapsed = now - self.start_time
        # Vehicles still on an approach count with the delay they have so far,
        # so that a starved approach shows up in the delay statistics.
        delays = [self.delay(entry, now) for entry in self._on_approach.values() if entry is not None]
        count = self._delay_count + len(delays)
        return {
            "simulation_time": now,
            "green_phase": self.green_index,
            "time_in_green": self.time_in_green,
            "mean_queue_length": self.total_queue_seconds / elapsed if elapsed > 0 else 0.0,
            "mean_delay": (self._delay_sum + sum(delays)) / count if count else 0.0,
            "max_delay": max([self._delay_max, *delays]),
            "throughput": self.throughput,
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
        """Run yellow, then all-red, then start the next green stage.

        Returns the reward accumulated while the transition was simulated.
        """
        current = self.green_states[self.green_index]
        next_index = (self.green_index + 1) % len(self.green_states)
        following = self.green_states[next_index]

        self.set_signal(yellow_state(current, following))
        reward = self.simulate(self.yellow_time)
        if self.all_red_time > 0:
            self.set_signal(all_red_state(current, following))
            reward += self.simulate(self.all_red_time)

        self.green_index = next_index
        self.set_signal(following)
        self.green_start_time = self.simulation_time
        self.phase_switches += 1
        return reward

    def follow_program(self) -> None:
        """When SUMO runs the signal, keep track of which green stage it shows."""
        state = self.sumo.trafficlight.getRedYellowGreenState(self.tls_id)
        if state in self.green_states and self.green_states.index(state) != self.green_index:
            self.green_index = self.green_states.index(state)
            self.green_start_time = self.simulation_time
            self.phase_switches += 1

    # -------------------------------------------------------- simulation stepping

    def simulate(self, seconds: float) -> float:
        """Advance SUMO by ``seconds`` and return the reward accumulated meanwhile."""
        reward = 0.0
        for _ in range(self.steps_for(seconds)):
            self.sumo.simulationStep()
            if self.sumo_controls_signal:
                self.follow_program()
            self.record_traffic()
            reward += self.calculate_reward() * self.step_length
        return reward

    def read_queue_lengths(self) -> np.ndarray:
        # SUMO counts a vehicle as halting when its speed is below 0.1 m/s.
        return np.array(
            [self.sumo.edge.getLastStepHaltingNumber(edge) for edge in self.incoming_edges],
            dtype=np.float32,
        )

    def vehicles_on_approaches(self) -> dict[str, str]:
        """Map each vehicle currently on an incoming approach to that approach."""
        return {
            vehicle: edge
            for edge in self.incoming_edges
            for vehicle in self.sumo.edge.getLastStepVehicleIDs(edge)
        }

    def delay(self, entry: tuple[float, str], until: float) -> float:
        """Time spent on the approach beyond driving it at the speed limit."""
        entered, edge = entry
        return max(0.0, until - entered - self.free_flow_times[edge])

    def record_traffic(self) -> None:
        """Update queue lengths and the episode's traffic metrics after one SUMO step."""
        now = self.simulation_time
        self.queue_lengths = self.read_queue_lengths()
        self.total_queue_seconds += float(np.sum(self.queue_lengths)) * self.step_length
        self.teleports += self.sumo.simulation.getStartingTeleportNumber()

        # A vehicle that has left the approaches has (almost always) crossed
        # the junction: count it and record its delay.
        current = self.vehicles_on_approaches()
        for vehicle in [v for v in self._on_approach if v not in current]:
            entry = self._on_approach.pop(vehicle)
            self.throughput += 1
            if entry is not None:
                delay = self.delay(entry, now)
                self._delay_sum += delay
                self._delay_max = max(self._delay_max, delay)
                self._delay_count += 1
        for vehicle, edge in current.items():
            if vehicle not in self._on_approach:
                self._on_approach[vehicle] = (now, edge)
