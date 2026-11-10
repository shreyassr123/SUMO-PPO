"""Example B: PPO speed control of a single vehicle.

One vehicle (ID ``ego`` in the route file) is driven by the agent, which
chooses every simulation step to decelerate, hold its speed or accelerate.
All other vehicles follow SUMO's default car-following model.

The agent sets a *desired* speed. SUMO's safety checks stay active (speed
mode 31), so SUMO still lowers the speed whenever the desired speed would
cause a rear-end collision. See the README section "Vehicle-control example".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gymnasium import spaces

from .sumo_base import SCENARIO_DIR, SumoEnvironment

DECELERATE, MAINTAIN, ACCELERATE = 0, 1, 2
# Speed change per second requested by each action (m/s^2). These are
# comfortable values, well inside the vehicle's limits (accel 2.6, decel 4.5).
ACTION_ACCELERATIONS = {DECELERATE: -2.0, MAINTAIN: 0.0, ACCELERATE: 1.0}

# SUMO speed mode 31 = all five safety checks on (safe following speed,
# maximum acceleration, maximum deceleration, right of way, braking at red
# lights). It is SUMO's default; we set it explicitly to make that visible.
SAFE_SPEED_MODE = 31

LEADER_VIEW_DISTANCE_M = 100.0   # leaders further away are treated as absent
DESIRED_HEADWAY_S = 1.5          # time gaps shorter than this are penalised
COMFORT_ACCELERATION = 2.0       # m/s^2; |acceleration| at which the comfort penalty is 1

# Reward weights (see calculate_reward).
DELAY_WEIGHT = 1.0
SAFETY_WEIGHT = 1.0
COMFORT_WEIGHT = 0.1
COLLISION_PENALTY = 10.0


class VehicleControlEnv(SumoEnvironment):
    """Longitudinal control of one vehicle on a single-lane road.

    Observation (5 values):
        0  speed / allowed speed                                  [0, 1]
        1  gap to the leader / 100 m (1 if no leader in view)      [0, 1]
        2  (leader speed - own speed) / allowed speed (0 if none)  [-1, 1]
        3  1 if a leader is within 100 m, else 0                   {0, 1}
        4  remaining route distance / route length                 [0, 1]

    Actions: 0 decelerate (-2 m/s^2), 1 maintain speed, 2 accelerate (+1 m/s^2).

    The episode terminates when the vehicle reaches the end of its route (or
    collides, or leaves the simulation for another reason) and is truncated
    after ``max_episode_seconds`` of driving.
    """

    def __init__(
        self,
        config_file: str | Path = SCENARIO_DIR / "vehicle_control" / "simulation.sumocfg",
        gui: bool = False,
        vehicle_id: str = "ego",
        max_episode_seconds: float = 300.0,
        max_departure_wait: float = 600.0,
        sumo_drives_vehicle: bool = False,
        sumo_options: list[str] | None = None,
    ):
        """
        Args:
            sumo_drives_vehicle: if True the agent's actions are ignored and
                SUMO's own driver model controls the vehicle. Used for the
                "SUMO driver" baseline; observations, rewards and metrics are
                computed exactly as for the agent.
        """
        super().__init__(config_file, gui=gui, sumo_options=sumo_options)
        self.vehicle_id = vehicle_id
        self.max_episode_seconds = max_episode_seconds
        self.max_departure_wait = max_departure_wait
        self.sumo_drives_vehicle = sumo_drives_vehicle

        low = np.array([0.0, 0.0, -1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(len(ACTION_ACCELERATIONS))

    # ------------------------------------------------------------- episode set-up

    def on_reset(self) -> None:
        # Background traffic starts first; wait until our vehicle is inserted.
        start = self.simulation_time
        while self.vehicle_id not in self.sumo.vehicle.getIDList():
            if self.simulation_time - start > self.max_departure_wait:
                raise RuntimeError(
                    f"Vehicle '{self.vehicle_id}' did not enter the simulation within "
                    f"{self.max_departure_wait:.0f} s. Check that the route file defines it."
                )
            self.sumo.simulationStep()

        if not self.sumo_drives_vehicle:
            self.sumo.vehicle.setSpeedMode(self.vehicle_id, SAFE_SPEED_MODE)
        if self.gui:
            self.sumo.gui.trackVehicle("View #0", self.vehicle_id)

        route = self.sumo.vehicle.getRoute(self.vehicle_id)
        self.final_edge = route[-1]
        self.final_edge_length = self.sumo.lane.getLength(f"{self.final_edge}_0")

        self.departure_time = self.simulation_time
        self.arrived = self.collided = False
        self.state = self.measure_vehicle()
        self.route_length = max(self.state["remaining_distance"], 1.0)

        self._speed_sum = 0.0
        self._abs_acceleration_sum = 0.0
        self._steps = 0
        self.min_time_headway = float("inf")
        self.short_headway_steps = 0

    # ---------------------------------------------------------------- RL interface

    def step(self, action):
        action = self.check_action(action)
        if not self.sumo_drives_vehicle:
            self.apply_action(action)

        self.sumo.simulationStep()

        # Check *before* any further TraCI call about the vehicle: it may have
        # left the simulation during this step.
        self.arrived = self.vehicle_id in self.sumo.simulation.getArrivedIDList()
        self.collided = self.vehicle_id in self.sumo.simulation.getCollidingVehiclesIDList()
        still_present = self.vehicle_id in self.sumo.vehicle.getIDList()

        if still_present:
            self.state = self.measure_vehicle()
            self.record_metrics()
            reward = self.calculate_reward()
        else:
            # The vehicle has left (normally by arriving); nothing to measure.
            self.state = dict(self.state, remaining_distance=0.0)
            reward = 0.0
        if self.collided:
            reward -= COLLISION_PENALTY

        terminated = self.arrived or self.collided or not still_present
        truncated = not terminated and self.simulation_time - self.departure_time >= self.max_episode_seconds
        return self.get_observation(), reward, terminated, truncated, self.get_info()

    def apply_action(self, action: int) -> None:
        """Translate a discrete action into a desired speed for the next step."""
        speed = self.state["speed"]
        target = speed + ACTION_ACCELERATIONS[action] * self.step_length
        # Never ask for more than the speed limit; SUMO may still choose a
        # lower speed if the target is unsafe.
        target = float(np.clip(target, 0.0, self.state["allowed_speed"]))
        self.sumo.vehicle.setSpeed(self.vehicle_id, target)

    def measure_vehicle(self) -> dict[str, float]:
        """Query SUMO for everything the observation and reward need."""
        vehicle = self.sumo.vehicle
        state = {
            "speed": vehicle.getSpeed(self.vehicle_id),
            "allowed_speed": vehicle.getAllowedSpeed(self.vehicle_id),
            "acceleration": vehicle.getAcceleration(self.vehicle_id),
            "leader_present": False,
            "gap": LEADER_VIEW_DISTANCE_M,
            "leader_speed": 0.0,
        }
        # getLeader returns None (or an empty ID in some SUMO versions) when
        # there is no vehicle ahead within the given distance.
        leader = vehicle.getLeader(self.vehicle_id, LEADER_VIEW_DISTANCE_M)
        if leader and leader[0]:
            state["leader_present"] = True
            state["gap"] = max(float(leader[1]), 0.0)
            state["leader_speed"] = vehicle.getSpeed(leader[0])
        distance = vehicle.getDrivingDistance(self.vehicle_id, self.final_edge, self.final_edge_length)
        state["remaining_distance"] = max(float(distance), 0.0)  # SUMO returns a large negative value if unreachable
        return state

    def get_observation(self) -> np.ndarray:
        s = self.state
        allowed = max(s["allowed_speed"], 0.1)
        relative_speed = (s["leader_speed"] - s["speed"]) / allowed if s["leader_present"] else 0.0
        observation = np.array([
            s["speed"] / allowed,
            s["gap"] / LEADER_VIEW_DISTANCE_M,
            relative_speed,
            1.0 if s["leader_present"] else 0.0,
            s["remaining_distance"] / self.route_length,
        ], dtype=np.float32)
        return np.clip(observation, self.observation_space.low, self.observation_space.high)

    def calculate_reward(self) -> float:
        """Reward for one step:  -w_d * delay - w_s * safety - w_c * comfort.

        delay   = 1 - speed / allowed speed           (0 when driving at the limit)
        safety  = max(0, 1 - headway / 1.5 s)         (0 when the time gap is >= 1.5 s)
        comfort = (acceleration / 2 m/s^2)^2
        """
        # Students can add or reweight terms here. Note that progress is
        # rewarded as "less delay": a positive reward per step would make
        # arriving early *end* the stream of rewards and teach the agent to
        # dawdle (see docs/extending_the_framework.md).
        s = self.state
        delay = 1.0 - min(s["speed"] / max(s["allowed_speed"], 0.1), 1.0)
        safety = max(0.0, 1.0 - self.time_headway() / DESIRED_HEADWAY_S)
        comfort = (s["acceleration"] / COMFORT_ACCELERATION) ** 2
        return -(DELAY_WEIGHT * delay + SAFETY_WEIGHT * safety + COMFORT_WEIGHT * comfort)

    def time_headway(self) -> float:
        """Seconds until the vehicle would reach the leader's current position."""
        if not self.state["leader_present"]:
            return float("inf")
        return self.state["gap"] / max(self.state["speed"], 0.1)

    # --------------------------------------------------------------------- metrics

    def record_metrics(self) -> None:
        self._steps += 1
        self._speed_sum += self.state["speed"]
        self._abs_acceleration_sum += abs(self.state["acceleration"])
        headway = self.time_headway()
        self.min_time_headway = min(self.min_time_headway, headway)
        if headway < DESIRED_HEADWAY_S:
            self.short_headway_steps += 1

    def get_info(self) -> dict:
        steps = max(self._steps, 1)
        return {
            "travel_time": self.simulation_time - self.departure_time,
            "mean_speed": self._speed_sum / steps,
            "mean_abs_acceleration": self._abs_acceleration_sum / steps,
            # NaN when no leader was ever within view (written as "nan" in CSV files).
            "min_time_headway": self.min_time_headway if np.isfinite(self.min_time_headway) else float("nan"),
            "short_headway_steps": self.short_headway_steps,
            "arrived": self.arrived,
            "collided": self.collided,
        }
