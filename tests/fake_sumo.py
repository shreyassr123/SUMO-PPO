"""Scripted stand-ins for TraCI, used to unit-test the environments without SUMO.

The fakes implement only the TraCI calls that the environments make, with
deliberately simple traffic dynamics. They test the environments' own logic
(signal sequencing, action translation, episode endings, bookkeeping), not
SUMO's behaviour; tests/test_sumo.py covers the real simulator.
"""

from __future__ import annotations

from types import SimpleNamespace


class FakeTraciModule:
    """Replaces the ``traci`` package: start(), getConnection() and constants."""

    def __init__(self, connection_class, **connection_kwargs):
        self.constants = SimpleNamespace(VAR_ACCUMULATED_WAITING_TIME=0x87)
        self.connection_class = connection_class
        self.connection_kwargs = connection_kwargs
        self.open_connections: dict[str, object] = {}
        self.commands: list[list[str]] = []

    def start(self, command, label):
        if label in self.open_connections:
            raise RuntimeError(f"Connection '{label}' is already active.")
        self.commands.append(command)
        self.open_connections[label] = self.connection_class(self, label, **self.connection_kwargs)

    def getConnection(self, label):
        return self.open_connections[label]


class UnknownObject(KeyError):
    """Raised like TraCI's TraCIException when an ID does not exist."""


# --------------------------------------------------------------------- signal

# Link indices of each approach in tests/test_environments.py's network.
APPROACH_LINKS = {"north_in": [0, 1, 2], "east_in": [3, 4, 5], "south_in": [6, 7, 8], "west_in": [9, 10, 11]}


class FakeIntersection:
    """One vehicle joins each approach's queue every second second; an approach
    with at least one green link discharges one queued vehicle per second.
    A vehicle departs every 2 s and arrives 10 s later with 5 s of waiting."""

    def __init__(self, module, label, n_links=12, expected_vehicles=1000):
        self.module, self.label = module, label
        self.time = 0.0
        self.signal = "r" * n_links
        self.queues = {edge: 0 for edge in APPROACH_LINKS}
        self.expected_vehicles = expected_vehicles
        self.signal_log: list[str] = []      # state in force during each simulated second
        self.queue_log: list[int] = []       # total queue after each simulated second
        self.subscribed: set[str] = set()
        self.closed = False

        self.simulation = SimpleNamespace(
            getTime=lambda: self.time,
            getDeltaT=lambda: 1.0,
            getMinExpectedNumber=lambda: self.expected_vehicles,
            getStartingTeleportNumber=lambda: 0,
            getDepartedIDList=self._departed,
            getArrivedIDList=self._arrived,
        )
        self.trafficlight = SimpleNamespace(
            getIDList=lambda: ["center"],
            getRedYellowGreenState=lambda tls: self.signal,
            setRedYellowGreenState=self._set_signal,
        )
        self.edge = SimpleNamespace(getLastStepHaltingNumber=lambda edge: self.queues[edge])
        self.vehicle = SimpleNamespace(
            subscribe=lambda vehicle, variables: self.subscribed.add(vehicle),
            getAllSubscriptionResults=lambda: {v: {0x87: 5.0} for v in self.subscribed},
        )

    def _set_signal(self, tls, state):
        assert len(state) == len(self.signal), "signal state has the wrong length"
        self.signal = state

    def _departed(self):
        if int(self.time) % 2 == 0:
            vehicle = f"veh{int(self.time)}"
            return [vehicle]
        return []

    def _arrived(self):
        vehicle = f"veh{int(self.time) - 10}"
        if vehicle in self.subscribed:
            self.subscribed.discard(vehicle)
            return [vehicle]
        return []

    def simulationStep(self):
        self.signal_log.append(self.signal)
        self.time += 1.0
        for edge, links in APPROACH_LINKS.items():
            if any(self.signal[i] in "Gg" for i in links):
                self.queues[edge] = max(0, self.queues[edge] - 1)
            if int(self.time) % 2 == 0:
                self.queues[edge] += 1
        self.queue_log.append(sum(self.queues.values()))

    def close(self):
        self.closed = True
        del self.module.open_connections[self.label]


# -------------------------------------------------------------------- vehicle

class FakeRoad:
    """A 300 m road (edges r1, r2) with a 10 m/s limit. The ego vehicle enters
    at t = 3 s with speed 5 m/s; an optional leader starts `leader_gap` metres
    ahead and drives at `leader_speed`. The ego follows its commanded speed
    within +2.6/-4.5 m/s^2 and never closer than a 2 m gap to the leader."""

    ROAD_LENGTH = 300.0
    ALLOWED_SPEED = 10.0

    def __init__(self, module, label, leader_gap=None, leader_speed=8.0, collide_at=None):
        self.module, self.label = module, label
        self.time = 0.0
        self.ego_present = self.ego_arrived = False
        self.ego_position, self.ego_speed, self.ego_accel = 0.0, 5.0, 0.0
        self.commanded_speed = None
        self.speed_commands: list[float] = []
        self.speed_mode = None
        self.leader_position = None if leader_gap is None else leader_gap + 5.0  # + leader length
        self.leader_speed = leader_speed
        self.collide_at = collide_at
        self.arrived_now: list[str] = []
        self.closed = False

        self.simulation = SimpleNamespace(
            getTime=lambda: self.time,
            getDeltaT=lambda: 1.0,
            getArrivedIDList=lambda: list(self.arrived_now),
            getCollidingVehiclesIDList=lambda: ["ego"] if self.collide_at == self.time else [],
        )
        self.vehicle = SimpleNamespace(
            getIDList=self._ids,
            getSpeed=lambda v: self._speed(v),
            getAllowedSpeed=lambda v: self._ego(v) and self.ALLOWED_SPEED,
            getAcceleration=lambda v: self._ego(v) and self.ego_accel,
            getLeader=self._leader,
            getDrivingDistance=lambda v, edge, pos: self._ego(v) and self.ROAD_LENGTH - self.ego_position,
            getRoute=lambda v: self._ego(v) and ("r1", "r2"),
            setSpeed=self._set_speed,
            setSpeedMode=lambda v, mode: setattr(self, "speed_mode", mode),
        )
        self.lane = SimpleNamespace(getLength=lambda lane: 150.0)
        self.gui = SimpleNamespace(trackVehicle=lambda view, v: None)

    def _ids(self):
        ids = ["leader"] if self.leader_position is not None else []
        return ids + (["ego"] if self.ego_present else [])

    def _ego(self, vehicle):
        if vehicle != "ego" or not self.ego_present:
            raise UnknownObject(f"Vehicle '{vehicle}' is not known.")
        return True

    def _speed(self, vehicle):
        if vehicle == "leader" and self.leader_position is not None:
            return self.leader_speed
        self._ego(vehicle)
        return self.ego_speed

    def _gap(self):
        return self.leader_position - 5.0 - self.ego_position

    def _leader(self, vehicle, distance):
        self._ego(vehicle)
        if self.leader_position is None or self._gap() > distance:
            return None
        return ("leader", self._gap())

    def _set_speed(self, vehicle, speed):
        self._ego(vehicle)
        self.commanded_speed = speed
        self.speed_commands.append(speed)

    def simulationStep(self):
        self.time += 1.0
        self.arrived_now = []
        if self.leader_position is not None:
            self.leader_position += self.leader_speed
        if not self.ego_present:
            if self.time >= 3.0 and not self.ego_arrived:
                self.ego_present = True
            return
        desired = self.ALLOWED_SPEED if self.commanded_speed is None else self.commanded_speed
        new_speed = min(max(desired, self.ego_speed - 4.5), self.ego_speed + 2.6)
        if self.leader_position is not None:
            new_speed = min(new_speed, max(self._gap() - 2.0, 0.0) + self.leader_speed)
        new_speed = max(new_speed, 0.0)
        self.ego_accel = new_speed - self.ego_speed
        self.ego_speed = new_speed
        self.ego_position += new_speed
        if self.ego_position >= self.ROAD_LENGTH:
            self.ego_present, self.ego_arrived = False, True
            self.arrived_now = ["ego"]

    def close(self):
        self.closed = True
        del self.module.open_connections[self.label]
