"""Environment logic tested against the scripted TraCI stand-ins in fake_sumo.py."""

import sys
from itertools import groupby

import numpy as np
import pytest

pytest.importorskip("gymnasium")

import envs.sumo_base as sumo_base  # noqa: E402
from envs.traffic_signal_env import (  # noqa: E402
    KEEP_PHASE, SWITCH_PHASE, TrafficSignalEnv, read_signal_layout, yellow_state)
from envs.vehicle_control_env import (  # noqa: E402
    ACCELERATE, COLLISION_PENALTY, DECELERATE, MAINTAIN, SAFE_SPEED_MODE, VehicleControlEnv)
from fake_sumo import FakeIntersection, FakeRoad, FakeTraciModule  # noqa: E402

# A trimmed .net.xml with the elements the signal environment reads. Links
# 0-2 belong to north_in, 3-5 to east_in, 6-8 to south_in and 9-11 to west_in.
NETWORK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20">
    <edge id=":center_0" function="internal"><lane id=":center_0_0" index="0" length="9.0"/></edge>
    <edge id="north_in" from="north" to="center"><lane id="north_in_0" index="0" length="190.0"/></edge>
    <edge id="east_in" from="east" to="center"><lane id="east_in_0" index="0" length="190.0"/></edge>
    <edge id="south_in" from="south" to="center"><lane id="south_in_0" index="0" length="190.0"/></edge>
    <edge id="west_in" from="west" to="center">
        <lane id="west_in_0" index="0" length="100.0"/><lane id="west_in_1" index="1" length="100.0"/>
    </edge>
    <edge id="north_out" from="center" to="north"><lane id="north_out_0" index="0" length="190.0"/></edge>
    <tlLogic id="center" type="static" programID="0" offset="0">
        <phase duration="42" state="GGgrrrGGgrrr"/>
        <phase duration="3"  state="yyyrrryyyrrr"/>
        <phase duration="42" state="rrrGGgrrrGGg"/>
        <phase duration="3"  state="rrryyyrrryyy"/>
    </tlLogic>
    <connection from="north_in" to="north_out" fromLane="0" toLane="0" tl="center" linkIndex="0"/>
    <connection from="east_in" to="north_out" fromLane="0" toLane="0" tl="center" linkIndex="3"/>
    <connection from="south_in" to="north_out" fromLane="0" toLane="0" tl="center" linkIndex="6"/>
    <connection from="west_in" to="north_out" fromLane="0" toLane="0" tl="center" linkIndex="9"/>
    <connection from=":center_0" to="north_out" fromLane="0" toLane="0"/>
</net>
"""

CONFIG_XML = """<configuration><input><net-file value="network.net.xml"/>
<route-files value="traffic.rou.xml"/></input></configuration>"""

GREEN_NS, GREEN_EW = "GGgrrrGGgrrr", "rrrGGgrrrGGg"


def write_scenario(folder, network=NETWORK_XML):
    (folder / "network.net.xml").write_text(network)
    (folder / "simulation.sumocfg").write_text(CONFIG_XML)
    return folder / "simulation.sumocfg"


@pytest.fixture
def fake_traci(monkeypatch):
    """Install a fake `traci` module; tests choose the connection class."""
    def install(connection_class, **kwargs):
        module = FakeTraciModule(connection_class, **kwargs)
        monkeypatch.setitem(sys.modules, "traci", module)
        monkeypatch.setattr(sumo_base, "find_sumo_tool", lambda name: name)
        return module
    return install


# ------------------------------------------------------------ traffic signal


def test_read_signal_layout(tmp_path):
    write_scenario(tmp_path)
    greens, lengths = read_signal_layout(tmp_path / "network.net.xml", "center")
    assert greens == [GREEN_NS, GREEN_EW]
    assert list(lengths) == ["east_in", "north_in", "south_in", "west_in"]
    assert lengths["west_in"] == 200.0       # both lanes counted
    with pytest.raises(ValueError, match="Available"):
        read_signal_layout(tmp_path / "network.net.xml", "missing")


def test_yellow_state_only_changes_ending_greens():
    assert yellow_state(GREEN_NS, GREEN_EW) == "yyyrrryyyrrr"
    assert yellow_state("GrG", "GGr") == "Gry"


def make_signal_env(tmp_path, **kwargs):
    return TrafficSignalEnv(config_file=write_scenario(tmp_path), **kwargs)


def test_signal_spaces_and_reset(tmp_path, fake_traci):
    module = fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path)
    assert env.observation_space.shape == (4 + 2 + 1,)
    assert env.action_space.n == 2

    observation, info = env.reset(seed=3)
    assert env.observation_space.contains(observation)
    assert observation[4:6].tolist() == [1.0, 0.0]              # first green phase
    assert info["green_phase"] == 0 and info["time_in_green"] == 0.0
    assert "--seed" in module.commands[0]

    with pytest.raises(ValueError):
        env.step(2)
    env.close()
    assert module.open_connections == {}


def test_switch_requests_respect_min_green_yellow_and_all_red(tmp_path, fake_traci):
    fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path, min_green=10, yellow_time=3, all_red_time=2, decision_interval=5)
    env.reset(seed=0)
    connection = env.sumo

    # A switch requested after 0 s of green is ignored.
    env.step(SWITCH_PHASE)
    assert set(connection.signal_log) == {GREEN_NS}

    for _ in range(12):
        env.step(SWITCH_PHASE)

    runs = [(state, len(list(group))) for state, group in groupby(connection.signal_log)]
    expected_cycle = [(GREEN_NS, 10), ("yyyrrryyyrrr", 3), ("r" * 12, 2),
                      (GREEN_EW, 10), ("rrryyyrrryyy", 3), ("r" * 12, 2)]
    assert runs[:6] == expected_cycle
    assert runs[6:12] == expected_cycle

    # Safety: no movement ever goes straight from green to red.
    for before, after in zip(connection.signal_log, connection.signal_log[1:]):
        for b, a in zip(before, after):
            assert not (b in "Gg" and a == "r"), (before, after)
    env.close()


def test_signal_reward_is_minus_queued_vehicle_seconds(tmp_path, fake_traci):
    fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path)
    env.reset(seed=0)
    connection = env.sumo

    for action in [KEEP_PHASE, KEEP_PHASE, SWITCH_PHASE, KEEP_PHASE]:
        seconds_before = len(connection.queue_log)
        _, reward, _, _, info = env.step(action)
        assert reward == pytest.approx(-sum(connection.queue_log[seconds_before:]))

    elapsed = len(connection.queue_log)
    assert info["mean_queue_length"] == pytest.approx(sum(connection.queue_log) / elapsed)
    assert info["mean_waiting_time"] == pytest.approx(5.0)
    assert info["throughput"] > 0
    env.close()


def test_signal_episode_is_truncated_at_time_limit(tmp_path, fake_traci):
    fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path, episode_seconds=60)
    observation, _ = env.reset(seed=1)
    rng = np.random.default_rng(0)
    steps, done = 0, False
    while not done:
        observation, reward, terminated, truncated, info = env.step(int(rng.integers(2)))
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        done = terminated or truncated
        steps += 1
    assert truncated and not terminated
    assert 60 <= info["simulation_time"] < 70
    env.close()


def test_signal_episode_terminates_when_traffic_is_finished(tmp_path, fake_traci):
    fake_traci(FakeIntersection, expected_vehicles=0)
    env = make_signal_env(tmp_path)
    env.reset(seed=0)
    _, _, terminated, truncated, _ = env.step(KEEP_PHASE)
    assert terminated and not truncated
    env.close()


def test_repeated_resets_close_the_previous_simulation(tmp_path, fake_traci):
    module = fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path)
    connections = []
    for seed in range(3):
        env.reset(seed=seed)
        connections.append(env.sumo)
        assert len(module.open_connections) == 1
    assert all(c.closed for c in connections[:-1])
    env.close()
    assert module.open_connections == {}


def test_seeded_resets_give_reproducible_sumo_seeds(tmp_path, fake_traci):
    module = fake_traci(FakeIntersection)
    env = make_signal_env(tmp_path)
    for _ in range(2):
        env.reset(seed=42)
        env.reset()
    env.close()
    seeds = [cmd[cmd.index("--seed") + 1] for cmd in module.commands]
    assert seeds[0] == seeds[2] and seeds[1] == seeds[3] and seeds[0] != seeds[1]


# ----------------------------------------------------------- vehicle control


def make_vehicle_env(tmp_path, **kwargs):
    return VehicleControlEnv(config_file=write_scenario(tmp_path, network="<net/>"), **kwargs)


def test_vehicle_reset_waits_for_departure_and_handles_missing_leader(tmp_path, fake_traci):
    fake_traci(FakeRoad)
    env = make_vehicle_env(tmp_path)
    observation, info = env.reset(seed=0)
    assert env.simulation_time == 3.0                    # ego entered at t = 3
    assert env.sumo.speed_mode == SAFE_SPEED_MODE
    assert env.observation_space.contains(observation)
    np.testing.assert_allclose(observation, [0.5, 1.0, 0.0, 0.0, 1.0])
    env.close()


def test_vehicle_leader_observation(tmp_path, fake_traci):
    fake_traci(FakeRoad, leader_gap=40.0, leader_speed=8.0)
    env = make_vehicle_env(tmp_path)
    observation, _ = env.reset(seed=0)
    # After 3 s the leader has moved 24 m: gap 64 m, relative speed (8 - 5) / 10.
    np.testing.assert_allclose(observation[1:4], [0.64, 0.3, 1.0], rtol=1e-6)
    env.close()


def test_vehicle_actions_set_bounded_speeds(tmp_path, fake_traci):
    fake_traci(FakeRoad)
    env = make_vehicle_env(tmp_path)
    env.reset(seed=0)
    road = env.sumo

    env.step(ACCELERATE)
    assert road.speed_commands[-1] == pytest.approx(6.0)     # 5 + 1.0 * 1 s
    env.step(MAINTAIN)
    assert road.speed_commands[-1] == pytest.approx(6.0)
    env.step(DECELERATE)
    assert road.speed_commands[-1] == pytest.approx(4.0)     # 6 - 2.0 * 1 s

    for _ in range(10):
        env.step(ACCELERATE)
    assert max(road.speed_commands) <= FakeRoad.ALLOWED_SPEED
    for _ in range(10):
        env.step(DECELERATE)
    assert min(road.speed_commands) >= 0.0
    env.close()


def test_vehicle_reward_matches_formula(tmp_path, fake_traci):
    fake_traci(FakeRoad, leader_gap=10.0, leader_speed=4.0)
    env = make_vehicle_env(tmp_path)
    env.reset(seed=0)
    _, reward, *_ = env.step(MAINTAIN)
    s = env.state
    headway = s["gap"] / max(s["speed"], 0.1)
    expected = -((1 - s["speed"] / 10.0) + max(0.0, 1 - headway / 1.5) + 0.1 * (s["acceleration"] / 2.0) ** 2)
    assert reward == pytest.approx(expected)
    assert s["leader_present"]
    env.close()


def test_vehicle_arrival_terminates_without_querying_missing_vehicle(tmp_path, fake_traci):
    fake_traci(FakeRoad)
    env = make_vehicle_env(tmp_path)
    env.reset(seed=0)
    done, steps = False, 0
    while not done:
        observation, reward, terminated, truncated, info = env.step(ACCELERATE)  # fake raises on unknown IDs
        done = terminated or truncated
        steps += 1
    assert terminated and not truncated
    assert info["arrived"] and not info["collided"]
    assert observation[4] == 0.0 and reward == 0.0
    assert steps < 60
    env.close()


def test_vehicle_episode_truncates_if_it_never_arrives(tmp_path, fake_traci):
    fake_traci(FakeRoad)
    env = make_vehicle_env(tmp_path, max_episode_seconds=20)
    env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(DECELERATE)
        done = terminated or truncated
    assert truncated and not terminated and not info["arrived"]
    assert info["travel_time"] == pytest.approx(20.0)
    env.close()


def test_vehicle_collision_terminates_with_penalty(tmp_path, fake_traci):
    fake_traci(FakeRoad, collide_at=5.0)
    env = make_vehicle_env(tmp_path)
    env.reset(seed=0)
    env.step(MAINTAIN)                                       # t = 4
    _, reward, terminated, _, info = env.step(MAINTAIN)      # t = 5
    assert terminated and info["collided"]
    assert reward <= -COLLISION_PENALTY
    env.close()


def test_sumo_driver_baseline_sends_no_speed_commands(tmp_path, fake_traci):
    fake_traci(FakeRoad)
    env = make_vehicle_env(tmp_path, sumo_drives_vehicle=True)
    env.reset(seed=0)
    for _ in range(5):
        env.step(ACCELERATE)
    assert env.sumo.speed_commands == [] and env.sumo.speed_mode is None
    env.close()
