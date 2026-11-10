"""Integration tests against a real SUMO installation.

Skipped automatically when TraCI or the ``sumo`` program cannot be found, or
when the network files have not been generated yet
(``python scenarios/build_networks.py``).
"""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from envs.sumo_base import SCENARIO_DIR, find_sumo_tool, import_traci  # noqa: E402


def _sumo_available():
    try:
        import_traci()
        find_sumo_tool("sumo")
    except (ImportError, FileNotFoundError):
        return False
    return all((SCENARIO_DIR / s / "network.net.xml").is_file() for s in ("simple_intersection", "vehicle_control"))


pytestmark = pytest.mark.skipif(not _sumo_available(), reason="SUMO, TraCI or the built networks are not available")

from envs.traffic_signal_env import SWITCH_PHASE, TrafficSignalEnv  # noqa: E402
from envs.vehicle_control_env import ACCELERATE, DECELERATE, MAINTAIN, VehicleControlEnv  # noqa: E402


def test_gymnasium_api_checks():
    from gymnasium.utils.env_checker import check_env
    for env in (TrafficSignalEnv(episode_seconds=60), VehicleControlEnv(max_episode_seconds=30)):
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()


def test_signal_environment_short_episode():
    env = TrafficSignalEnv(episode_seconds=600)
    try:
        observation, info = env.reset(seed=0)
        assert env.observation_space.contains(observation)
        rng = np.random.default_rng(0)
        done = False
        while not done:
            observation, reward, terminated, truncated, info = env.step(int(rng.integers(2)))
            assert env.observation_space.contains(observation) and np.isfinite(reward)
            done = terminated or truncated
        assert truncated and info["simulation_time"] >= 600
        assert info["throughput"] > 0 and info["phase_switches"] > 0
    finally:
        env.close()


def test_sumo_keeps_the_signal_state_set_by_the_environment():
    env = TrafficSignalEnv()
    try:
        env.reset(seed=0)
        log = []
        original_set_signal = env.set_signal

        def logged_set_signal(state):
            log.append((env.simulation_time, state))
            original_set_signal(state)
        env.set_signal = logged_set_signal
        for _ in range(4):
            env.step(SWITCH_PHASE)
            # Between decisions SUMO must not advance the signal on its own.
            assert env.sumo.trafficlight.getRedYellowGreenState(env.tls_id) == env.green_states[env.green_index]
        states = [state for _, state in log]
        assert "y" in states[0] and set(states[1]) == {"r"} and states[2] == env.green_states[1]
        times = [t for t, _ in log]
        assert times[1] - times[0] == env.yellow_time and times[2] - times[1] == env.all_red_time
    finally:
        env.close()


def test_repeated_resets_and_close():
    env = TrafficSignalEnv(episode_seconds=30)
    for seed in range(3):
        env.reset(seed=seed)
        env.step(0)
    env.close()
    assert env.sumo is None
    env.close()  # closing twice is harmless


def test_vehicle_speed_commands_take_effect():
    env = VehicleControlEnv()
    try:
        env.reset(seed=0)
        for _ in range(3):
            env.step(DECELERATE)
        slow = env.state["speed"]
        for _ in range(3):
            env.step(ACCELERATE)
        assert env.state["speed"] <= env.state["allowed_speed"] + 1e-6
        if not env.state["leader_present"] or env.state["gap"] > 50:
            assert env.state["speed"] > slow
    finally:
        env.close()


def test_vehicle_episode_ends_on_arrival():
    env = VehicleControlEnv()
    try:
        env.reset(seed=3)
        done = False
        while not done:
            observation, reward, terminated, truncated, info = env.step(ACCELERATE)
            assert env.observation_space.contains(observation) and np.isfinite(reward)
            done = terminated or truncated
        assert terminated and info["arrived"] and not info["collided"]
    finally:
        env.close()


def test_sumo_driver_baseline_completes():
    env = VehicleControlEnv(sumo_drives_vehicle=True)
    try:
        env.reset(seed=1)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(MAINTAIN)
            done = terminated or truncated
        assert info["arrived"]
    finally:
        env.close()
