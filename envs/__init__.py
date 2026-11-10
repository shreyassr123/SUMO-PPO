"""Environments that connect PPO to SUMO (and CartPole for testing PPO alone)."""

from __future__ import annotations

import gymnasium as gym

from .traffic_signal_env import TrafficSignalEnv
from .vehicle_control_env import VehicleControlEnv

SUMO_ENVIRONMENTS = {
    "traffic_signal": TrafficSignalEnv,
    "vehicle_control": VehicleControlEnv,
}
ENVIRONMENT_NAMES = [*SUMO_ENVIRONMENTS, "cartpole"]


def make_env(name: str, gui: bool = False, **kwargs) -> gym.Env:
    """Create an environment by name.

    ``kwargs`` are passed to the environment's constructor, e.g.
    ``make_env("traffic_signal", min_green=15)``. ``"cartpole"`` returns
    Gymnasium's CartPole-v1, which is useful for checking PPO without SUMO.
    To add your own environment, register its class in ``SUMO_ENVIRONMENTS``.
    """
    if name == "cartpole":
        return gym.make("CartPole-v1")
    if name not in SUMO_ENVIRONMENTS:
        raise ValueError(f"Unknown environment '{name}'. Choose from: {', '.join(ENVIRONMENT_NAMES)}")
    return SUMO_ENVIRONMENTS[name](gui=gui, **kwargs)
