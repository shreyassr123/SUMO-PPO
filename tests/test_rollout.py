"""Rollout collection and evaluation loops, using a toy environment and agent.

These tests check the bookkeeping around episode boundaries without needing
PyTorch, Gymnasium or SUMO.
"""

from types import SimpleNamespace

import numpy as np

from ppo.buffer import RolloutBuffer
from ppo.rollout import RolloutCollector, evaluate_policy


class CountingEnv:
    """Observation = step counter. Terminates at `terminate_at`, truncates at `limit`."""

    def __init__(self, terminate_at=None, limit=3):
        self.terminate_at, self.limit = terminate_at, limit
        self.resets, self.seeds = 0, []

    def reset(self, seed=None):
        self.resets += 1
        self.seeds.append(seed)
        self.t = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action):
        self.t += 1
        terminated = self.terminate_at is not None and self.t >= self.terminate_at
        truncated = not terminated and self.t >= self.limit
        return np.array([float(self.t)], dtype=np.float32), 1.0, terminated, truncated, {"t": self.t, "name": "x"}


class ConstantAgent:
    """Always takes action 0; the value of an observation is 10 * observation."""

    def __init__(self):
        self.config = SimpleNamespace(gamma=1.0, gae_lambda=1.0)
        self.value_queries = []

    def act(self, observation, deterministic=False):
        return 0, -0.5, float(10 * observation[0])

    def predict_value(self, observation):
        self.value_queries.append(float(observation[0]))
        return float(10 * observation[0])


def test_truncated_episode_bootstraps_from_final_observation():
    env, agent = CountingEnv(limit=3), ConstantAgent()
    buffer = RolloutBuffer(3, observation_size=1)
    collector = RolloutCollector(env, seed=7)
    finished = collector.collect_rollout(agent, buffer)

    assert env.seeds[0] == 7
    assert buffer.truncated[:3].tolist() == [False, False, True]
    assert buffer.final_values[2] == 30.0          # V(final observation = 3)
    assert len(finished) == 1 and finished[0]["return"] == 3.0 and finished[0]["length"] == 3
    assert finished[0]["t"] == 3.0 and "name" not in finished[0]   # only numeric info kept
    # Return target of the last step: r + gamma * V(s_final) = 1 + 30.
    assert buffer.returns[2] == 31.0


def test_terminated_episode_does_not_bootstrap():
    env, agent = CountingEnv(terminate_at=2, limit=10), ConstantAgent()
    buffer = RolloutBuffer(2, observation_size=1)
    RolloutCollector(env).collect_rollout(agent, buffer)
    assert buffer.terminated[1] and buffer.final_values[1] == 0.0
    assert buffer.returns[1] == 1.0


def test_episodes_continue_across_rollouts():
    env, agent = CountingEnv(limit=3), ConstantAgent()
    buffer = RolloutBuffer(2, observation_size=1)
    collector = RolloutCollector(env)
    assert collector.collect_rollout(agent, buffer) == []      # steps 1-2 of episode 1
    finished = collector.collect_rollout(agent, buffer)        # step 3 ends it, step 1 of episode 2
    assert len(finished) == 1 and finished[0]["length"] == 3
    assert collector.episode_length == 1


def test_collection_stops_at_episode_limit_without_extra_reset():
    env, agent = CountingEnv(limit=2), ConstantAgent()
    buffer = RolloutBuffer(100, observation_size=1)
    collector = RolloutCollector(env)
    finished = collector.collect_rollout(agent, buffer, max_episodes=3)
    assert len(finished) == 3 and buffer.size == 6
    assert env.resets == 3       # no fourth simulation started


def test_evaluate_policy_uses_consecutive_seeds():
    env = CountingEnv(limit=4)
    records = evaluate_policy(env, lambda obs, info: 0, episodes=3, seed=100)
    assert env.seeds == [100, 101, 102]
    assert [r["seed"] for r in records] == [100, 101, 102]
    assert all(r["return"] == 4.0 and r["length"] == 4 for r in records)
