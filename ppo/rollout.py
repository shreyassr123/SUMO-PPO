"""Interaction loops between an agent and a Gymnasium-style environment.

``RolloutCollector`` gathers on-policy training data for PPO.
``evaluate_policy`` runs complete episodes with any policy, including the
non-learning baselines, and records the same metrics for each of them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .buffer import RolloutBuffer

if TYPE_CHECKING:  # imported for type hints only, so baselines run without PyTorch
    from .agent import PPOAgent

# A policy maps (observation, info) to an action. ``info`` lets rule-based
# baselines read quantities, such as elapsed green time, that are not part of
# the learning agent's observation.
Policy = Callable[[np.ndarray, dict], int]


def scalar_metrics(info: dict[str, Any]) -> dict[str, float]:
    """Keep the numeric entries of an ``info`` dictionary (for CSV output)."""
    return {
        key: float(value)
        for key, value in info.items()
        if isinstance(value, (int, float, bool, np.integer, np.floating))
    }


def _check_finite(observation: np.ndarray) -> np.ndarray:
    observation = np.asarray(observation, dtype=np.float32)
    if not np.all(np.isfinite(observation)):
        raise ValueError(f"Environment returned a non-finite observation: {observation}")
    return observation


class RolloutCollector:
    """Steps one environment with the current policy and fills a buffer.

    The collector keeps the environment's current observation between calls,
    so an episode can start in one rollout and finish in the next. This is
    standard PPO: the rollout length is fixed in environment steps, not in
    episodes.
    """

    def __init__(self, env, seed: int | None = None):
        self.env = env
        self.observation = _check_finite(env.reset(seed=seed)[0])
        self.episode_return = 0.0
        self.episode_length = 0
        self.episodes_completed = 0

    def collect_rollout(
        self,
        agent: PPOAgent,
        buffer: RolloutBuffer,
        max_episodes: int | None = None,
    ) -> list[dict[str, float]]:
        """Fill ``buffer`` with fresh experience and compute its advantages.

        Collection stops when the buffer is full or, if ``max_episodes`` is
        given, as soon as that many episodes have been completed in total.
        Returns one summary dictionary for each episode finished during this
        rollout.
        """
        buffer.reset()
        finished: list[dict[str, float]] = []

        while not buffer.is_full():
            action, log_prob, value = agent.act(self.observation)
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            next_observation = _check_finite(next_observation)

            # A time limit does not end the task itself, so the value of the
            # state where the episode was cut off is still V(s), not zero.
            final_value = agent.predict_value(next_observation) if truncated and not terminated else 0.0
            buffer.add(self.observation, action, log_prob, value, float(reward),
                       bool(terminated), bool(truncated), final_value)

            self.episode_return += float(reward)
            self.episode_length += 1
            self.observation = next_observation

            if terminated or truncated:
                self.episodes_completed += 1
                finished.append({
                    "episode": self.episodes_completed,
                    "return": self.episode_return,
                    "length": self.episode_length,
                    "terminated": bool(terminated),
                    **scalar_metrics(info),
                })
                self.episode_return, self.episode_length = 0.0, 0
                if max_episodes is not None and self.episodes_completed >= max_episodes:
                    break
                self.observation = _check_finite(self.env.reset()[0])

        # If the last stored step ended an episode this value is ignored.
        last_value = agent.predict_value(self.observation)
        buffer.compute_advantages(last_value, agent.config.gamma, agent.config.gae_lambda)
        return finished


def agent_policy(agent: PPOAgent, deterministic: bool = True) -> Policy:
    """Wrap a PPO agent as a ``Policy`` for ``evaluate_policy``."""
    return lambda observation, info: agent.act(observation, deterministic=deterministic)[0]


def evaluate_policy(env, policy: Policy, episodes: int, seed: int) -> list[dict[str, float]]:
    """Run ``episodes`` complete episodes and return one record per episode.

    Episode ``i`` is reset with seed ``seed + i``. Two controllers evaluated
    with the same ``seed`` therefore face exactly the same traffic, which
    makes their results directly comparable.
    """
    records = []
    for i in range(episodes):
        observation, info = env.reset(seed=seed + i)
        episode_return, length, done = 0.0, 0, False
        while not done:
            action = policy(observation, info)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            length += 1
            done = terminated or truncated
        records.append({
            "episode": i + 1,
            "seed": seed + i,
            "return": episode_return,
            "length": length,
            "terminated": bool(terminated),
            **scalar_metrics(info),
        })
    return records
