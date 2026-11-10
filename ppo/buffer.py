"""Rollout storage and Generalised Advantage Estimation (GAE).

This module uses only NumPy so the advantage calculation can be read and
tested independently of PyTorch.
"""

from __future__ import annotations

import numpy as np


class RolloutBuffer:
    """Fixed-capacity store for the transitions of one PPO rollout.

    Entry ``t`` holds the observation s_t, the action a_t taken in it, the
    behaviour policy's log-probability log pi_old(a_t | s_t), the critic's
    estimate V(s_t), and the reward r_t received *after* taking a_t.

    A rollout may contain several episodes and may end in the middle of one.
    Episode boundaries are described by two flags:

    ``terminated[t]``
        s_{t+1} is a true terminal state (for example, the vehicle reached its
        destination). Its value is zero by definition.
    ``truncated[t]``
        The episode was stopped by a time limit. The state s_{t+1} still has a
        future, so we bootstrap from ``final_values[t]`` = V(s_{t+1}).
    """

    def __init__(self, capacity: int, observation_size: int):
        if capacity < 1:
            raise ValueError("Buffer capacity must be at least 1.")
        self.capacity = capacity
        self.observation_size = observation_size

        self.observations = np.zeros((capacity, observation_size), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.terminated = np.zeros(capacity, dtype=bool)
        self.truncated = np.zeros(capacity, dtype=bool)
        self.final_values = np.zeros(capacity, dtype=np.float32)

        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.size = 0

    def reset(self) -> None:
        self.size = 0

    def is_full(self) -> bool:
        return self.size >= self.capacity

    def add(
        self,
        observation: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        terminated: bool,
        truncated: bool,
        final_value: float = 0.0,
    ) -> None:
        if self.is_full():
            raise RuntimeError("RolloutBuffer is full; call reset() before adding more data.")
        t = self.size
        self.observations[t] = observation
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.final_values[t] = final_value
        self.size += 1

    def compute_advantages(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and critic targets for the stored rollout.

        For each step, working backwards in time:

            delta_t = r_t + gamma * V_next - V(s_t)
            A_t     = delta_t + gamma * lambda * c_t * A_{t+1}
            G_t     = A_t + V(s_t)

        where V_next is V(s_{t+1}) inside an episode, ``final_values[t]`` after
        a truncation, zero after a termination, and ``last_value`` for the final
        stored step when the rollout stops mid-episode. ``c_t`` is zero at an
        episode boundary so that advantages never leak between episodes.

        Args:
            last_value: V(s) for the observation that follows the last stored
                step. Ignored if that step ended an episode.
            gamma: Discount factor.
            gae_lambda: GAE parameter; 0 gives one-step TD errors, 1 gives
                Monte Carlo returns minus the value baseline.
        """
        n = self.size
        advantage = 0.0
        for t in reversed(range(n)):
            if self.terminated[t]:
                next_value, continues = 0.0, 0.0
            elif self.truncated[t]:
                next_value, continues = float(self.final_values[t]), 0.0
            elif t == n - 1:
                next_value, continues = float(last_value), 1.0
            else:
                next_value, continues = float(self.values[t + 1]), 1.0

            delta = self.rewards[t] + gamma * next_value - self.values[t]
            advantage = delta + gamma * gae_lambda * continues * advantage
            self.advantages[t] = advantage

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def data(self) -> dict[str, np.ndarray]:
        """Return views of the filled part of the buffer."""
        n = self.size
        return {
            "observations": self.observations[:n],
            "actions": self.actions[:n],
            "log_probs": self.log_probs[:n],
            "values": self.values[:n],
            "advantages": self.advantages[:n],
            "returns": self.returns[:n],
        }
