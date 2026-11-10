"""Actor and critic networks used by the PPO agent.

The architecture follows the original Mobility_PPO learner: a two-layer tanh
actor with a small-gain output layer and a slightly wider two-layer tanh critic.
Both networks take the same observation vector as input.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network pi_theta(a | s) for a discrete action space.

    The network outputs one logit per action. A softmax over the logits gives
    the action probabilities, wrapped here in a ``Categorical`` distribution.
    """

    def __init__(self, observation_size: int, n_actions: int, hidden_size: int = 64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.logits = nn.Linear(hidden_size, n_actions)

        # A small output layer makes the initial policy close to uniform, so
        # every action is explored before the agent has learned anything.
        nn.init.orthogonal_(self.logits.weight, gain=0.01)
        nn.init.zeros_(self.logits.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.logits(self.body(observations))

    def distribution(self, observations: torch.Tensor) -> Categorical:
        return Categorical(logits=self.forward(observations))


class Critic(nn.Module):
    """State-value network V_phi(s).

    The critic predicts *normalised* returns; ``PPOAgent`` converts them back
    to reward units using running return statistics.
    """

    def __init__(self, observation_size: int, hidden_size: int = 128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.body(observations).squeeze(-1)
