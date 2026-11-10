"""Proximal Policy Optimisation (PPO) with a clipped surrogate objective.

The agent knows nothing about SUMO. It needs only the size of the observation
vector and the number of discrete actions, so the same class trains on the
traffic-signal example, the vehicle-control example and CartPole.

The learner is adapted from the PPO baseline in the original Mobility_PPO
research code: separate actor and critic optimisers, running normalisation
of the critic's targets, whole-rollout advantage normalisation, gradient
clipping, and an approximate-KL early stop that halts only the actor.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .buffer import RolloutBuffer
from .networks import Actor, Critic

CHECKPOINT_FORMAT = "sumo-ppo-checkpoint-v1"


@dataclass
class PPOConfig:
    """PPO hyperparameters. Defaults follow the original research code."""

    gamma: float = 0.99               # discount factor
    gae_lambda: float = 0.95          # GAE bias/variance trade-off
    clip_range: float = 0.2           # epsilon in the clipped objective
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    rollout_steps: int = 2048         # environment steps collected per update
    update_epochs: int = 10           # passes over each rollout
    minibatch_size: int = 256
    entropy_coef: float = 0.01        # weight of the entropy bonus
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.02    # stop actor updates beyond this KL; None disables
    actor_hidden_size: int = 64
    critic_hidden_size: int = 128


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class RunningMeanStd:
    """Running mean and variance of the critic's return targets.

    The critic regresses normalised targets, so its loss stays on a similar
    scale whether the rewards are of order 1 (CartPole) or of order 100
    (queued vehicle-seconds at an intersection).
    """

    def __init__(self) -> None:
        self.mean, self.var, self.count = 0.0, 1.0, 1e-4

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        batch_mean, batch_var, batch_count = values.mean(), values.var(), values.size
        delta = batch_mean - self.mean
        total = self.count + batch_count
        # Parallel-variance combination of the old and new statistics.
        m2 = self.var * self.count + batch_var * batch_count + delta**2 * self.count * batch_count / total
        self.mean = float(self.mean + delta * batch_count / total)
        self.var = float(max(m2 / total, 1e-6))
        self.count = float(total)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def state_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.mean, self.var, self.count = state["mean"], state["var"], state["count"]


class PPOAgent:
    """Actor-critic PPO agent for discrete actions."""

    def __init__(self, observation_size: int, n_actions: int, config: PPOConfig | None = None):
        self.observation_size = int(observation_size)
        self.n_actions = int(n_actions)
        self.config = config or PPOConfig()

        self.actor = Actor(self.observation_size, self.n_actions, self.config.actor_hidden_size)
        self.critic = Critic(self.observation_size, self.config.critic_hidden_size)
        # Separate optimisers let the critic learn faster than the policy.
        # Because the networks share no parameters, the usual value-loss
        # coefficient would only rescale the critic's learning rate, so it is
        # not needed.
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr, eps=1e-5)
        self.return_normalizer = RunningMeanStd()

    # ------------------------------------------------------------------ acting

    def _to_tensor(self, observation: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(np.asarray(observation), dtype=torch.float32)
        if tensor.shape[-1] != self.observation_size:
            raise ValueError(
                f"Observation has {tensor.shape[-1]} values but the agent expects "
                f"{self.observation_size}. Did the environment's observation change "
                "after the agent (or checkpoint) was created?"
            )
        return tensor

    def _value_in_reward_units(self, normalised_value: torch.Tensor) -> torch.Tensor:
        return normalised_value * self.return_normalizer.std + self.return_normalizer.mean

    @torch.no_grad()
    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        """Choose an action for one observation.

        Returns ``(action, log_prob, value)``. During training actions are
        sampled from pi(a|s) so that the agent keeps exploring. With
        ``deterministic=True`` the most probable action is taken instead,
        which is the usual choice for evaluation.
        """
        obs = self._to_tensor(observation).unsqueeze(0)
        dist = self.actor.distribution(obs)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        value = self._value_in_reward_units(self.critic(obs))
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    @torch.no_grad()
    def predict_value(self, observation: np.ndarray) -> float:
        """Return the critic's estimate V(s) in reward units."""
        obs = self._to_tensor(observation).unsqueeze(0)
        return float(self._value_in_reward_units(self.critic(obs)).item())

    @torch.no_grad()
    def action_probabilities(self, observation: np.ndarray) -> np.ndarray:
        obs = self._to_tensor(observation).unsqueeze(0)
        return self.actor.distribution(obs).probs.squeeze(0).numpy()

    # ---------------------------------------------------------------- learning

    def update_policy(self, buffer: RolloutBuffer) -> dict[str, float]:
        """Run the PPO optimisation phase on one rollout.

        ``buffer.compute_advantages`` must have been called first. The rollout
        is reused for ``update_epochs`` passes, each split into shuffled
        minibatches. Every minibatch makes one critic step and, unless the KL
        early stop has fired, one actor step.

        Returns a dictionary of diagnostics averaged over the minibatches:
        ``policy_loss`` (the negated clipped surrogate), ``value_loss`` (mean
        squared error on normalised returns), ``entropy``, ``approx_kl``
        (between the behaviour policy and the updated policy), ``clip_fraction``
        (share of samples whose ratio left [1-eps, 1+eps]), ``explained_variance``
        of the critic before the update, and ``actor_updates``.
        """
        cfg = self.config
        data = buffer.data()
        n = len(data["actions"])
        if n == 0:
            raise ValueError("Cannot update from an empty rollout.")

        observations = self._to_tensor(data["observations"])
        actions = torch.as_tensor(data["actions"], dtype=torch.long)
        old_log_probs = torch.as_tensor(data["log_probs"], dtype=torch.float32)
        returns = data["returns"].astype(np.float64)

        # The critic learns normalised targets (see RunningMeanStd).
        self.return_normalizer.update(returns)
        value_targets = torch.as_tensor(
            (returns - self.return_normalizer.mean) / self.return_normalizer.std, dtype=torch.float32
        )

        # Normalising advantages over the whole rollout keeps the policy step
        # size independent of the reward scale.
        advantages = torch.as_tensor(data["advantages"], dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        actor_updates = critic_updates = 0
        actor_stopped = False

        for _ in range(cfg.update_epochs):
            permutation = torch.randperm(n)
            for start in range(0, n, cfg.minibatch_size):
                batch = permutation[start:start + cfg.minibatch_size]

                # Critic: regress V(s) towards the GAE return targets.
                value_loss = ((self.critic(observations[batch]) - value_targets[batch]) ** 2).mean()
                self._optimise(self.critic_optimizer, self.critic, value_loss)
                totals["value_loss"] += value_loss.item()
                critic_updates += 1

                if actor_stopped:
                    continue

                dist = self.actor.distribution(observations[batch])
                new_log_probs = dist.log_prob(actions[batch])
                entropy = dist.entropy().mean()

                # Compare the updated policy with the behaviour policy that
                # generated the rollout. Clipping limits excessively large updates.
                log_ratio = new_log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Low-variance estimator of KL(pi_old || pi_new).
                    approx_kl = ((ratio - 1.0) - log_ratio).mean().item()
                    clip_fraction = ((ratio - 1.0).abs() > cfg.clip_range).float().mean().item()
                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    # The policy has moved far enough from the data-collecting
                    # policy; further actor steps on this rollout are unreliable.
                    actor_stopped = True
                    continue

                batch_advantages = advantages[batch]
                unclipped = ratio * batch_advantages
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Subtracting the entropy rewards uncertainty and discourages
                # the policy from collapsing onto one action too early.
                actor_loss = policy_loss - cfg.entropy_coef * entropy
                self._optimise(self.actor_optimizer, self.actor, actor_loss)

                totals["policy_loss"] += policy_loss.item()
                totals["entropy"] += entropy.item()
                totals["approx_kl"] += approx_kl
                totals["clip_fraction"] += clip_fraction
                actor_updates += 1

        stats = {"value_loss": totals["value_loss"] / critic_updates}
        for key in ("policy_loss", "entropy", "approx_kl", "clip_fraction"):
            stats[key] = totals[key] / actor_updates if actor_updates else float("nan")
        stats["explained_variance"] = explained_variance(data["values"], data["returns"])
        stats["actor_updates"] = float(actor_updates)
        return stats

    def _optimise(self, optimizer: torch.optim.Optimizer, network: nn.Module, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "PPO produced a non-finite loss. Check the environment for NaN or very "
                "large observations or rewards, or try a smaller learning rate."
            )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(), self.config.max_grad_norm)
        optimizer.step()

    # ------------------------------------------------------------- checkpoints

    def save_checkpoint(self, path: str | Path, metadata: dict | None = None) -> None:
        """Save the networks, normaliser and configuration to ``path``.

        ``metadata`` can hold anything JSON-like that identifies the run, such
        as the environment name and seed; it is returned by ``load_checkpoint``.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "observation_size": self.observation_size,
                "n_actions": self.n_actions,
                "config": asdict(self.config),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "return_normalizer": self.return_normalizer.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> tuple[PPOAgent, dict]:
        """Rebuild an agent from a checkpoint. Returns ``(agent, metadata)``."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path} was not written by PPOAgent.save_checkpoint().")

        known = {f.name for f in fields(PPOConfig)}
        config = PPOConfig(**{k: v for k, v in checkpoint["config"].items() if k in known})
        agent = cls(checkpoint["observation_size"], checkpoint["n_actions"], config)
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.return_normalizer.load_state_dict(checkpoint["return_normalizer"])
        return agent, checkpoint["metadata"]


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    """1 - Var(targets - predictions) / Var(targets).

    1 means the critic predicts the returns perfectly, 0 means it does no
    better than a constant, and negative values mean it is worse than a
    constant. NaN is returned when the targets have no variance.
    """
    target_var = float(np.var(targets))
    if target_var == 0.0:
        return float("nan")
    return 1.0 - float(np.var(targets - predictions)) / target_var
