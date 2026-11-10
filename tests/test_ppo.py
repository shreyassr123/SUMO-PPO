"""PPO agent: networks, action selection, update step and checkpoints.

Requires PyTorch. The final test runs a short training loop on Gymnasium's
CartPole to check that PPO works end to end without SUMO.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ppo.agent import PPOAgent, PPOConfig, RunningMeanStd, explained_variance, set_seed  # noqa: E402
from ppo.buffer import RolloutBuffer  # noqa: E402
from ppo.networks import Actor, Critic  # noqa: E402


def random_rollout(agent, n=64, seed=0):
    """Fill a buffer with random observations and rewards, acting with the agent."""
    rng = np.random.default_rng(seed)
    buffer = RolloutBuffer(n, agent.observation_size)
    for t in range(n):
        observation = rng.normal(size=agent.observation_size).astype(np.float32)
        action, log_prob, value = agent.act(observation)
        buffer.add(observation, action, log_prob, value, float(rng.normal()),
                   terminated=(t % 16 == 15), truncated=False)
    buffer.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    return buffer


def test_network_output_shapes():
    actor, critic = Actor(5, 3), Critic(5)
    batch = torch.zeros(7, 5)
    assert actor(batch).shape == (7, 3)
    assert critic(batch).shape == (7,)


def test_action_probabilities_and_log_probabilities_agree():
    set_seed(0)
    agent = PPOAgent(4, 3)
    observation = np.array([0.1, -0.2, 0.3, 0.0], dtype=np.float32)
    probabilities = agent.action_probabilities(observation)
    assert probabilities.shape == (3,)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    # A fresh actor is close to uniform because of the small output layer.
    np.testing.assert_allclose(probabilities, 1 / 3, atol=0.02)

    action, log_prob, value = agent.act(observation)
    assert action in (0, 1, 2)
    assert log_prob == pytest.approx(np.log(probabilities[action]), abs=1e-5)
    assert np.isfinite(value)


def test_deterministic_action_is_most_probable():
    set_seed(1)
    agent = PPOAgent(2, 4)
    with torch.no_grad():
        agent.actor.logits.bias.copy_(torch.tensor([0.0, 0.0, 5.0, 0.0]))
    for _ in range(5):
        assert agent.act(np.zeros(2, dtype=np.float32), deterministic=True)[0] == 2


def test_wrong_observation_size_is_reported():
    agent = PPOAgent(3, 2)
    with pytest.raises(ValueError, match="expects 3"):
        agent.act(np.zeros(4, dtype=np.float32))


def test_update_returns_finite_diagnostics_and_changes_weights():
    set_seed(2)
    agent = PPOAgent(6, 3, PPOConfig(minibatch_size=16, update_epochs=4))
    buffer = random_rollout(agent)
    actor_before = [p.detach().clone() for p in agent.actor.parameters()]
    critic_before = [p.detach().clone() for p in agent.critic.parameters()]

    stats = agent.update_policy(buffer)

    for key in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance"):
        assert np.isfinite(stats[key]), key
    assert stats["actor_updates"] >= 1
    assert any(not torch.equal(a, b) for a, b in zip(actor_before, agent.actor.parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(critic_before, agent.critic.parameters()))


def test_first_minibatch_ratio_is_one():
    """Before any update the new and old policies coincide, so KL and clipping are zero."""
    set_seed(3)
    agent = PPOAgent(4, 2, PPOConfig(minibatch_size=1000, update_epochs=1))
    stats = agent.update_policy(random_rollout(agent))
    assert stats["approx_kl"] == pytest.approx(0.0, abs=1e-6)
    assert stats["clip_fraction"] == 0.0


def test_kl_early_stop_halts_actor_only():
    set_seed(4)
    config = PPOConfig(minibatch_size=8, update_epochs=20, actor_lr=0.05, target_kl=1e-4)
    agent = PPOAgent(4, 3, config)
    stats = agent.update_policy(random_rollout(agent))
    critic_minibatches = config.update_epochs * (64 // config.minibatch_size)
    assert 1 <= stats["actor_updates"] < critic_minibatches


def test_running_mean_std_matches_numpy():
    rng = np.random.default_rng(0)
    batches = [rng.normal(3.0, 2.0, size=n) for n in (10, 50, 7)]
    normalizer = RunningMeanStd()
    for batch in batches:
        normalizer.update(batch)
    everything = np.concatenate(batches)
    assert normalizer.mean == pytest.approx(everything.mean(), rel=1e-3)
    assert normalizer.std == pytest.approx(everything.std(), rel=1e-3)


def test_explained_variance():
    targets = np.array([1.0, 2.0, 3.0])
    assert explained_variance(targets, targets) == pytest.approx(1.0)
    assert explained_variance(np.full(3, targets.mean()), targets) == pytest.approx(0.0)


def test_checkpoint_round_trip(tmp_path):
    set_seed(5)
    agent = PPOAgent(4, 3, PPOConfig(gamma=0.95))
    agent.update_policy(random_rollout(agent))
    path = tmp_path / "agent.pt"
    agent.save_checkpoint(path, metadata={"env": "test", "seed": 5})

    loaded, metadata = PPOAgent.load_checkpoint(path)
    assert metadata == {"env": "test", "seed": 5}
    assert loaded.config.gamma == 0.95
    observation = np.array([0.3, -1.0, 0.5, 2.0], dtype=np.float32)
    np.testing.assert_allclose(loaded.action_probabilities(observation), agent.action_probabilities(observation))
    assert loaded.predict_value(observation) == pytest.approx(agent.predict_value(observation))


def test_loading_a_foreign_file_fails_clearly(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(2)}, path)
    with pytest.raises(ValueError, match="save_checkpoint"):
        PPOAgent.load_checkpoint(path)
    with pytest.raises(FileNotFoundError):
        PPOAgent.load_checkpoint(tmp_path / "missing.pt")


def test_short_cartpole_training_and_evaluation(tmp_path):
    gym = pytest.importorskip("gymnasium")
    from ppo.rollout import RolloutCollector, agent_policy, evaluate_policy

    set_seed(0)
    env = gym.make("CartPole-v1")
    agent = PPOAgent(4, 2, PPOConfig(rollout_steps=512, minibatch_size=64))
    buffer = RolloutBuffer(agent.config.rollout_steps, 4)
    collector = RolloutCollector(env, seed=0)
    episodes = []
    for _ in range(3):
        episodes += collector.collect_rollout(agent, buffer)
        stats = agent.update_policy(buffer)
        assert all(np.isfinite(v) for v in stats.values())
    assert len(episodes) > 0

    agent.save_checkpoint(tmp_path / "cartpole.pt", {"env": "cartpole"})
    loaded, _ = PPOAgent.load_checkpoint(tmp_path / "cartpole.pt")
    records = evaluate_policy(env, agent_policy(loaded), episodes=2, seed=123)
    env.close()
    assert len(records) == 2 and all(r["length"] >= 1 for r in records)
