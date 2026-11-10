"""GAE and return calculations, checked against values worked out by hand."""

import numpy as np
import pytest

from ppo.buffer import RolloutBuffer


def fill(buffer, rewards, values, terminated=None, truncated=None, final_values=None):
    n = len(rewards)
    terminated = terminated or [False] * n
    truncated = truncated or [False] * n
    final_values = final_values or [0.0] * n
    for t in range(n):
        buffer.add(np.zeros(buffer.observation_size), 0, 0.0, values[t], rewards[t],
                   terminated[t], truncated[t], final_values[t])


def test_terminal_episode_matches_hand_calculation():
    gamma, lam = 0.9, 0.8
    buffer = RolloutBuffer(3, observation_size=2)
    fill(buffer, rewards=[1.0, 2.0, 3.0], values=[0.5, 1.0, 1.5], terminated=[False, False, True])
    buffer.compute_advantages(last_value=99.0, gamma=gamma, gae_lambda=lam)  # 99 must be ignored

    delta2 = 3.0 - 1.5                         # terminal: no bootstrap
    delta1 = 2.0 + gamma * 1.5 - 1.0
    delta0 = 1.0 + gamma * 1.0 - 0.5
    a2 = delta2
    a1 = delta1 + gamma * lam * a2
    a0 = delta0 + gamma * lam * a1
    np.testing.assert_allclose(buffer.advantages[:3], [a0, a1, a2], rtol=1e-6)
    np.testing.assert_allclose(buffer.returns[:3], [a0 + 0.5, a1 + 1.0, a2 + 1.5], rtol=1e-6)


def test_truncation_bootstraps_from_final_value():
    buffer = RolloutBuffer(1, observation_size=1)
    fill(buffer, rewards=[1.0], values=[2.0], truncated=[True], final_values=[10.0])
    buffer.compute_advantages(last_value=0.0, gamma=0.5, gae_lambda=0.95)
    assert buffer.advantages[0] == pytest.approx(1.0 + 0.5 * 10.0 - 2.0)


def test_termination_ignores_final_value():
    buffer = RolloutBuffer(1, observation_size=1)
    fill(buffer, rewards=[1.0], values=[2.0], terminated=[True], final_values=[10.0])
    buffer.compute_advantages(last_value=0.0, gamma=0.5, gae_lambda=0.95)
    assert buffer.advantages[0] == pytest.approx(1.0 - 2.0)


def test_rollout_cut_mid_episode_uses_last_value():
    buffer = RolloutBuffer(2, observation_size=1)
    fill(buffer, rewards=[0.0, 1.0], values=[0.0, 0.0])
    buffer.compute_advantages(last_value=4.0, gamma=0.5, gae_lambda=1.0)
    assert buffer.advantages[1] == pytest.approx(1.0 + 0.5 * 4.0)
    assert buffer.advantages[0] == pytest.approx(0.5 * buffer.advantages[1])


def test_advantages_do_not_cross_episode_boundaries():
    buffer = RolloutBuffer(4, observation_size=1)
    # Episode 1 = steps 0-1 (terminated), episode 2 = steps 2-3 (still running).
    fill(buffer, rewards=[0.0, 0.0, 100.0, 100.0], values=[0.0] * 4, terminated=[False, True, False, False])
    buffer.compute_advantages(last_value=0.0, gamma=0.99, gae_lambda=0.95)
    np.testing.assert_allclose(buffer.advantages[:2], [0.0, 0.0])
    assert buffer.advantages[2] > 100.0


def test_lambda_one_gives_monte_carlo_returns():
    rewards = [1.0, -2.0, 0.5, 3.0]
    gamma = 0.9
    buffer = RolloutBuffer(4, observation_size=1)
    fill(buffer, rewards=rewards, values=[0.3, -1.0, 2.0, 0.7], terminated=[False, False, False, True])
    buffer.compute_advantages(last_value=0.0, gamma=gamma, gae_lambda=1.0)
    discounted = [sum(gamma**k * r for k, r in enumerate(rewards[t:])) for t in range(4)]
    np.testing.assert_allclose(buffer.returns[:4], discounted, rtol=1e-6)


def test_lambda_zero_gives_one_step_td_errors():
    buffer = RolloutBuffer(3, observation_size=1)
    fill(buffer, rewards=[1.0, 1.0, 1.0], values=[0.2, 0.4, 0.6])
    buffer.compute_advantages(last_value=0.8, gamma=0.9, gae_lambda=0.0)
    expected = [1.0 + 0.9 * 0.4 - 0.2, 1.0 + 0.9 * 0.6 - 0.4, 1.0 + 0.9 * 0.8 - 0.6]
    np.testing.assert_allclose(buffer.advantages[:3], expected, rtol=1e-6)


def test_buffer_capacity_and_reset():
    buffer = RolloutBuffer(2, observation_size=1)
    fill(buffer, rewards=[0.0, 0.0], values=[0.0, 0.0])
    assert buffer.is_full()
    with pytest.raises(RuntimeError):
        fill(buffer, rewards=[0.0], values=[0.0])
    buffer.reset()
    assert buffer.size == 0 and not buffer.is_full()
    assert len(buffer.data()["actions"]) == 0
