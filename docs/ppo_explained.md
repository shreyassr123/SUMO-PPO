# PPO in this repository

This note follows one PPO iteration through the code and connects each step
to its equation. It assumes the basic ideas from the README (policy, value,
return, advantage).

Files: `ppo/networks.py`, `ppo/buffer.py`, `ppo/agent.py`, `ppo/rollout.py`.

## 1. Actor and critic

`Actor` (in `networks.py`) maps an observation vector *s* to one *logit* per
action through two hidden layers of 64 tanh units. The policy is the softmax of
the logits:

π_θ(a | s) = exp(z_a) / Σ_b exp(z_b),

represented by a PyTorch `Categorical` distribution (`Actor.distribution`).
The output layer is initialised with very small weights (orthogonal, gain
0.01), so all logits start near zero and the initial policy is close to
uniform: every action is tried before anything has been learned.

`Critic` maps the same observation to a single number through two hidden
layers of 128 tanh units. It estimates the state value V_φ(s). The two
networks share no parameters and have separate Adam optimisers.

**Value normalisation.** Returns can be of order 1 (CartPole) or of order
−1000 (queued vehicle-seconds). The critic therefore predicts *normalised*
values, and `PPOAgent` keeps a running mean μ and standard deviation σ of all
return targets seen so far (`RunningMeanStd`). Everywhere outside the
critic's loss, values are converted back to reward units: V(s) = σ·critic(s) + μ
(`_value_in_reward_units`).

## 2. Choosing actions: sampling and greedy

`PPOAgent.act(observation, deterministic=False)`:

* during training, samples a ~ π_θ(·|s), so that actions with lower
  probability are still tried (exploration);
* with `deterministic=True`, returns argmax_a π_θ(a|s), the usual choice for
  evaluation;
* in both cases also returns log π_θ(a|s) and V(s), which the rollout buffer
  stores. Everything runs under `torch.no_grad()`, because no gradients are
  needed while acting.

## 3. Rollout collection

`RolloutCollector.collect_rollout()` (in `rollout.py`) repeats

1. `action, log_prob, value = agent.act(s)`
2. `s', r, terminated, truncated, info = env.step(action)`
3. `buffer.add(s, action, log_prob, value, r, terminated, truncated, final_value)`

until the buffer holds `rollout_steps` transitions (2048 by default). When an
episode ends, the environment is reset and collection continues, so a rollout
usually contains several episodes and may end in the middle of one. The
collector keeps the current observation between rollouts.

The stored `log_prob` is log π_old(aₜ|sₜ), the probability under the policy
that actually chose the action (the *behaviour policy*). It is fixed for the
rest of the iteration and is what the probability ratio compares against.

**Two kinds of episode end.** Gymnasium separates them, and the difference
matters for the value of the next state:

* `terminated`: the task really ended (the car arrived). Nothing follows, so
  V(sₜ₊₁) = 0.
* `truncated`: a time limit stopped the episode, but the state sₜ₊₁ still has
  a future. Treating it as terminal would teach the critic that states near
  the time limit are worth nothing. The collector therefore asks the critic
  for V(sₜ₊₁) of the final observation and stores it as `final_value`, and
  GAE uses it as the bootstrap value.

## 4. Advantages: the GAE recursion

`RolloutBuffer.compute_advantages(last_value, gamma, gae_lambda)` works
backwards through the buffer:

δₜ = rₜ + γ · V_next − V(sₜ)

Âₜ = δₜ + γλ · cₜ · Âₜ₊₁

where

| situation at step *t* | V_next | cₜ |
|---|---|---|
| episode continues | V(sₜ₊₁), the next stored value | 1 |
| last stored step, episode continues | `last_value` = V(s) of the current observation | 1 |
| `truncated` | `final_values[t]` | 0 |
| `terminated` | 0 | 0 |

cₜ = 0 stops the recursion at an episode boundary, so advantages never mix
two episodes. Expanding the recursion gives the familiar form

Âₜ = Σ_{l≥0} (γλ)^l δₜ₊ₗ  (up to the end of the episode or rollout).

With λ = 0, Âₜ = δₜ (one-step TD error: low variance, more bias from the
critic); with λ = 1, Âₜ = Gₜ − V(sₜ) (Monte Carlo: unbiased, high
variance). The default λ = 0.95 lies in between. Both limits are checked in
`tests/test_buffer.py`.

The critic's regression targets are the **λ-returns** Gₜ = Âₜ + V(sₜ).

## 5. The PPO update

`PPOAgent.update_policy(buffer)` does the following.

**Targets and advantages.** The return statistics are updated with the new
targets Gₜ, which are then normalised: Ĝₜ = (Gₜ − μ)/σ. The advantages are
standardised over the whole rollout: Âₜ ← (Âₜ − mean)/(std + 10⁻⁸). This keeps
the size of the policy step independent of the reward scale.

**Epochs and minibatches.** The rollout is used for `update_epochs` (10)
passes. Each pass shuffles the indices (`torch.randperm`) and splits them into
minibatches of `minibatch_size` (256). For every minibatch:

1. **Critic step.** L_V = mean[(critic(sₜ) − Ĝₜ)²], one Adam step on the
   critic.

2. **Probability ratio.**
   rₜ(θ) = π_θ(aₜ|sₜ) / π_old(aₜ|sₜ) = exp(log π_θ(aₜ|sₜ) − log π_old(aₜ|sₜ)),
   computed as `ratio = torch.exp(new_log_probs - old_log_probs)`. On the very
   first minibatch rₜ = 1 exactly, because the policy has not changed yet.

3. **KL early stop.** The approximate KL divergence between the behaviour
   policy and the current policy is estimated as
   mean[(rₜ − 1) − log rₜ]. If it exceeds `target_kl` (0.02), the actor is not
   updated any more on this rollout; the critic continues. This guards
   against one rollout pushing the policy too far.

4. **Clipped surrogate objective.**
   L^CLIP(θ) = mean[ min( rₜ Âₜ, clip(rₜ, 1−ε, 1+ε) Âₜ ) ],  ε = 0.2.
   For a positive advantage, increasing the action's probability helps only
   until rₜ = 1 + ε; for a negative advantage, decreasing it helps only until
   rₜ = 1 − ε. Beyond those points the gradient is zero, so a single rollout
   cannot move the policy far. In code, `policy_loss = -torch.min(unclipped,
   clipped).mean()`; the minus sign turns maximisation into minimisation.

5. **Entropy bonus.** H = mean[−Σ_a π(a|s) log π(a|s)] is high when the policy
   is uncertain. The actor minimises `policy_loss − entropy_coef · H`
   (entropy_coef = 0.01), which discourages the policy from becoming
   deterministic too early.

6. **Gradient clipping.** Before each optimiser step the gradient norm is
   clipped to `max_grad_norm` (0.5) for both networks (`_optimise`).

**The usual combined loss.** Many texts write one loss,
L = −L^CLIP + c₁ L_V − c₂ H, for an actor and critic that share parameters.
Here the networks and optimisers are separate, so the value coefficient c₁
would only rescale the critic's learning rate; the critic's learning rate
(`critic_lr`) plays that role directly.

**Diagnostics.** `update_policy` returns the mean policy loss, value loss,
entropy, approximate KL, *clip fraction* (share of samples with
|rₜ − 1| > ε) and the critic's *explained variance* before the update. They
are written to `updates.csv` by `train.py`.

After the update the buffer is cleared: PPO is *on-policy*, so data from an
older policy is not reused in later iterations.

## 6. Hyperparameters

| Name (`PPOConfig`) | Default | Effect of increasing it |
|---|---|---|
| `gamma` | 0.99 | longer planning horizon (≈ 1/(1−γ) = 100 decisions); slower, noisier credit assignment |
| `gae_lambda` | 0.95 | less bias, more variance in advantages |
| `clip_range` | 0.2 | larger policy changes per update |
| `actor_lr` | 3e-4 | faster but less stable policy learning |
| `critic_lr` | 1e-3 | faster value fitting; too high can make values oscillate |
| `rollout_steps` | 2048 | less noisy updates, fewer updates per episode |
| `update_epochs` | 10 | more reuse of each rollout; risk of overfitting it (the KL stop limits this) |
| `minibatch_size` | 256 | smoother but fewer gradient steps |
| `entropy_coef` | 0.01 | more exploration; too high prevents a decided policy |
| `max_grad_norm` | 0.5 | allows larger individual gradient steps |
| `target_kl` | 0.02 | allows more policy movement per rollout (`None` disables the stop) |

With a decision interval of 5 s (signal example), γ = 0.99 corresponds to
looking about 500 simulated seconds ahead; with 1 s decisions (vehicle
example), about 100 s.

## 7. Equation-to-code summary

| Concept | Equation | Code |
|---|---|---|
| policy | π_θ(a\|s) = softmax(z)_a | `Actor.distribution` |
| value | V_φ(s) = σ·critic(s) + μ | `PPOAgent.predict_value`, `_value_in_reward_units` |
| TD error | δₜ = rₜ + γV_next − V(sₜ) | `RolloutBuffer.compute_advantages` |
| GAE | Âₜ = δₜ + γλcₜÂₜ₊₁ | `RolloutBuffer.compute_advantages` |
| critic target | Gₜ = Âₜ + V(sₜ) | `RolloutBuffer.compute_advantages` |
| ratio | rₜ = exp(log π_θ − log π_old) | `PPOAgent.update_policy` |
| clipped objective | min(rÂ, clip(r)Â) | `PPOAgent.update_policy` |
| value loss | (critic(s) − Ĝ)² | `PPOAgent.update_policy` |
| entropy | −Σ π log π | `dist.entropy()` in `update_policy` |
| approx. KL | (r − 1) − log r | `PPOAgent.update_policy` |

## 8. Differences from the original research code

The original learner processed sequences of decisions made by individual
travellers, each chain ending in one arrival-time reward. Every chain end was
treated as terminal, decisions carried a per-decision discount and shaping
term, and some actions were forced or masked. Those features were specific
to the journey-recommendation problem and have been removed. The learner here
works on ordinary time-ordered episodes, distinguishes termination from
truncation, and bootstraps after time limits. The network sizes, optimiser
settings, value normalisation, advantage normalisation, KL early stop and
gradient clipping follow the original.
