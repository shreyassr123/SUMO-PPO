"""Train a PPO agent on one of the example environments.

Examples:
    python scripts/train.py --env cartpole --episodes 300
    python scripts/train.py --env traffic_signal --episodes 200
    python scripts/train.py --env vehicle_control --episodes 300 --seed 1

Each run writes to its own folder, by default results/<env>_seed<seed>_<time>/:
    config.json            settings of the run
    training_episodes.csv  return, length and environment metrics per episode
    updates.csv            PPO losses and diagnostics per update
    training_curves.png    return, critic loss and entropy
    checkpoints/           final.pt and a checkpoint every few updates
    evaluation.csv         greedy-policy results on separate evaluation seeds
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from common import plot_training, print_summary, write_csv  # also puts the project root on sys.path

from envs import ENVIRONMENT_NAMES, make_env
from ppo.agent import PPOAgent, PPOConfig, set_seed
from ppo.buffer import RolloutBuffer
from ppo.rollout import RolloutCollector, agent_policy, evaluate_policy

# Evaluation episodes use seeds 10000, 10001, ... so that their traffic is
# independent of the training episodes (whose seeds come from --seed).
EVALUATION_SEED = 10_000
CHECKPOINT_EVERY_UPDATES = 10


def parse_args() -> argparse.Namespace:
    defaults = PPOConfig()
    parser = argparse.ArgumentParser(description="Train PPO on a SUMO (or CartPole) environment.")
    parser.add_argument("--env", choices=ENVIRONMENT_NAMES, required=True)
    parser.add_argument("--episodes", type=int, default=200, help="number of training episodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=defaults.actor_lr, help="actor (policy) learning rate")
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="discount factor")
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps,
                        help="environment steps collected before each PPO update")
    parser.add_argument("--eval-episodes", type=int, default=5,
                        help="greedy evaluation episodes after training (0 to skip)")
    parser.add_argument("--output-dir", default="results", help="parent folder for run folders")
    parser.add_argument("--gui", action="store_true", help="show the SUMO window (much slower)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    run_dir = Path(args.output_dir) / f"{args.env}_seed{args.seed}_{datetime.now():%Y%m%d-%H%M%S}"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    config = PPOConfig(gamma=args.gamma, actor_lr=args.lr, critic_lr=args.critic_lr,
                       rollout_steps=args.rollout_steps)
    run_info = {"env": args.env, "seed": args.seed, "episodes": args.episodes,
                "evaluation_seed": EVALUATION_SEED, "ppo": asdict(config), "command": sys.argv}
    (run_dir / "config.json").write_text(json.dumps(run_info, indent=2))
    print(f"Writing results to {run_dir}")

    env = make_env(args.env, gui=args.gui)
    observation_size = env.observation_space.shape[0]
    agent = PPOAgent(observation_size, env.action_space.n, config)
    buffer = RolloutBuffer(config.rollout_steps, observation_size)
    metadata = {"env": args.env, "seed": args.seed}
    print(f"Environment '{args.env}': {observation_size} observation values, {env.action_space.n} actions")

    episodes: list[dict] = []
    updates: list[dict] = []
    interrupted = False
    try:
        collector = RolloutCollector(env, seed=args.seed)
        try:
            while collector.episodes_completed < args.episodes:
                # 1) Run the current policy to gather experience.
                finished = collector.collect_rollout(agent, buffer, max_episodes=args.episodes)
                # 2) Improve the policy and critic on that experience.
                stats = agent.update_policy(buffer)

                episodes.extend(finished)
                updates.append({"update": len(updates) + 1, "episodes_completed": collector.episodes_completed,
                                "steps": buffer.size, **stats})
                recent = [e["return"] for e in episodes[-10:]]
                recent_text = f"{sum(recent) / len(recent):10.2f}" if recent else "       n/a"
                print(f"update {len(updates):4d} | episodes {collector.episodes_completed:4d}/{args.episodes} "
                      f"| mean return (last 10) {recent_text} | policy loss {stats['policy_loss']:+.4f} "
                      f"| value loss {stats['value_loss']:.4f} | entropy {stats['entropy']:.3f} "
                      f"| KL {stats['approx_kl']:.4f}")
                if len(updates) % CHECKPOINT_EVERY_UPDATES == 0:
                    agent.save_checkpoint(checkpoint_dir / f"update_{len(updates):04d}.pt", metadata)
        except KeyboardInterrupt:
            interrupted = True
            print("\nTraining interrupted; saving the results collected so far (evaluation skipped).")

        agent.save_checkpoint(checkpoint_dir / "final.pt", metadata)
        write_csv(run_dir / "training_episodes.csv", episodes)
        write_csv(run_dir / "updates.csv", updates)
        if episodes and updates:
            plot_training(episodes, updates, run_dir / "training_curves.png", f"PPO on {args.env}")
        print_summary("Training episodes", episodes)

        if args.eval_episodes > 0 and not interrupted:
            records = evaluate_policy(env, agent_policy(agent, deterministic=True),
                                      args.eval_episodes, EVALUATION_SEED)
            write_csv(run_dir / "evaluation.csv", records)
            print_summary(f"Greedy evaluation on seeds {EVALUATION_SEED}+", records)
    finally:
        env.close()

    print(f"\nFinal checkpoint: {checkpoint_dir / 'final.pt'}")


if __name__ == "__main__":
    main()
