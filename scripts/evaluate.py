"""Evaluate a trained PPO checkpoint on fresh traffic.

Examples:
    python scripts/evaluate.py --checkpoint results/<run>/checkpoints/final.pt
    python scripts/evaluate.py --checkpoint results/<run>/checkpoints/final.pt --episodes 20 --seed 20000
    python scripts/evaluate.py --checkpoint results/<run>/checkpoints/final.pt --gui --episodes 1

The environment is read from the checkpoint. Episode i uses seed --seed + i,
so running scripts/run_baseline.py with the same --seed evaluates the
baselines on exactly the same traffic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from common import print_summary, write_csv  # also puts the project root on sys.path

from envs import ENVIRONMENT_NAMES, make_env
from ppo.agent import PPOAgent
from ppo.rollout import agent_policy, evaluate_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="path to a .pt file written by train.py")
    parser.add_argument("--env", choices=ENVIRONMENT_NAMES,
                        help="environment name (default: the one stored in the checkpoint)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000, help="seed of the first evaluation episode")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions from the policy instead of taking the most probable one")
    parser.add_argument("--output", help="CSV file for the results (default: next to the checkpoint's run folder)")
    parser.add_argument("--gui", action="store_true", help="show the SUMO window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent, metadata = PPOAgent.load_checkpoint(args.checkpoint)

    env_name = args.env or metadata.get("env")
    if env_name is None:
        raise SystemExit("The checkpoint does not record its environment; pass --env.")
    if metadata.get("env") not in (None, env_name):
        raise SystemExit(f"The checkpoint was trained on '{metadata['env']}', not '{env_name}'.")

    env = make_env(env_name, gui=args.gui)
    try:
        if (env.observation_space.shape[0], env.action_space.n) != (agent.observation_size, agent.n_actions):
            raise SystemExit(
                f"Checkpoint expects {agent.observation_size} observations and {agent.n_actions} actions, "
                f"but '{env_name}' now has {env.observation_space.shape[0]} and {env.action_space.n}. "
                "The environment was changed after training; retrain or restore the old definition."
            )
        policy = agent_policy(agent, deterministic=not args.stochastic)
        records = evaluate_policy(env, policy, args.episodes, args.seed)
    finally:
        env.close()

    mode = "stochastic" if args.stochastic else "greedy"
    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else (
        checkpoint.parent.parent / f"evaluation_{checkpoint.stem}_{mode}_seed{args.seed}.csv")
    write_csv(output, records)
    print_summary(f"PPO ({mode}) on {env_name}, seeds {args.seed}-{args.seed + args.episodes - 1}", records)
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
