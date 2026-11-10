"""Run a non-learning controller, for comparison with PPO.

Controllers:
    traffic_signal   fixed_time  switch after a fixed green time (--fixed-green)
                     random      keep or switch at random
    vehicle_control  sumo        SUMO's own driver model drives the vehicle
                     random      decelerate / maintain / accelerate at random

Examples:
    python scripts/run_baseline.py --env traffic_signal --controller fixed_time
    python scripts/run_baseline.py --env traffic_signal --controller random --episodes 5
    python scripts/run_baseline.py --env vehicle_control --controller sumo --gui --episodes 1

All controllers act through the same environment as PPO, so signal changes
still pass through the minimum-green, yellow and all-red rules, and the same
reward and metrics are recorded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import print_summary, write_csv  # also puts the project root on sys.path

from envs import make_env
from envs.traffic_signal_env import KEEP_PHASE, SWITCH_PHASE
from envs.vehicle_control_env import MAINTAIN
from ppo.rollout import evaluate_policy

CONTROLLERS = {
    "traffic_signal": ["fixed_time", "random"],
    "vehicle_control": ["sumo", "random"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a baseline controller.")
    parser.add_argument("--env", choices=list(CONTROLLERS), required=True)
    parser.add_argument("--controller", required=True,
                        help="fixed_time or random (traffic_signal); sumo or random (vehicle_control)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=10_000, help="seed of the first episode")
    parser.add_argument("--fixed-green", type=float, default=30.0,
                        help="green time in seconds for the fixed_time controller")
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--gui", action="store_true", help="show the SUMO window")
    args = parser.parse_args()
    if args.controller not in CONTROLLERS[args.env]:
        parser.error(f"controller for {args.env} must be one of {CONTROLLERS[args.env]}")
    return args


def make_controller(args, env):
    """Return a policy function (observation, info) -> action."""
    if args.controller == "random":
        rng = np.random.default_rng(args.seed)

        def random_policy(observation, info):
            return int(rng.integers(env.action_space.n))
        return random_policy

    if args.controller == "fixed_time":
        def fixed_time_policy(observation, info):
            # Request a switch once the green has lasted --fixed-green seconds.
            # Decisions happen every 5 s, so a multiple of 5 is exact.
            return SWITCH_PHASE if info["time_in_green"] >= args.fixed_green else KEEP_PHASE
        return fixed_time_policy

    def sumo_policy(observation, info):
        return MAINTAIN  # ignored: the environment lets SUMO drive the vehicle
    return sumo_policy


def main() -> None:
    args = parse_args()
    if args.controller == "sumo":
        env = make_env(args.env, gui=args.gui, sumo_drives_vehicle=True)
    else:
        env = make_env(args.env, gui=args.gui)
    policy = make_controller(args, env)

    try:
        records = evaluate_policy(env, policy, args.episodes, args.seed)
    finally:
        env.close()

    output = Path(args.output_dir) / f"{args.env}_{args.controller}_seed{args.seed}.csv"
    write_csv(output, records)
    print_summary(f"{args.controller} controller on {args.env}, seeds {args.seed}-{args.seed + args.episodes - 1}",
                  records)
    print(f"\nResults written to {output}")


if __name__ == "__main__":
    main()
