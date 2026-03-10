from __future__ import annotations

import argparse

from framework import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_NUM_ENVS,
    DEFAULT_TRAIN_EPISODES,
    MAX_EVAL_EPISODES,
    MAX_TRAIN_EPISODES,
    evaluate_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the current Simverse-backed autoRL candidate.")
    parser.add_argument(
        "--train-episodes",
        type=int,
        default=DEFAULT_TRAIN_EPISODES,
        help=f"PPO training episodes to run (2-{MAX_TRAIN_EPISODES}, default: {DEFAULT_TRAIN_EPISODES})",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Greedy evaluation episodes to run (1-{MAX_EVAL_EPISODES}, default: {DEFAULT_EVAL_EPISODES})",
    )
    parser.add_argument("--seed-count", type=int, default=2)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = evaluate_candidate(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        seed_count=args.seed_count,
        num_envs=args.num_envs,
        device=args.device,
    )

    print("---")
    print(f"env_name: {result.env_name}")
    print(f"env_description: {result.env_description}")
    print(f"score: {result.score:.6f}")
    print(f"mean_eval_return: {result.mean_eval_return:.6f}")
    print(f"mean_solve_rate: {result.mean_solve_rate:.6f}")
    print(f"mean_train_return: {result.mean_train_return:.6f}")
    print(f"learning_gain: {result.learning_gain:.6f}")
    print(f"headroom_bonus: {result.headroom_bonus:.6f}")
    print(f"stability: {result.stability:.6f}")
    print(f"complexity_penalty: {result.complexity_penalty:.6f}")
    print(f"mean_episode_length: {result.mean_episode_length:.6f}")
    print(f"device: {result.device}")
    print(f"observation_shape: {result.observation_shape}")
    print(f"num_actions: {result.num_actions}")
    print(f"max_steps: {result.max_steps}")
    print(f"num_envs: {result.num_envs}")
    print(f"train_episodes: {result.train_episodes}")
    print(f"eval_episodes: {result.eval_episodes}")
    print(f"seed_count: {result.seed_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
