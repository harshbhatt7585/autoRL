from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import traceback

from framework import (
    DEFAULT_EVAL_EPISODES,
    DEFAULT_NUM_ENVS,
    DEFAULT_TRAIN_EPISODES,
    MAX_EVAL_EPISODES,
    MAX_TRAIN_EPISODES,
    _resolve_candidate_max_steps,
    evaluate_candidate,
)

RESULTS_HEADER = (
    "commit\ttrain_episodes\teval_episodes\tmax_steps\t"
    "score\tsolve_rate\teval_return\tstatus\tdescription"
)
LEGACY_RESULTS_HEADER = "commit\tscore\tsolve_rate\teval_return\tstatus\tdescription"


def _normalize_description(value: str) -> str:
    return " ".join(value.replace("\t", " ").split())


def _resolve_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return "nogit"
    commit = completed.stdout.strip()
    return commit or "nogit"


def _ensure_results_tsv(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(f"{RESULTS_HEADER}\n", encoding="utf-8")
        return

    contents = path.read_text(encoding="utf-8")
    lines = contents.splitlines()
    if not lines:
        path.write_text(f"{RESULTS_HEADER}\n", encoding="utf-8")
        return

    header = lines[0]
    if header == RESULTS_HEADER:
        return
    if header == LEGACY_RESULTS_HEADER and len(lines) == 1:
        path.write_text(f"{RESULTS_HEADER}\n", encoding="utf-8")
        return

    raise RuntimeError(
        f"{path} has an unexpected header. Expected:\n{RESULTS_HEADER}\nGot:\n{header}"
    )


def _append_result_row(
    *,
    path: Path,
    train_episodes: int,
    eval_episodes: int,
    max_steps: int,
    score: float,
    solve_rate: float,
    eval_return: float,
    status: str,
    description: str,
) -> None:
    _ensure_results_tsv(path)
    description_text = _normalize_description(description)
    row = "\t".join(
        (
            _resolve_git_commit(),
            str(train_episodes),
            str(eval_episodes),
            str(max_steps),
            f"{score:.6f}",
            f"{solve_rate:.6f}",
            f"{eval_return:.6f}",
            status,
            description_text,
        )
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{row}\n")


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
    parser.add_argument(
        "--status",
        choices=("pending", "keep", "discard"),
        default="pending",
        help="Status label to write into results.tsv for successful runs.",
    )
    parser.add_argument(
        "--description",
        default="",
        help="Short free-text experiment description to write into results.tsv.",
    )
    parser.add_argument(
        "--results-path",
        default="results.tsv",
        help="Path to the TSV ledger where run summaries are appended.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results_path = Path(args.results_path)

    try:
        result = evaluate_candidate(
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            seed_count=args.seed_count,
            num_envs=args.num_envs,
            device=args.device,
        )
    except Exception as exc:
        max_steps = 0
        try:
            max_steps = _resolve_candidate_max_steps(
                None,
                num_envs=args.num_envs,
                device=args.device,
            )
        except Exception:
            max_steps = 0

        _append_result_row(
            path=results_path,
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=max_steps,
            score=0.0,
            solve_rate=0.0,
            eval_return=0.0,
            status="crash",
            description=args.description or f"crash: {exc}",
        )
        traceback.print_exc()
        return 1

    _append_result_row(
        path=results_path,
        train_episodes=result.train_episodes,
        eval_episodes=result.eval_episodes,
        max_steps=result.max_steps,
        score=result.score,
        solve_rate=result.mean_solve_rate,
        eval_return=result.mean_eval_return,
        status=args.status,
        description=args.description,
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
