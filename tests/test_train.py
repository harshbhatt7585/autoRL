from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from candidate.train import TASK_MAX_STEPS
from framework import _resolve_candidate_max_steps, evaluate_candidate
from train import RESULTS_HEADER, _append_result_row


class TrainHarnessTests(unittest.TestCase):
    def test_candidate_evaluates_with_small_budget(self) -> None:
        result = evaluate_candidate(
            train_episodes=2,
            eval_episodes=1,
            seed_count=1,
            num_envs=4,
            device="cpu",
        )
        self.assertAlmostEqual(result.score, result.mean_eval_return)
        self.assertEqual(result.num_actions, 4)
        self.assertEqual(result.observation_shape, (1, 5, 5))
        self.assertEqual(result.max_steps, TASK_MAX_STEPS)
        self.assertIn("empty", result.env_description.lower())

    def test_budget_caps_are_enforced(self) -> None:
        with self.assertRaisesRegex(ValueError, "at most 1000"):
            evaluate_candidate(
                train_episodes=1001,
                eval_episodes=1,
                seed_count=1,
                num_envs=4,
                device="cpu",
            )

        with self.assertRaisesRegex(ValueError, "at most 100"):
            evaluate_candidate(
                train_episodes=2,
                eval_episodes=101,
                seed_count=1,
                num_envs=4,
                device="cpu",
            )

    def test_candidate_can_define_task_horizon(self) -> None:
        with patch("framework.training_overrides", return_value={"max_steps": 100}):
            self.assertEqual(_resolve_candidate_max_steps(None, num_envs=4, device="cpu"), 100)

    def test_results_tsv_row_is_appended(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_path = Path(tmpdir) / "results.tsv"
            with patch("train._resolve_git_commit", return_value="abc123"):
                _append_result_row(
                    path=results_path,
                    train_episodes=12,
                    eval_episodes=8,
                    max_steps=100,
                    score=1.25,
                    solve_rate=0.5,
                    eval_return=1.25,
                    status="pending",
                    description="trading baseline",
                )

            self.assertEqual(
                results_path.read_text(encoding="utf-8"),
                (
                    f"{RESULTS_HEADER}\n"
                    "abc123\t12\t8\t100\t1.250000\t0.500000\t1.250000\tpending\ttrading baseline\n"
                ),
            )


if __name__ == "__main__":
    unittest.main()
