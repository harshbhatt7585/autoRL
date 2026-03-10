from __future__ import annotations

import unittest
from unittest.mock import patch

from candidate.train import TASK_MAX_STEPS
from framework import _resolve_candidate_max_steps, evaluate_candidate


class TrainHarnessTests(unittest.TestCase):
    def test_candidate_evaluates_with_small_budget(self) -> None:
        result = evaluate_candidate(
            train_episodes=2,
            eval_episodes=1,
            seed_count=1,
            num_envs=4,
            device="cpu",
        )
        self.assertGreaterEqual(result.score, 0.0)
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


if __name__ == "__main__":
    unittest.main()
