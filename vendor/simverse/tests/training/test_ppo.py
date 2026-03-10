from __future__ import annotations

import math

import pytest
import torch
from simverse.training.ppo import PPOTrainer
from simverse.training.utils import build_ppo_training_config


def make_trainer() -> PPOTrainer:
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return PPOTrainer(
        optimizer=optimizer,
        episodes=1,
        training_epochs=1,
        device="cpu",
        batch_size=4,
        buffer_size=16,
        use_wandb=False,
    )


def test_vectorized_gae_bootstraps_last_unfinished_state() -> None:
    trainer = make_trainer()

    advantages = trainer._compute_vectorized_gae(
        rewards=torch.tensor([1.0, 1.0]),
        values=torch.tensor([0.5, 0.25]),
        dones=torch.tensor([False, False]),
        valid=torch.tensor([True, True]),
        final_values=torch.tensor([2.0]),
        env_count=1,
    )

    last_delta = 1.0 + trainer.gamma * 2.0 - 0.25
    first_delta = 1.0 + trainer.gamma * 0.25 - 0.5
    expected_first = first_delta + trainer.gamma * trainer.gae_lambda * last_delta

    assert advantages.shape == (2,)
    assert advantages[1].item() == pytest.approx(last_delta, rel=1e-6)
    assert advantages[0].item() == pytest.approx(expected_first, rel=1e-6)


def test_vectorized_gae_ignores_invalid_post_terminal_slots() -> None:
    trainer = make_trainer()

    advantages = trainer._compute_vectorized_gae(
        rewards=torch.tensor([1.0, 10.0, 2.0, 0.0]),
        values=torch.tensor([0.0, 0.0, 0.0, 5.0]),
        dones=torch.tensor([False, True, True, True]),
        valid=torch.tensor([True, True, True, False]),
        final_values=torch.tensor([0.0, 0.0]),
        env_count=2,
    ).reshape(2, 2)

    assert advantages[0, 1].item() == pytest.approx(10.0, rel=1e-6)
    assert advantages[1, 1].item() == pytest.approx(0.0, rel=1e-6)
    assert math.isfinite(advantages[0, 0].item())
    assert math.isfinite(advantages[1, 0].item())


def test_build_ppo_training_config_enforces_rollout_window_floor() -> None:
    config = build_ppo_training_config(
        num_agents=2,
        num_envs=4,
        max_steps=200,
        episodes=1,
        training_epochs=1,
        lr=3e-4,
        batch_size=8,
        buffer_size=64,
        device="cpu",
        dtype=torch.float32,
    )

    assert config["buffer_size"] == 4 * 2 * 64
