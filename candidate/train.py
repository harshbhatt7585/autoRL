from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

TASK_MAX_STEPS = 20
HIDDEN_DIM = 128
TRAINING_EPOCHS = 2
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.01


class TinyGridPolicy(nn.Module):
    def __init__(self, obs_space: Any, action_space: Any) -> None:
        super().__init__()
        input_dim = math.prod(obs_space.shape)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(HIDDEN_DIM, action_space.n)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        encoded = self.encoder(obs)
        return self.action_head(encoded), self.value_head(encoded)


def build_policy(obs_space: Any, action_space: Any) -> nn.Module:
    return TinyGridPolicy(obs_space, action_space)


def training_overrides(*, num_envs: int, max_steps: int, device: str) -> dict[str, Any]:
    del device, max_steps
    resolved_max_steps = TASK_MAX_STEPS
    return {
        "max_steps": resolved_max_steps,
        "training_epochs": TRAINING_EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": max(32, num_envs * 4),
        "buffer_size": max(num_envs * resolved_max_steps, 256),
        "entropy_coef": ENTROPY_COEF,
        "normalize_advantages": True,
        "torch_fastpath": True,
    }
