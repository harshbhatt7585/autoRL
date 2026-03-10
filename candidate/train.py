from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn


class TinyGridPolicy(nn.Module):
    def __init__(self, obs_space: Any, action_space: Any) -> None:
        super().__init__()
        input_dim = math.prod(obs_space.shape)
        hidden_dim = 128
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, action_space.n)
        self.value_head = nn.Linear(hidden_dim, 1)

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
    del device
    return {
        "training_epochs": 2,
        "lr": 3e-4,
        "batch_size": max(32, num_envs * 4),
        "buffer_size": max(num_envs * max_steps, 256),
        "entropy_coef": 0.01,
        "normalize_advantages": True,
        "torch_fastpath": True,
    }
