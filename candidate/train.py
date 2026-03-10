from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

TASK_MAX_STEPS = 100
HIDDEN_CHANNELS = 32
HIDDEN_DIM = 128
TRAINING_EPOCHS = 6
LEARNING_RATE = 4e-4
ENTROPY_COEF = 0.002
GAMMA = 0.99
GAE_LAMBDA = 0.97
CLIP_EPSILON = 0.15


class TradingPolicy(nn.Module):
    def __init__(self, obs_space: Any, action_space: Any) -> None:
        super().__init__()
        channels, height, width = obs_space.shape
        flattened_dim = HIDDEN_CHANNELS * height * width
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(HIDDEN_CHANNELS, HIDDEN_CHANNELS, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(flattened_dim, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(HIDDEN_DIM, action_space.n)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.action_head.bias)
        self.action_head.bias.data[2] = 0.15

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        encoded = self.encoder(obs)
        return self.action_head(encoded), self.value_head(encoded)


def build_policy(obs_space: Any, action_space: Any) -> nn.Module:
    return TradingPolicy(obs_space, action_space)


def training_overrides(*, num_envs: int, max_steps: int, device: str) -> dict[str, Any]:
    del device, max_steps
    resolved_max_steps = TASK_MAX_STEPS
    return {
        "max_steps": resolved_max_steps,
        "training_epochs": TRAINING_EPOCHS,
        "lr": LEARNING_RATE,
        "batch_size": max(128, num_envs * 4),
        "buffer_size": max(num_envs * resolved_max_steps, 1024),
        "entropy_coef": ENTROPY_COEF,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON,
        "normalize_advantages": True,
        "torch_fastpath": True,
    }
