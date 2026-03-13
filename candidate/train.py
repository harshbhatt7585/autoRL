from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

TASK_MAX_STEPS = 100
CONV_WIDTH = 96
MLP_WIDTH = 256
TRAINING_EPOCHS = 6
LEARNING_RATE = 1.8e-4
ENTROPY_COEF = 0.0012
CLIP_EPSILON = 0.16
GAMMA = 0.995
GAE_LAMBDA = 0.97


class TradingPolicy(nn.Module):
    def __init__(self, obs_space: Any, action_space: Any) -> None:
        super().__init__()
        channels = int(obs_space.shape[0])
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, CONV_WIDTH, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(CONV_WIDTH, CONV_WIDTH, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(CONV_WIDTH, CONV_WIDTH, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(CONV_WIDTH * obs_space.shape[1] * obs_space.shape[2], MLP_WIDTH),
            nn.SiLU(),
        )
        self.action_head = nn.Linear(MLP_WIDTH, action_space.n)
        self.value_head = nn.Linear(MLP_WIDTH, 1)

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
        "batch_size": max(256, num_envs * 10),
        "buffer_size": max(num_envs * resolved_max_steps, 4096),
        "entropy_coef": ENTROPY_COEF,
        "clip_epsilon": CLIP_EPSILON,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "normalize_advantages": True,
        "torch_fastpath": True,
    }
