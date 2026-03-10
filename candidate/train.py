from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

TASK_MAX_STEPS = 100
TEMPORAL_CHANNELS = 32
TEMPORAL_HIDDEN = 48
SCALAR_HIDDEN = 32
SUMMARY_HIDDEN = 32
HIDDEN_DIM = 128
TRAINING_EPOCHS = 8
LEARNING_RATE = 3e-4
ENTROPY_COEF = 0.001
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPSILON = 0.12
MAX_GRAD_NORM = 0.8


class TradingPolicy(nn.Module):
    def __init__(self, obs_space: Any, action_space: Any) -> None:
        super().__init__()
        channels, height, width = obs_space.shape
        history_width = height * width
        scalar_features = channels - 2

        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(2, TEMPORAL_CHANNELS, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.Conv1d(TEMPORAL_CHANNELS, TEMPORAL_HIDDEN, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.history_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(TEMPORAL_HIDDEN * history_width, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_features, SCALAR_HIDDEN),
            nn.Tanh(),
            nn.Linear(SCALAR_HIDDEN, SCALAR_HIDDEN),
            nn.Tanh(),
        )
        self.summary_encoder = nn.Sequential(
            nn.Linear(8, SUMMARY_HIDDEN),
            nn.Tanh(),
            nn.Linear(SUMMARY_HIDDEN, SUMMARY_HIDDEN),
            nn.Tanh(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(HIDDEN_DIM + SCALAR_HIDDEN + SUMMARY_HIDDEN, HIDDEN_DIM),
            nn.Tanh(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(HIDDEN_DIM, action_space.n)
        self.value_head = nn.Linear(HIDDEN_DIM, 1)
        nn.init.zeros_(self.action_head.bias)
        self.action_head.bias.data[2] = 0.1

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)

        temporal = obs[:, :2].reshape(obs.shape[0], 2, -1)
        price_history = temporal[:, 0]
        return_history = temporal[:, 1]
        scalars = obs[:, 2:, 0, 0]
        summary = torch.stack(
            (
                price_history[:, -4:].mean(dim=-1),
                price_history[:, -12:].mean(dim=-1),
                price_history.mean(dim=-1),
                return_history[:, -4:].mean(dim=-1),
                return_history[:, -12:].mean(dim=-1),
                return_history[:, -12:].abs().mean(dim=-1),
                price_history.max(dim=-1).values,
                price_history.min(dim=-1).values,
            ),
            dim=-1,
        )
        encoded_history = self.history_head(self.temporal_encoder(temporal))
        encoded_scalars = self.scalar_encoder(scalars)
        encoded_summary = self.summary_encoder(summary)
        encoded = self.trunk(torch.cat((encoded_history, encoded_scalars, encoded_summary), dim=-1))
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
        "batch_size": max(256, num_envs * 8),
        "buffer_size": max(num_envs * resolved_max_steps, 2048),
        "entropy_coef": ENTROPY_COEF,
        "gamma": GAMMA,
        "gae_lambda": GAE_LAMBDA,
        "clip_epsilon": CLIP_EPSILON,
        "max_grad_norm": MAX_GRAD_NORM,
        "normalize_advantages": True,
        "torch_fastpath": True,
    }
