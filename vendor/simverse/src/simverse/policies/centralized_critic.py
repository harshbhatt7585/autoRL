from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import functional as F


class CentralizedCritic(nn.Module):
    """State-value critic that consumes global observations."""

    def __init__(self, obs_space) -> None:
        super().__init__()
        channels, height, width = obs_space.shape
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, channels, height, width)
            conv_out_dim = self.obs_encoder(dummy).shape[1]

        self.fc1 = nn.Linear(conv_out_dim, 128)
        self.value_head = nn.Linear(128, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.fc1.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        x = self.obs_encoder(obs)
        x = F.relu(self.fc1(x))
        return self.value_head(x)
