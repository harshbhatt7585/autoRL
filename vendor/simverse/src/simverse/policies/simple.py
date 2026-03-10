import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[2]  # path/to/src
    sys.path.insert(0, str(_src))


import torch
import torch.nn as nn


class SimplePolicy(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
    ):
        super().__init__()

        channels, _, _ = obs_space.shape
        self.obs_encoder = nn.Sequential(
            # Aggressive downsampling keeps compute stable at larger spatial sizes (e.g., 64x64).
            nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(96, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(128, 96),
            nn.SiLU(),
        )

        # action head
        self.action_head = nn.Linear(96, action_space.n)

        # value head
        self.value_head = nn.Linear(96, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        x = self.obs_encoder(obs)
        x = self.trunk(x)

        logits = self.action_head(x)

        value = self.value_head(x)

        return logits, value


if __name__ == "__main__":
    import gymnasium as gym

    obs_space = gym.spaces.Box(0, 1, shape=(3, 30, 20))
    action_space = gym.spaces.Discrete(6)
    policy = SimplePolicy(obs_space, action_space)
    obs = torch.randn(1, *obs_space.shape)
    logits, value = policy(obs)
    print(logits)
    print(value)
