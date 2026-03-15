import sys
from pathlib import Path

if __name__ == "__main__" and __package__ is None:
    _src = Path(__file__).resolve().parents[2]  # path/to/src
    sys.path.insert(0, str(_src))


import numpy as np
import torch
import torch.nn as nn


class SimplePolicy(nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
    ):
        super().__init__()

        if hasattr(obs_space, "spaces"):
            image_space = obs_space["obs"]
            feature_space = obs_space.spaces.get("features")
        else:
            image_space = obs_space
            feature_space = None

        channels, _, _ = image_space.shape
        self.feature_dim = int(np.prod(feature_space.shape)) if feature_space is not None else 0
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

        head_input_dim = 96
        if self.feature_dim > 0:
            self.feature_encoder = nn.Sequential(
                nn.Linear(self.feature_dim, 32),
                nn.SiLU(),
            )
            head_input_dim += 32

        # action head
        self.action_head = nn.Linear(head_input_dim, action_space.n)

        # value head
        self.value_head = nn.Linear(head_input_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
        target = self.action_head.weight
        if obs.device != target.device or obs.dtype != target.dtype:
            obs = obs.to(device=target.device, dtype=target.dtype)
        x = self.obs_encoder(obs)
        x = self.trunk(x)

        if self.feature_dim > 0:
            if features is None:
                features = torch.zeros(
                    (obs.shape[0], self.feature_dim),
                    device=target.device,
                    dtype=target.dtype,
                )
            elif features.dim() == 1:
                features = features.unsqueeze(0)
            if features.device != target.device or features.dtype != target.dtype:
                features = features.to(device=target.device, dtype=target.dtype)
            features = features.reshape(features.shape[0], self.feature_dim)
            feature_x = self.feature_encoder(features)
            x = torch.cat((x, feature_x), dim=-1)

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
