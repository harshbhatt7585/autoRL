from __future__ import annotations

import gymnasium as gym
import torch
from simverse.policies.simple import SimplePolicy


def test_simple_policy_supports_optional_feature_vectors() -> None:
    obs_space = gym.spaces.Dict(
        {
            "obs": gym.spaces.Box(low=0.0, high=1.0, shape=(2, 13, 13)),
            "features": gym.spaces.Box(low=0.0, high=1.0, shape=(3,)),
        }
    )
    action_space = gym.spaces.Discrete(6)
    policy = SimplePolicy(obs_space=obs_space, action_space=action_space)

    obs = torch.randn(4, 2, 13, 13)
    features = torch.randn(4, 3)
    logits, value = policy(obs, features)

    assert tuple(logits.shape) == (4, 6)
    assert tuple(value.shape) == (4, 1)
