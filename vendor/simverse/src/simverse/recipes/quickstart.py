"""Utility function to run a short random-policy Gymnasium rollout."""

from __future__ import annotations

from typing import Dict, List, Type

import gymnasium as gym
import numpy as np

from simverse.policies import RandomPolicy


def quicktrain(
    env_id: str = "CartPole-v1",
    episodes: int = 5,
    max_steps: int = 200,
    render: bool = False,
    policy_cls: Type[RandomPolicy] = RandomPolicy,
) -> Dict[str, float]:
    """Runs a basic random-policy rollout against a Gymnasium environment."""

    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    policy = policy_cls(env.action_space)

    rewards: List[float] = []
    for _episode in range(episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        for _step in range(max_steps):
            action = policy.act(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(total_reward)

    env.close()

    stats = {
        "env_id": env_id,
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }
    return stats


__all__ = ["quicktrain"]
