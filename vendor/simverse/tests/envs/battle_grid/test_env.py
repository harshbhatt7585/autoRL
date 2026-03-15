from __future__ import annotations

import torch
from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.env import create_env


def test_battle_grid_observation_includes_feature_vector() -> None:
    env = create_env(
        BattleGridConfig(num_envs=4, num_agents=2, width=13, height=13, policies=[]),
        num_envs=4,
        device="cpu",
        dtype=torch.float32,
    )

    observation = env.reset()

    assert tuple(observation["obs"].shape) == (4, 2, 13, 13)
    assert tuple(observation["features"].shape) == (4, 3)
    assert env.observation_space["obs"].shape == (2, 13, 13)
    assert env.observation_space["features"].shape == (3,)


def test_battle_grid_observation_updates_positions_incrementally() -> None:
    env = create_env(
        BattleGridConfig(num_envs=1, num_agents=2, width=13, height=13, policies=[]),
        num_envs=1,
        device="cpu",
        dtype=torch.float32,
    )

    env.done.zero_()
    env.steps.zero_()
    env.health.fill_(env.max_health)
    env.agent_pos[0, 0] = torch.tensor([1, 1], dtype=torch.int64)
    env.agent_pos[0, 1] = torch.tensor([3, 3], dtype=torch.int64)
    env._reset_observation_cache()

    first_obs = env._get_observation()["obs"]
    assert first_obs[0, 0, 1, 1].item() == 1.0
    assert first_obs[0, 1, 3, 3].item() == 1.0

    env.agent_pos[0, 0] = torch.tensor([2, 1], dtype=torch.int64)
    env.health[0, 1] = 0
    second_obs = env._get_observation()["obs"]

    assert second_obs[0, 0, 1, 1].item() == 0.0
    assert second_obs[0, 0, 1, 2].item() == 1.0
    assert second_obs[0, 1, 3, 3].item() == 0.0
