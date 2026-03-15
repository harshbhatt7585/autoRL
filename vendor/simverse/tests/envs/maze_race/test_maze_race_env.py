from __future__ import annotations

import torch
from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.env import create_env


def test_maze_race_observation_reuses_static_channels_and_updates_positions() -> None:
    env = create_env(
        MazeRaceConfig(num_envs=1, num_agents=2, width=7, height=7, policies=[]),
        num_envs=1,
        device="cpu",
        dtype=torch.float32,
    )

    env.done.zero_()
    env.steps.zero_()
    env.agent_pos[0, 0] = torch.tensor([1, 1], dtype=torch.int64)
    env.agent_pos[0, 1] = torch.tensor([5, 1], dtype=torch.int64)
    env._reset_dynamic_observation_channels()

    first_obs = env._get_observation()["obs"]
    static_before = first_obs[0, : env.static_obs_channels].clone()
    assert first_obs[0, env.static_obs_channels + 0, 1, 1].item() == 1.0
    assert first_obs[0, env.static_obs_channels + 1, 1, 5].item() == 1.0

    env.agent_pos[0, 0] = torch.tensor([1, 2], dtype=torch.int64)
    second_obs = env._get_observation()["obs"]

    assert torch.equal(second_obs[0, : env.static_obs_channels], static_before)
    assert second_obs[0, env.static_obs_channels + 0, 1, 1].item() == 0.0
    assert second_obs[0, env.static_obs_channels + 0, 2, 1].item() == 1.0
    assert second_obs[0, env.static_obs_channels + 1, 1, 5].item() == 1.0
