from __future__ import annotations

import numpy as np
import torch
from simverse.envs.maze_race.agent import MazeRaceAgent
from simverse.envs.maze_race.config import MazeRaceConfig
from simverse.envs.maze_race.env import MazeRaceEnv, create_env
from simverse.policies.simple import SimplePolicy
from simverse.training.utils import (
    build_ppo_training_config,
    configure_torch_backend,
    resolve_rollout_dtype,
    resolve_torch_device,
    run_ppo_training,
)


def agent_factory(agent_id: int, policy: torch.nn.Module, env: MazeRaceEnv) -> MazeRaceAgent:
    action_values = np.arange(getattr(env.action_space, "n", 5), dtype=np.int64)
    return MazeRaceAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"maze_race_agent_{agent_id}",
    )


def train(
    num_envs: int = 512,
    episodes: int = 150,
    use_wandb: bool = False,
    use_compile: bool = True,
) -> None:
    device = resolve_torch_device(prefer_mps=True)
    dtype = resolve_rollout_dtype(device, cpu_dtype=torch.bfloat16)
    configure_torch_backend(device)

    config = MazeRaceConfig(
        width=7,
        height=7,
        num_agents=2,
        num_envs=max(1, int(num_envs)),
        max_steps=200,
        win_reward=1.0,
        lose_penalty=1.0,
        draw_reward=0.0,
        policies=[],
    )

    env = create_env(config, num_envs=config.num_envs, device=device, dtype=dtype)
    training_config = build_ppo_training_config(
        num_agents=config.num_agents,
        num_envs=config.num_envs,
        max_steps=config.max_steps,
        episodes=int(episodes),
        training_epochs=1,
        lr=3e-4,
        batch_size=config.num_envs * 2,
        buffer_size=config.num_envs * config.num_agents * 8,
        device=device,
        dtype=dtype,
    )

    run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=agent_factory,
        policy_factory=lambda obs_space, action_space: SimplePolicy(
            obs_space=obs_space,
            action_space=action_space,
        ),
        title="Maze Race Training",
        run_name="ppo-maze-race",
        episode_save_dir="recordings/maze_race",
        use_wandb=use_wandb,
        use_compile=use_compile,
        policy_name_prefix="maze_race_agent",
    )


if __name__ == "__main__":
    train()
