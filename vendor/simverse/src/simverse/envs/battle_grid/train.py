from __future__ import annotations

import numpy as np
import torch.nn as nn
from simverse.envs.battle_grid.agent import BattleGridAgent
from simverse.envs.battle_grid.config import BattleGridConfig
from simverse.envs.battle_grid.env import BattleGridEnv, create_env
from simverse.policies.simple import SimplePolicy
from simverse.training.utils import (
    build_ppo_training_config,
    configure_torch_backend,
    resolve_rollout_dtype,
    resolve_torch_device,
    run_ppo_training,
)


def agent_factory(agent_id: int, policy: nn.Module, env: BattleGridEnv) -> BattleGridAgent:
    action_values = np.arange(getattr(env.action_space, "n", 6), dtype=np.int64)
    return BattleGridAgent(
        agent_id=agent_id,
        action_space=action_values,
        policy=policy,
        name=f"battle_agent_{agent_id}",
    )


def train(
    num_envs: int = 512,
    episodes: int = 150,
    use_wandb: bool = False,
    use_compile: bool = True,
) -> None:
    device = resolve_torch_device(prefer_mps=False)
    dtype = resolve_rollout_dtype(device)
    configure_torch_backend(device)

    config = BattleGridConfig(
        width=13,
        height=13,
        num_agents=2,
        num_envs=max(1, int(num_envs)),
        max_steps=200,
        max_health=3,
        attack_damage=1,
        attack_range=1,
        step_penalty=0.01,
        damage_reward=0.05,
        kill_reward=1.0,
        death_penalty=1.0,
        timeout_win_reward=0.5,
        timeout_lose_penalty=0.5,
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
        batch_size=config.num_envs * 4,
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
        title="Battle Grid Training",
        run_name="ppo-battle-grid",
        episode_save_dir="recordings/battle_grid",
        use_wandb=use_wandb,
        use_compile=use_compile,
        policy_name_prefix="battle_agent",
    )


if __name__ == "__main__":
    train()
