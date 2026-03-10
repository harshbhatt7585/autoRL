from __future__ import annotations

import random

import torch
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig, build_training_config
from simverse.envs.farmtila.env import FarmtilaEnv, create_env
from simverse.policies.simple import SimplePolicy
from simverse.training.utils import run_ppo_training


def agent_factory(agent_id: int, policy: torch.nn.Module, env: FarmtilaEnv) -> FarmtilaAgent:
    return FarmtilaAgent(
        agent_id=agent_id,
        position=(
            random.randint(0, env.config.width - 1),
            random.randint(0, env.config.height - 1),
        ),
        action_space=env.action_space,
        policy=policy,
    )


def train(use_wandb: bool = True, use_compile: bool = True):
    training_config = build_training_config(
        num_agents=2,
        num_envs=2048,
        max_steps=1500,
        episodes=100,
        training_epochs=1,
        lr=0.001,
        clip_epsilon=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        total_seeds=500,
        batch_size=None,
        buffer_size=None,
        dtype=torch.float16,
    )

    config = FarmtilaConfig(
        width=training_config["width"],
        height=training_config["height"],
        num_agents=training_config["num_agents"],
        num_envs=training_config["num_envs"],
        total_seeds_per_episode=training_config["total_seeds"],
        max_steps=training_config["max_steps"],
        spawn_seed_every=100,
        seeds_per_spawn=10,
        policies=[],
    )
    env = create_env(
        config=config,
        num_envs=training_config["num_envs"],
        device=training_config["device"],
        dtype=training_config["dtype"],
    )
    run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=agent_factory,
        policy_factory=lambda obs_space, action_space: SimplePolicy(
            obs_space=obs_space,
            action_space=action_space,
        ),
        title="Farmtila Training",
        run_name="ppo-training",
        episode_save_dir="recordings/farmtila",
        use_wandb=use_wandb,
        use_compile=use_compile,
        policy_name_prefix="simple_agent",
    )


if __name__ == "__main__":
    train()
