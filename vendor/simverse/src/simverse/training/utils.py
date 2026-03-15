from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence
from urllib.parse import urlencode

import torch

from simverse.core.simulator import Simulator
from simverse.policies import CentralizedCritic
from simverse.training.logging import training_logger
from simverse.training.ppo import PPOTrainer
from simverse.training.stats import TrainingStats
from simverse.training.wandb import DEFAULT_WANDB_PROJECT


@dataclass
class PolicySpec:
    name: str
    model: torch.nn.Module


MARLPPOVariant = Literal["ippo", "mappo"]


def _default_centralized_critic_factory(obs_space: Any) -> torch.nn.Module:
    return CentralizedCritic(obs_space)


def _infer_render_game_slug(title: str, episode_save_dir: str | None) -> str:
    if episode_save_dir:
        replay_dir_name = Path(episode_save_dir).name.lower()
        mapping = {
            "snake": "snake",
            "maze_race": "maze",
            "battle_grid": "battle-grid",
        }
        if replay_dir_name in mapping:
            return mapping[replay_dir_name]
    normalized = title.lower()
    if "maze" in normalized:
        return "maze"
    if "battle" in normalized:
        return "battle-grid"
    return "snake"


def _build_render_url(game_slug: str, replay_dir: str | None) -> str:
    render_base = os.getenv("SIMVERSE_RENDER_URL", "http://127.0.0.1:5173/render").rstrip("/")
    params = {"game": game_slug}
    if replay_dir:
        params["dir"] = replay_dir
    return f"{render_base}?{urlencode(params)}"


def resolve_torch_device(*, prefer_mps: bool = True) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_rollout_dtype(
    device: str,
    *,
    cuda_dtype: torch.dtype = torch.float16,
    mps_dtype: torch.dtype = torch.bfloat16,
    cpu_dtype: torch.dtype = torch.float32,
) -> torch.dtype:
    if device == "cuda":
        return cuda_dtype
    if device == "mps":
        return mps_dtype
    return cpu_dtype


def configure_torch_backend(device: str) -> None:
    if device != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def compile_policy_models(
    policy_specs: Sequence[PolicySpec],
    *,
    use_compile: bool,
    device: str,
) -> list[torch.nn.Module]:
    models = [policy.model for policy in policy_specs]
    if use_compile and hasattr(torch, "compile") and device == "cuda":
        return [torch.compile(model, mode="max-autotune") for model in models]
    return models


def build_adam_optimizers(
    policy_models: Sequence[torch.nn.Module],
    *,
    lr: float,
    device: str,
) -> dict[int, torch.optim.Optimizer]:
    adam_kwargs: dict[str, Any] = {}
    if "fused" in inspect.signature(torch.optim.Adam).parameters and device == "cuda":
        adam_kwargs["fused"] = True
    return {
        agent_id: torch.optim.Adam(model.parameters(), lr=lr, **adam_kwargs)
        for agent_id, model in enumerate(policy_models)
    }


def build_ppo_training_config(
    *,
    num_agents: int,
    num_envs: int,
    max_steps: int,
    episodes: int,
    training_epochs: int,
    lr: float,
    batch_size: int,
    buffer_size: int,
    device: str,
    dtype: torch.dtype,
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rollout_multiple = int(num_envs) * int(num_agents)
    min_rollout_steps = max(1, min(int(max_steps), 64))
    min_buffer_size = rollout_multiple * min_rollout_steps
    resolved_buffer_size = max(int(buffer_size), min_buffer_size)
    resolved_buffer_size = (
        (resolved_buffer_size + rollout_multiple - 1) // rollout_multiple
    ) * rollout_multiple
    config = {
        "num_agents": int(num_agents),
        "num_envs": int(num_envs),
        "max_steps": int(max_steps),
        "episodes": int(episodes),
        "training_epochs": int(training_epochs),
        "clip_epsilon": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "buffer_size": resolved_buffer_size,
        "device": device,
        "dtype": dtype,
    }
    if extras:
        config.update(dict(extras))
    return config


def run_ppo_training(
    *,
    env: Any,
    training_config: Mapping[str, Any],
    agent_factory: Callable[[int, torch.nn.Module, Any], Any],
    policy_factory: Callable[[Any, Any], torch.nn.Module],
    title: str,
    run_name: str = "ppo-training",
    episode_save_dir: str | None = None,
    use_wandb: bool = True,
    use_compile: bool = False,
    project_name: str = DEFAULT_WANDB_PROJECT,
    policy_name_prefix: str = "agent",
    variant: MARLPPOVariant = "ippo",
    centralized_critic_factory: Callable[[Any], torch.nn.Module] | None = None,
) -> list[torch.nn.Module]:
    num_agents = int(training_config["num_agents"])
    device = str(training_config["device"])
    policy_specs = [
        PolicySpec(
            name=f"{policy_name_prefix}_{agent_id}",
            model=policy_factory(env.observation_space, env.action_space),
        )
        for agent_id in range(num_agents)
    ]
    if hasattr(env, "config"):
        env.config.policies = policy_specs

    policy_models = compile_policy_models(
        policy_specs,
        use_compile=use_compile,
        device=device,
    )
    optimizers = build_adam_optimizers(
        policy_models,
        lr=float(training_config["lr"]),
        device=device,
    )
    variant_name = str(variant).lower()
    if variant_name not in {"ippo", "mappo"}:
        raise ValueError(f"Unsupported PPO variant {variant!r}. Expected 'ippo' or 'mappo'.")
    is_mappo = variant_name == "mappo"
    critic_model: torch.nn.Module | None = None
    critic_optimizer: torch.optim.Optimizer | None = None
    if is_mappo:
        if centralized_critic_factory is None:
            centralized_critic_factory = _default_centralized_critic_factory
        critic_obs_space = env.observation_space
        if hasattr(critic_obs_space, "spaces") and "obs" in critic_obs_space.spaces:
            critic_obs_space = critic_obs_space["obs"]
        critic_model = centralized_critic_factory(critic_obs_space)
        critic_lr = float(training_config.get("critic_lr", training_config["lr"]))
        critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr=critic_lr)

    resolved_training_config = dict(training_config)
    resolved_training_config.update(
        {
            "ppo_variant": variant_name,
            "ctde": is_mappo,
            # Paper-aligned stable defaults for cooperative MARL PPO.
            "normalize_advantages": bool(training_config.get("normalize_advantages", True)),
            "clip_epsilon": float(training_config.get("clip_epsilon", 0.2)),
            "value_normalization": bool(training_config.get("value_normalization", True)),
        }
    )

    loss_trainer = PPOTrainer(
        optimizers=optimizers,
        centralized_critic=critic_model,
        centralized_critic_optimizer=critic_optimizer,
        episodes=int(training_config["episodes"]),
        training_epochs=int(training_config["training_epochs"]),
        clip_epsilon=float(training_config["clip_epsilon"]),
        gamma=float(training_config["gamma"]),
        gae_lambda=float(training_config["gae_lambda"]),
        stats=TrainingStats(),
        config=resolved_training_config,
        project_name=project_name,
        run_name=run_name,
        episode_save_dir=episode_save_dir,
        device=device,
        batch_size=int(training_config["batch_size"]),
        buffer_size=int(training_config["buffer_size"]),
        dtype=training_config["dtype"],
        use_wandb=use_wandb,
    )

    simulator = Simulator(
        env=env,
        num_agents=num_agents,
        policies=policy_models,
        loss_trainer=loss_trainer,
        agent_factory=agent_factory,
    )
    resolved_replay_dir = (
        str(Path(episode_save_dir).expanduser().resolve()) if episode_save_dir else None
    )
    game_slug = _infer_render_game_slug(title, resolved_replay_dir)
    render_url = _build_render_url(game_slug, resolved_replay_dir)
    training_logger.info(f"Render URL: {render_url}")
    previous_clone_payload_tensors = getattr(env, "clone_payload_tensors", None)
    fast_payload_toggled = hasattr(env, "set_fast_payload_mode")
    if fast_payload_toggled:
        env.set_fast_payload_mode(True)
    try:
        simulator.train(title=title)
    finally:
        if fast_payload_toggled:
            env.clone_payload_tensors = bool(previous_clone_payload_tensors)
    return policy_models


def run_ippo_training(
    *,
    env: Any,
    training_config: Mapping[str, Any],
    agent_factory: Callable[[int, torch.nn.Module, Any], Any],
    policy_factory: Callable[[Any, Any], torch.nn.Module],
    title: str,
    run_name: str = "ippo-training",
    episode_save_dir: str | None = None,
    use_wandb: bool = True,
    use_compile: bool = False,
    project_name: str = DEFAULT_WANDB_PROJECT,
    policy_name_prefix: str = "agent",
) -> list[torch.nn.Module]:
    return run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=agent_factory,
        policy_factory=policy_factory,
        title=title,
        run_name=run_name,
        episode_save_dir=episode_save_dir,
        use_wandb=use_wandb,
        use_compile=use_compile,
        project_name=project_name,
        policy_name_prefix=policy_name_prefix,
        variant="ippo",
    )


def run_mappo_training(
    *,
    env: Any,
    training_config: Mapping[str, Any],
    agent_factory: Callable[[int, torch.nn.Module, Any], Any],
    policy_factory: Callable[[Any, Any], torch.nn.Module],
    title: str,
    run_name: str = "mappo-training",
    episode_save_dir: str | None = None,
    use_wandb: bool = True,
    use_compile: bool = False,
    project_name: str = DEFAULT_WANDB_PROJECT,
    policy_name_prefix: str = "agent",
    centralized_critic_factory: Callable[[Any], torch.nn.Module] | None = None,
) -> list[torch.nn.Module]:
    return run_ppo_training(
        env=env,
        training_config=training_config,
        agent_factory=agent_factory,
        policy_factory=policy_factory,
        title=title,
        run_name=run_name,
        episode_save_dir=episode_save_dir,
        use_wandb=use_wandb,
        use_compile=use_compile,
        project_name=project_name,
        policy_name_prefix=policy_name_prefix,
        variant="mappo",
        centralized_critic_factory=centralized_critic_factory,
    )
