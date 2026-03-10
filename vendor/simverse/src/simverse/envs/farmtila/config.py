from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class FarmtilaConfig:
    width: int = 50
    height: int = 50
    num_agents: int = 2
    num_envs: int = 1
    spawn_seed_every: int = 100
    seeds_per_spawn: int = 10
    max_steps: int = 10000
    total_seeds_per_episode: int = 500
    step_cost: float = 0.0
    score_delta_reward: float = 1.0
    terminal_win_reward: float = 1.0
    policies: List[Any] = field(default_factory=list)


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _derive_batch_size(
    *,
    num_envs: int,
    requested_batch_size: Optional[int],
    device: str,
) -> int:
    batch_size = int(requested_batch_size) if requested_batch_size is not None else num_envs * 2
    batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    if device == "mps" and num_envs <= 2048:
        batch_size = min(batch_size, 2048)
        batch_size = _round_up_to_multiple(max(num_envs, batch_size), num_envs)
    return batch_size


def _derive_buffer_size(
    *,
    num_envs: int,
    num_agents: int,
    max_steps: int,
    batch_size: int,
    requested_buffer_size: Optional[int],
) -> int:
    min_buffer_size = batch_size * num_agents
    rollout_buffer_size = num_envs * num_agents * max(1, min(int(max_steps), 64))
    default_buffer_size = max(min_buffer_size * 4, rollout_buffer_size)
    buffer_size = (
        int(requested_buffer_size) if requested_buffer_size is not None else default_buffer_size
    )
    return _round_up_to_multiple(
        max(min_buffer_size, rollout_buffer_size, buffer_size),
        num_envs * num_agents,
    )


def build_training_config(
    *,
    width: int = 20,
    height: int = 20,
    num_agents: int = 4,
    num_envs: int = 256,
    max_steps: int = 1000,
    episodes: int = 100,
    training_epochs: int = 1,
    lr: float = 0.001,
    clip_epsilon: float = 0.2,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    total_seeds: int = 500,
    batch_size: Optional[int] = None,
    buffer_size: Optional[int] = None,
    device: Optional[str] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, Any]:
    resolved_device = device or select_device()

    resolved_num_envs = max(1, int(num_envs))
    resolved_num_agents = max(1, int(num_agents))
    if resolved_device == "mps":
        resolved_num_envs = min(resolved_num_envs, 128)

    resolved_batch_size = _derive_batch_size(
        num_envs=resolved_num_envs,
        requested_batch_size=batch_size,
        device=resolved_device,
    )
    resolved_buffer_size = _derive_buffer_size(
        num_envs=resolved_num_envs,
        num_agents=resolved_num_agents,
        max_steps=max_steps,
        batch_size=resolved_batch_size,
        requested_buffer_size=buffer_size,
    )

    return {
        "width": width,
        "height": height,
        "num_agents": resolved_num_agents,
        "num_envs": resolved_num_envs,
        "max_steps": max_steps,
        "episodes": episodes,
        "training_epochs": training_epochs,
        "lr": lr,
        "clip_epsilon": clip_epsilon,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "total_seeds": total_seeds,
        "batch_size": resolved_batch_size,
        "buffer_size": resolved_buffer_size,
        "device": resolved_device,
        "dtype": dtype,
    }
