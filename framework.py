from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any


VENDORED_SIMVERSE_SRC = Path(__file__).resolve().parent / "vendor" / "simverse" / "src"
if VENDORED_SIMVERSE_SRC.is_dir() and str(VENDORED_SIMVERSE_SRC) not in sys.path:
    sys.path.insert(0, str(VENDORED_SIMVERSE_SRC))

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from candidate import env as candidate_env_module
    from candidate.train import build_policy, training_overrides
    from simverse.core.agent import SimAgent
    from simverse.core.simulator import Simulator
    from simverse.training.ppo import PPOTrainer
    from simverse.training.stats import TrainingStats
    from simverse.training.utils import (
        build_adam_optimizers,
        build_ppo_training_config,
        configure_torch_backend,
        resolve_torch_device,
    )
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "Missing Simverse runtime dependencies. Use `.venv/bin/pip install -e vendor/simverse` "
        "and run the harness with `.venv/bin/python train.py`."
    ) from exc


DEFAULT_NUM_ENVS = 32
DEFAULT_MAX_STEPS = 20
DEFAULT_TRAIN_EPISODES = 12
DEFAULT_EVAL_EPISODES = 8
MAX_TRAIN_EPISODES = 1000
MAX_EVAL_EPISODES = 100
DEFAULT_SEED_COUNT = 2
DEFAULT_LR = 3e-4
ALLOWED_TRAINING_OVERRIDE_KEYS = {
    "batch_size",
    "buffer_size",
    "clip_epsilon",
    "entropy_coef",
    "gae_lambda",
    "gamma",
    "lr",
    "max_grad_norm",
    "normalize_advantages",
    "torch_fastpath",
    "training_epochs",
    "max_steps",
}


@dataclass(frozen=True)
class AutoRLEnvConfig:
    num_agents: int = 1
    num_envs: int = DEFAULT_NUM_ENVS
    max_steps: int = DEFAULT_MAX_STEPS
    seed: int | None = None
    policies: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    env_name: str
    env_description: str
    score: float
    mean_eval_return: float
    mean_solve_rate: float
    mean_train_return: float
    learning_gain: float
    headroom_bonus: float
    stability: float
    complexity_penalty: float
    mean_episode_length: float
    observation_shape: tuple[int, ...]
    num_actions: int
    max_steps: int
    num_envs: int
    train_episodes: int
    eval_episodes: int
    seed_count: int
    device: str


@dataclass(frozen=True)
class RolloutMetrics:
    mean_return: float
    solve_rate: float
    mean_episode_length: float


class SingleAgent(SimAgent):
    def __init__(
        self,
        agent_id: int,
        action_count: int,
        policy: nn.Module | None = None,
        name: str | None = None,
    ) -> None:
        action_space = np.arange(action_count, dtype=np.int64)
        super().__init__(name=name or f"agent_{agent_id}", action_space=action_space, policy=policy)
        self.agent_id = agent_id
        self.memory: dict[str, Any] = {}
        self.reward = 0.0

    def action(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.array([0], dtype=np.int64)

    def info(self) -> dict[str, float]:
        return {"agent_id": float(self.agent_id), "reward": float(self.reward)}

    def reset(self) -> None:
        self.reward = 0.0
        self.memory.clear()

    def get_action_space(self) -> np.ndarray:
        return self.action_space

    def get_memory(self) -> dict[str, Any]:
        return self.memory

    def current_state(self) -> np.ndarray:
        return np.array([self.reward], dtype=np.float32)

    def get_policy(self) -> nn.Module | None:
        return self.policy

    def set_policy(self, policy: nn.Module | None) -> None:
        self.policy = policy


def agent_factory(agent_id: int, policy: nn.Module, env: Any) -> SingleAgent:
    return SingleAgent(
        agent_id=agent_id,
        action_count=env.action_space.n,
        policy=policy,
        name=f"{env.__class__.__name__.lower()}_{agent_id}",
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return resolve_torch_device(prefer_mps=True)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def _compute_stability(values: list[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mean_value = statistics.fmean(values)
    spread = statistics.pstdev(values)
    return _clamp(1.0 - (spread / (abs(mean_value) + 1.0)), 0.0, 1.0)


def _complexity_penalty(
    observation_shape: tuple[int, ...],
    *,
    num_actions: int,
) -> float:
    if len(observation_shape) != 3:
        return 5.0
    channels, height, width = observation_shape
    spatial_penalty = max(0, (height * width) - 25) * 0.08
    channel_penalty = max(0, channels - 6) * 0.40
    action_penalty = max(0, num_actions - 5) * 1.50
    return spatial_penalty + channel_penalty + action_penalty


def _mean_window(values: list[float]) -> float:
    if not values:
        return 0.0
    window = max(1, min(len(values), max(2, len(values) // 3)))
    return statistics.fmean(values[-window:])


def _build_candidate_training_config(
    *,
    train_episodes: int,
    num_envs: int,
    max_steps: int,
    lr: float,
    device: str,
) -> dict[str, Any]:
    overrides = _read_candidate_training_overrides(
        num_envs=num_envs,
        max_steps=max_steps,
        device=device,
    )
    overrides.pop("max_steps", None)

    resolved_lr = float(overrides.pop("lr", lr))
    resolved_training_epochs = int(overrides.pop("training_epochs", 2))
    resolved_batch_size = int(overrides.pop("batch_size", max(32, num_envs * 4)))
    resolved_buffer_size = int(overrides.pop("buffer_size", max(num_envs * max_steps, 256)))

    training_config = build_ppo_training_config(
        num_agents=1,
        num_envs=num_envs,
        max_steps=max_steps,
        episodes=train_episodes,
        training_epochs=resolved_training_epochs,
        lr=resolved_lr,
        batch_size=resolved_batch_size,
        buffer_size=resolved_buffer_size,
        device=device,
        dtype=torch.float32,
    )
    training_config.update(overrides)
    training_config["num_agents"] = 1
    training_config["num_envs"] = num_envs
    training_config["max_steps"] = max_steps
    training_config["episodes"] = train_episodes
    training_config["device"] = device
    training_config["dtype"] = torch.float32
    return training_config


def _read_candidate_training_overrides(
    *,
    num_envs: int,
    max_steps: int,
    device: str,
) -> dict[str, Any]:
    overrides = dict(training_overrides(num_envs=num_envs, max_steps=max_steps, device=device) or {})
    unexpected = sorted(set(overrides) - ALLOWED_TRAINING_OVERRIDE_KEYS)
    if unexpected:
        raise ValueError(
            "candidate.train.training_overrides() returned unsupported keys: "
            + ", ".join(unexpected)
        )
    return overrides


def _resolve_candidate_max_steps(
    requested_max_steps: int | None,
    *,
    num_envs: int,
    device: str,
) -> int:
    if requested_max_steps is not None:
        resolved = int(requested_max_steps)
        if resolved < 1:
            raise ValueError("max_steps must be at least 1.")
        return resolved

    overrides = _read_candidate_training_overrides(
        num_envs=num_envs,
        max_steps=DEFAULT_MAX_STEPS,
        device=device,
    )
    max_steps = int(overrides.get("max_steps", DEFAULT_MAX_STEPS))
    if max_steps < 1:
        raise ValueError("candidate.train.training_overrides()['max_steps'] must be at least 1.")
    return max_steps


def _make_env(
    *,
    num_envs: int,
    max_steps: int,
    seed: int,
    device: str,
    dtype: torch.dtype,
):
    config = AutoRLEnvConfig(
        num_agents=1,
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
    )
    return candidate_env_module.create_env(config, num_envs=num_envs, device=device, dtype=dtype)


def _evaluate_policy(
    policy: nn.Module,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    device: str,
) -> RolloutMetrics:
    eval_env = _make_env(
        num_envs=1,
        max_steps=max_steps,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[int] = []

    policy.eval()
    with torch.no_grad():
        for _episode_idx in range(episodes):
            obs = eval_env.reset()
            episode_return = 0.0
            episode_success = False
            for step_idx in range(max_steps):
                obs_tensor = obs["obs"].to(device=device, dtype=torch.float32)
                logits, _ = policy(obs_tensor)
                action = torch.argmax(logits, dim=-1)
                obs, reward, done, info = eval_env.step(action)

                episode_return += float(reward[0, 0].item())
                episode_success = episode_success or bool(info["success"][0].item())

                if bool(done[0].item()):
                    recorded_return = float(info["episode_return"][0].item())
                    if recorded_return != 0.0:
                        episode_return = recorded_return
                    recorded_length = int(info["episode_length"][0].item())
                    lengths.append(recorded_length if recorded_length > 0 else step_idx + 1)
                    successes.append(float(episode_success))
                    returns.append(episode_return)
                    break
            else:
                lengths.append(max_steps)
                successes.append(float(episode_success))
                returns.append(episode_return)

    eval_env.close()
    return RolloutMetrics(
        mean_return=statistics.fmean(returns) if returns else 0.0,
        solve_rate=statistics.fmean(successes) if successes else 0.0,
        mean_episode_length=statistics.fmean(lengths) if lengths else float(max_steps),
    )


def _train_single_seed(
    *,
    seed: int,
    train_episodes: int,
    eval_episodes: int,
    num_envs: int,
    max_steps: int,
    lr: float,
    device: str,
) -> dict[str, float]:
    _seed_everything(seed)
    configure_torch_backend(device)

    env = _make_env(
        num_envs=num_envs,
        max_steps=max_steps,
        seed=seed,
        device=device,
        dtype=torch.float32,
    )
    policy_model = build_policy(env.observation_space, env.action_space)

    baseline_metrics = _evaluate_policy(
        policy_model,
        episodes=max(2, min(eval_episodes, 4)),
        seed=seed + 10_000,
        max_steps=max_steps,
        device=device,
    )

    policy_models = [policy_model]
    training_config = _build_candidate_training_config(
        train_episodes=train_episodes,
        num_envs=num_envs,
        max_steps=max_steps,
        lr=lr,
        device=device,
    )
    optimizers = build_adam_optimizers(
        policy_models,
        lr=float(training_config["lr"]),
        device=device,
    )
    stats = TrainingStats()

    trainer = PPOTrainer(
        optimizers=optimizers,
        episodes=int(training_config["episodes"]),
        training_epochs=int(training_config["training_epochs"]),
        clip_epsilon=float(training_config["clip_epsilon"]),
        gamma=float(training_config["gamma"]),
        gae_lambda=float(training_config["gae_lambda"]),
        stats=stats,
        config=dict(training_config),
        run_name=f"autorl-seed-{seed}",
        use_wandb=False,
        device=device,
        batch_size=int(training_config["batch_size"]),
        buffer_size=int(training_config["buffer_size"]),
        dtype=torch.float32,
    )
    simulator = Simulator(
        env=env,
        num_agents=1,
        policies=policy_models,
        loss_trainer=trainer,
        agent_factory=agent_factory,
    )

    try:
        simulator.train(title="autoRL candidate")
    finally:
        env.close()

    training_rewards = [float(value) for value in stats.episode_rewards]
    early_window = max(1, min(len(training_rewards), max(2, len(training_rewards) // 3)))
    early_train = statistics.fmean(training_rewards[:early_window]) if training_rewards else 0.0
    late_train = statistics.fmean(training_rewards[-early_window:]) if training_rewards else 0.0

    post_metrics = _evaluate_policy(
        policy_model,
        episodes=eval_episodes,
        seed=seed + 20_000,
        max_steps=max_steps,
        device=device,
    )

    return {
        "eval_return": post_metrics.mean_return,
        "solve_rate": post_metrics.solve_rate,
        "episode_length": post_metrics.mean_episode_length,
        "learning_gain": late_train - early_train,
        "headroom_bonus": max(0.0, 1.0 - baseline_metrics.solve_rate) * post_metrics.solve_rate,
        "mean_train_return": _mean_window(training_rewards),
    }


def evaluate_candidate(
    *,
    train_episodes: int = DEFAULT_TRAIN_EPISODES,
    eval_episodes: int = DEFAULT_EVAL_EPISODES,
    seed_count: int = DEFAULT_SEED_COUNT,
    num_envs: int = DEFAULT_NUM_ENVS,
    max_steps: int | None = None,
    lr: float = DEFAULT_LR,
    device: str = "cpu",
) -> EvaluationResult:
    if train_episodes < 2:
        raise ValueError("train_episodes must be at least 2.")
    if train_episodes > MAX_TRAIN_EPISODES:
        raise ValueError(f"train_episodes must be at most {MAX_TRAIN_EPISODES}.")
    if eval_episodes < 1:
        raise ValueError("eval_episodes must be at least 1.")
    if eval_episodes > MAX_EVAL_EPISODES:
        raise ValueError(f"eval_episodes must be at most {MAX_EVAL_EPISODES}.")
    if seed_count < 1:
        raise ValueError("seed_count must be at least 1.")
    if num_envs < 1:
        raise ValueError("num_envs must be at least 1.")

    resolved_device = _resolve_device(device)
    resolved_max_steps = _resolve_candidate_max_steps(
        max_steps,
        num_envs=num_envs,
        device=resolved_device,
    )
    prototype_env = _make_env(
        num_envs=1,
        max_steps=resolved_max_steps,
        seed=0,
        device=resolved_device,
        dtype=torch.float32,
    )
    observation_shape = tuple(int(dim) for dim in prototype_env.observation_space.shape)
    num_actions = int(prototype_env.action_space.n)
    env_description = (
        prototype_env.describe()
        if hasattr(prototype_env, "describe")
        else prototype_env.__class__.__name__
    )
    env_name = prototype_env.__class__.__name__
    prototype_env.close()

    seed_results = [
        _train_single_seed(
            seed=101 + idx * 37,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            num_envs=num_envs,
            max_steps=resolved_max_steps,
            lr=lr,
            device=resolved_device,
        )
        for idx in range(seed_count)
    ]

    eval_returns = [item["eval_return"] for item in seed_results]
    mean_eval_return = statistics.fmean(eval_returns)
    mean_solve_rate = statistics.fmean(item["solve_rate"] for item in seed_results)
    mean_episode_length = statistics.fmean(item["episode_length"] for item in seed_results)
    learning_gain = statistics.fmean(item["learning_gain"] for item in seed_results)
    headroom_bonus = statistics.fmean(item["headroom_bonus"] for item in seed_results)
    mean_train_return = statistics.fmean(item["mean_train_return"] for item in seed_results)
    stability = _compute_stability(eval_returns)
    complexity_penalty = _complexity_penalty(
        observation_shape,
        num_actions=num_actions,
    )

    normalized_gain = _clamp(math.tanh(learning_gain / 0.25), 0.0, 1.0)
    normalized_train_return = _clamp((mean_train_return + 0.5) / 1.5, 0.0, 1.0)
    score = (
        42.0 * mean_solve_rate
        + 20.0 * headroom_bonus
        + 18.0 * normalized_gain
        + 10.0 * stability
        + 10.0 * normalized_train_return
        - complexity_penalty
    )

    return EvaluationResult(
        env_name=env_name,
        env_description=env_description,
        score=max(0.0, score),
        mean_eval_return=mean_eval_return,
        mean_solve_rate=mean_solve_rate,
        mean_train_return=mean_train_return,
        learning_gain=learning_gain,
        headroom_bonus=headroom_bonus,
        stability=stability,
        complexity_penalty=complexity_penalty,
        mean_episode_length=mean_episode_length,
        observation_shape=observation_shape,
        num_actions=num_actions,
        max_steps=resolved_max_steps,
        num_envs=num_envs,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        seed_count=seed_count,
        device=resolved_device,
    )
