from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from simverse.core.env import SimEnv

# Gymnasium releases that still use np.float_ break on NumPy 2.x; restore alias for compatibility.
if not hasattr(np, "float_"):
    np.float_ = np.float64  # type: ignore[attr-defined]


@dataclass
class GymTorchConfig:
    env_id: str = "CartPole-v1"
    num_agents: int = 1
    num_envs: int = 256
    max_steps: int = 500
    seed: int | None = None
    policies: list[Any] = field(default_factory=list)


def observation_batch_to_chw(obs_batch: np.ndarray) -> np.ndarray:
    """Convert a batch of Gym observations into NCHW format."""
    obs = np.asarray(obs_batch, dtype=np.float32)
    if obs.ndim == 1:
        obs = np.expand_dims(obs, axis=0)

    if obs.ndim == 2:
        # [N, D] -> [N, 1, 1, D]
        return obs[:, None, None, :]
    if obs.ndim == 3:
        # [N, H, W] -> [N, 1, H, W]
        return obs[:, None, :, :]
    if obs.ndim == 4:
        # Heuristic: if channel appears last and first spatial dim is not channel-like,
        # treat as NHWC and transpose to NCHW.
        if obs.shape[-1] in (1, 3, 4) and obs.shape[1] not in (1, 3, 4):
            return np.transpose(obs, (0, 3, 1, 2))
        return obs

    flattened = obs.reshape(obs.shape[0], -1)
    return flattened[:, None, None, :]


class GymTorchEnv(SimEnv):
    """Torch-friendly wrapper around Gymnasium vector environments.

    This wrapper currently supports single-agent discrete-action Gym environments.
    Observations are converted to channel-first tensors so they can plug into
    existing Simverse policy/training code paths.
    """

    WINNER_NONE = -1

    def __init__(
        self,
        config: GymTorchConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if int(self.config.num_agents) != 1:
            raise ValueError("GymTorchEnv currently supports exactly one agent")

        self.num_envs = self._resolve_num_envs(num_envs, self.config)
        self.num_agents = 1
        self.agents: list[Any] = []

        env_fns = [self._make_env_fn(index) for index in range(self.num_envs)]
        self.vector_env = gym.vector.SyncVectorEnv(env_fns)

        single_action_space = self.vector_env.single_action_space
        single_obs_space = self.vector_env.single_observation_space
        if not isinstance(single_action_space, gym.spaces.Discrete):
            raise TypeError(
                "GymTorchEnv requires gym.spaces.Discrete action spaces; "
                f"got {type(single_action_space).__name__}"
            )
        if not isinstance(single_obs_space, gym.spaces.Box):
            raise TypeError(
                "GymTorchEnv requires gym.spaces.Box observation spaces; "
                f"got {type(single_obs_space).__name__}"
            )

        self._single_action_space = single_action_space
        self._single_obs_space = single_obs_space

        sample_obs = np.zeros((1, *single_obs_space.shape), dtype=np.float32)
        transformed_sample = observation_batch_to_chw(sample_obs)
        if transformed_sample.ndim != 4:
            raise ValueError("Unable to convert observation space to NCHW")
        self._obs_shape = tuple(int(v) for v in transformed_sample.shape[1:])

        low = self._transform_single_observation(np.asarray(single_obs_space.low, dtype=np.float32))
        high = self._transform_single_observation(
            np.asarray(single_obs_space.high, dtype=np.float32)
        )
        self._observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), self.WINNER_NONE, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))

        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._last_obs: torch.Tensor | None = None
        self._reset_calls = 0

    def _make_env_fn(self, index: int):
        del index

        def _build() -> gym.Env:
            env = gym.make(self.config.env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            return env

        return _build

    def _next_seed(self) -> int | None:
        base_seed = self.config.seed
        if base_seed is None:
            return None
        next_seed = int(base_seed) + self._reset_calls
        self._reset_calls += 1
        return next_seed

    def _transform_single_observation(self, obs: np.ndarray) -> np.ndarray:
        batch = np.expand_dims(obs, axis=0)
        return observation_batch_to_chw(batch)[0]

    @property
    def action_space(self):
        return self._single_action_space

    @property
    def observation_space(self):
        return self._observation_space

    def assign_agents(self, agents: list[Any]) -> None:
        self._assign_agents(agents, expected_count=1, label="GymTorchEnv")

    def reset(self) -> Dict[str, torch.Tensor]:
        obs_batch, _ = self.vector_env.reset(seed=self._next_seed())
        obs_tensor = self._obs_batch_to_tensor(obs_batch)

        self._reset_episode_state(winner_none=self.WINNER_NONE)
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        self._last_obs = obs_tensor

        return self._pack_observation(obs_tensor)

    def step(
        self,
        actions: torch.Tensor | Sequence[int] | np.ndarray | Dict[int, int] | None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_array = self._normalize_actions(actions)
        obs_batch, rewards, terminated, truncated, info = self.vector_env.step(action_array)

        done_np = np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)
        reward_np = np.asarray(rewards, dtype=np.float32)

        self._episode_returns += reward_np
        self._episode_lengths += 1

        finished_return = np.zeros(self.num_envs, dtype=np.float32)
        finished_length = np.zeros(self.num_envs, dtype=np.int64)

        done_indices = np.flatnonzero(done_np)
        if done_indices.size > 0:
            finished_return[done_indices] = self._episode_returns[done_indices]
            finished_length[done_indices] = self._episode_lengths[done_indices]

            auto_reset = isinstance(info, dict) and any(
                key in info for key in ("final_observation", "final_obs", "final_info")
            )
            if not auto_reset:
                for idx in done_indices.tolist():
                    reset_obs, _ = self.vector_env.envs[idx].reset(seed=self._next_seed())
                    obs_batch[idx] = reset_obs

            self._episode_returns[done_indices] = 0.0
            self._episode_lengths[done_indices] = 0

        obs_tensor = self._obs_batch_to_tensor(obs_batch)
        reward_tensor = torch.as_tensor(reward_np, dtype=self.dtype, device=self.device).unsqueeze(
            1
        )
        done_tensor = torch.as_tensor(done_np, dtype=torch.bool, device=self.device)

        self.done.copy_(done_tensor)
        self.steps.add_(1)
        if done_indices.size > 0:
            done_idx_tensor = torch.as_tensor(done_indices, dtype=torch.int64, device=self.device)
            self.steps[done_idx_tensor] = 0

        self._last_obs = obs_tensor

        info_dict: Dict[str, Any] = self._build_info(
            extra={
                "episode_return": torch.as_tensor(
                    finished_return,
                    dtype=torch.float32,
                    device=self.device,
                ),
                "episode_length": torch.as_tensor(
                    finished_length,
                    dtype=torch.int64,
                    device=self.device,
                ),
            }
        )
        if isinstance(info, dict):
            info_dict["gym_info"] = info

        return self._pack_observation(obs_tensor), reward_tensor, done_tensor.clone(), info_dict

    def _pack_observation(self, obs_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._pack_observation_dict(obs_tensor)

    def _obs_batch_to_tensor(self, obs_batch: np.ndarray) -> torch.Tensor:
        chw_batch = observation_batch_to_chw(obs_batch)
        return torch.as_tensor(chw_batch, dtype=self.dtype, device=self.device)

    def _normalize_actions(
        self,
        actions: torch.Tensor | Sequence[int] | np.ndarray | Dict[int, int] | None,
    ) -> np.ndarray:
        action_tensor = self._normalize_single_agent_actions(
            actions,
            missing_action=0,
            dict_default=0,
        )
        action_tensor = action_tensor.to(dtype=torch.int64, device="cpu")
        action_tensor = torch.clamp(action_tensor, 0, self._single_action_space.n - 1)
        return action_tensor.numpy()

    def get_observation(self) -> Dict[str, torch.Tensor]:
        if self._last_obs is None:
            return self.reset()
        return self._pack_observation(self._last_obs)

    def close(self) -> None:
        self.vector_env.close()

    def __del__(self) -> None:  # pragma: no cover
        with torch.no_grad():
            try:
                self.close()
            except Exception:
                pass


GymEnv = GymTorchEnv


def create_env(
    config: GymTorchConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GymTorchEnv:
    return GymTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
