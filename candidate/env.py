from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from simverse.core.env import SimEnv


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

GRID_SIZE = 5
CHANNEL_COUNT = 1
START_X = GRID_SIZE // 2
START_Y = GRID_SIZE // 2


class EmptyCanvasEnv(SimEnv):
    """Intentionally minimal starter environment to be replaced by the search agent."""

    def __init__(
        self,
        config: Any,
        *,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        self.num_agents = 1
        self.num_envs = self._resolve_num_envs(num_envs, config, default=32)
        self.max_steps = int(config.max_steps)

        self._action_space = gym.spaces.Discrete(4)
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(CHANNEL_COUNT, GRID_SIZE, GRID_SIZE),
            dtype=float,
        )

        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("agent_x", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("agent_y", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer(
            "episode_return",
            torch.zeros(self.num_envs, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "episode_length",
            torch.zeros(self.num_envs, dtype=torch.int64, device=self.device),
        )

        self.reset()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def describe(self) -> str:
        return "An intentionally empty 5x5 starter canvas with movement only and no objective yet."

    def assign_agents(self, agents: list[Any]) -> None:
        self._assign_agents(agents, expected_count=1, label="EmptyCanvasEnv")

    def reset(self) -> dict[str, torch.Tensor]:
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_subset(all_envs)
        self.done.zero_()
        self.steps.zero_()
        self.episode_return.zero_()
        self.episode_length.zero_()
        return self._pack_observation_dict(self._build_observation())

    def step(
        self,
        actions: torch.Tensor | list[int] | tuple[int, ...] | dict[int, int] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        action_tensor = self._normalize_single_agent_actions(
            actions,
            missing_action=UP,
            dict_default=UP,
        )

        self.done.zero_()
        self.steps.add_(1)
        self.episode_length.add_(1)

        delta_x = torch.tensor([0, 0, -1, 1], dtype=torch.int64, device=self.device)
        delta_y = torch.tensor([-1, 1, 0, 0], dtype=torch.int64, device=self.device)
        next_x = torch.clamp(self.agent_x + delta_x[action_tensor], 0, GRID_SIZE - 1)
        next_y = torch.clamp(self.agent_y + delta_y[action_tensor], 0, GRID_SIZE - 1)
        self.agent_x.copy_(next_x)
        self.agent_y.copy_(next_y)

        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.episode_return.add_(reward)
        done = self.steps >= self.max_steps
        self.done.copy_(done)

        finished_return = torch.where(done, self.episode_return, torch.zeros_like(self.episode_return))
        finished_length = torch.where(done, self.episode_length, torch.zeros_like(self.episode_length))

        if bool(done.any().item()):
            self._reset_subset(done)
            self.steps.copy_(torch.where(done, torch.zeros_like(self.steps), self.steps))
            self.episode_return.copy_(
                torch.where(done, torch.zeros_like(self.episode_return), self.episode_return)
            )
            self.episode_length.copy_(
                torch.where(done, torch.zeros_like(self.episode_length), self.episode_length)
            )

        observation = self._pack_observation_dict(self._build_observation())
        info = self._build_info(
            extra={
                "success": success,
                "episode_return": finished_return,
                "episode_length": finished_length,
            }
        )
        return observation, reward.unsqueeze(1), done.clone(), info

    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros(
            (self.num_envs, CHANNEL_COUNT, GRID_SIZE, GRID_SIZE),
            dtype=self.dtype,
            device=self.device,
        )
        batch_index = torch.arange(self.num_envs, device=self.device)
        obs[batch_index, 0, self.agent_y, self.agent_x] = 1.0
        return obs

    def _reset_subset(self, mask: torch.Tensor) -> None:
        self.agent_x.copy_(torch.where(mask, torch.full_like(self.agent_x, START_X), self.agent_x))
        self.agent_y.copy_(torch.where(mask, torch.full_like(self.agent_y, START_Y), self.agent_y))

    def close(self) -> None:
        return None


CandidateEnv = EmptyCanvasEnv


def create_env(
    config: Any,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> EmptyCanvasEnv:
    return EmptyCanvasEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
