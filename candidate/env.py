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
CHANNEL_COUNT = 7
START_X = 0
START_Y = 2
KEY_X = 1
KEY_Y = 2
DOOR_X = 2
DOOR_Y = 2
GOAL_X = 4
GOAL_Y = 2


class KeyDoorSimverseEnv(SimEnv):
    """Baseline Simverse environment. This file is part of the mutable candidate."""

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

        wall_template = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=self.dtype, device=self.device)
        wall_template[:, DOOR_X] = 1.0
        wall_template[DOOR_Y, DOOR_X] = 0.0
        self.register_buffer("wall_template", wall_template)

        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("agent_x", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("agent_y", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("has_key", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.register_buffer("door_open", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
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
        return "A 5x5 key-door corridor: grab the key, pass the center gate, then finish on the far side."

    def assign_agents(self, agents: list[Any]) -> None:
        self._assign_agents(agents, expected_count=1, label="KeyDoorSimverseEnv")

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

        reward = torch.full((self.num_envs,), -0.005, dtype=self.dtype, device=self.device)
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        target_x, target_y = self._current_target()
        move_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        delta_x = torch.tensor([0, 0, -1, 1], dtype=torch.int64, device=self.device)
        delta_y = torch.tensor([-1, 1, 0, 0], dtype=torch.int64, device=self.device)
        current_x = self.agent_x.clone()
        current_y = self.agent_y.clone()
        raw_x = current_x + delta_x[action_tensor]
        raw_y = current_y + delta_y[action_tensor]

        oob = (raw_x < 0) | (raw_x >= GRID_SIZE) | (raw_y < 0) | (raw_y >= GRID_SIZE)
        door_cell = (raw_x == DOOR_X) & (raw_y == DOOR_Y)
        barrier = raw_x == DOOR_X
        blocked = barrier & ~door_cell
        blocked = blocked | (door_cell & ~self.door_open & ~self.has_key)
        invalid_move = move_mask & (oob | blocked)
        valid_move = move_mask & ~invalid_move

        clipped_x = torch.clamp(raw_x, 0, GRID_SIZE - 1)
        clipped_y = torch.clamp(raw_y, 0, GRID_SIZE - 1)
        self.agent_x = torch.where(valid_move, clipped_x, self.agent_x)
        self.agent_y = torch.where(valid_move, clipped_y, self.agent_y)

        before_distance = (current_x - target_x).abs() + (current_y - target_y).abs()
        moved_distance = (self.agent_x - target_x).abs() + (self.agent_y - target_y).abs()
        shaping = (before_distance - moved_distance).to(dtype=self.dtype) * 0.06
        reward = reward + torch.where(valid_move, shaping, torch.zeros_like(shaping))
        reward = reward - invalid_move.to(dtype=self.dtype) * 0.03

        on_key = (
            valid_move
            & ~self.has_key
            & (self.agent_x == KEY_X)
            & (self.agent_y == KEY_Y)
        )
        self.has_key = self.has_key | on_key
        reward = reward + on_key.to(dtype=self.dtype) * 0.30

        opened_door = (
            valid_move
            & self.has_key
            & ~self.door_open
            & (self.agent_x == DOOR_X)
            & (self.agent_y == DOOR_Y)
        )
        self.door_open = self.door_open | opened_door
        reward = reward + opened_door.to(dtype=self.dtype) * 0.40

        success = self.door_open & (self.agent_x == GOAL_X) & (self.agent_y == GOAL_Y)
        reward = reward + success.to(dtype=self.dtype) * 1.00
        reward = torch.clamp(reward, -1.0, 1.0)

        self.episode_return.add_(reward)
        done = success | (self.steps >= self.max_steps)
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
        reward_tensor = reward.unsqueeze(1)
        info = self._build_info(
            extra={
                "success": success.clone(),
                "episode_return": finished_return,
                "episode_length": finished_length,
            }
        )
        return observation, reward_tensor, done.clone(), info

    def _current_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        target_x = torch.full_like(self.agent_x, GOAL_X)
        target_y = torch.full_like(self.agent_y, GOAL_Y)

        need_key = ~self.has_key
        need_door = self.has_key & ~self.door_open

        target_x = torch.where(need_key, torch.full_like(target_x, KEY_X), target_x)
        target_y = torch.where(need_key, torch.full_like(target_y, KEY_Y), target_y)
        target_x = torch.where(need_door, torch.full_like(target_x, DOOR_X), target_x)
        target_y = torch.where(need_door, torch.full_like(target_y, DOOR_Y), target_y)
        return target_x, target_y

    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros(
            (self.num_envs, CHANNEL_COUNT, GRID_SIZE, GRID_SIZE),
            dtype=self.dtype,
            device=self.device,
        )
        obs[:, 0] = self.wall_template
        closed_door = ~self.door_open
        obs[closed_door, 0, DOOR_Y, DOOR_X] = 1.0

        batch_index = torch.arange(self.num_envs, device=self.device)
        obs[batch_index, 1, self.agent_y, self.agent_x] = 1.0

        key_visible = ~self.has_key
        obs[key_visible, 2, KEY_Y, KEY_X] = 1.0
        obs[:, 3, DOOR_Y, DOOR_X] = 1.0
        obs[:, 4, GOAL_Y, GOAL_X] = 1.0

        has_key_plane = self.has_key.to(dtype=self.dtype).view(self.num_envs, 1, 1)
        door_open_plane = self.door_open.to(dtype=self.dtype).view(self.num_envs, 1, 1)
        obs[:, 5] = has_key_plane.expand(-1, GRID_SIZE, GRID_SIZE)
        obs[:, 6] = door_open_plane.expand(-1, GRID_SIZE, GRID_SIZE)
        return obs

    def _reset_subset(self, mask: torch.Tensor) -> None:
        self.agent_x.copy_(torch.where(mask, torch.full_like(self.agent_x, START_X), self.agent_x))
        self.agent_y.copy_(torch.where(mask, torch.full_like(self.agent_y, START_Y), self.agent_y))
        self.has_key.copy_(torch.where(mask, torch.zeros_like(self.has_key), self.has_key))
        self.door_open.copy_(torch.where(mask, torch.zeros_like(self.door_open), self.door_open))

    def close(self) -> None:
        return None


CandidateEnv = KeyDoorSimverseEnv


def create_env(
    config: Any,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> KeyDoorSimverseEnv:
    return KeyDoorSimverseEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
