from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from simverse.core.env import SimEnv
from simverse.envs.maze_race.agent import MazeRaceAgent
from simverse.envs.maze_race.config import MazeRaceConfig


class MazeRaceTorchEnv(SimEnv):
    """Simple maze racing env. First agent to its goal wins."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_SPACE = gym.spaces.Discrete(5)

    WINNER_NONE = -1
    WINNER_DRAW = -2

    def __init__(
        self,
        config: MazeRaceConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents not in (1, 2):
            raise ValueError("MazeRaceTorchEnv supports 1 or 2 agents")

        self.num_envs = self._resolve_num_envs(num_envs, config)
        self.num_agents = self.config.num_agents
        self.width = int(self.config.width)
        self.height = int(self.config.height)
        if self.width < 7 or self.height < 7:
            raise ValueError("MazeRace requires width/height >= 7")

        self.agents: list[MazeRaceAgent] = []

        self.start_positions = [(1, 1)]
        if self.num_agents > 1:
            self.start_positions.append((self.width - 2, 1))
        self.goal_positions = [(self.width - 2, self.height - 2)]
        if self.num_agents > 1:
            self.goal_positions.append((1, self.height - 2))
        self.start0 = self.start_positions[0]
        self.start1 = self.start_positions[1] if self.num_agents > 1 else self.start_positions[0]
        self.goal0 = self.goal_positions[0]
        self.goal1 = self.goal_positions[1] if self.num_agents > 1 else self.goal_positions[0]

        self.register_buffer("walls", self._build_maze())
        self.register_buffer("wall_map", self.walls.to(self.dtype).unsqueeze(0))

        goal_maps = torch.zeros((self.num_agents, self.height, self.width), dtype=self.dtype)
        for idx, (gx, gy) in enumerate(self.goal_positions):
            goal_maps[idx, gy, gx] = 1.0

        self.register_buffer("goal_maps", goal_maps)

        self.register_buffer(
            "agent_pos", torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64)
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner", torch.full((self.num_envs,), self.WINNER_NONE, dtype=torch.int64)
        )
        self.obs_channels = 1 + 2 * self.num_agents
        self.register_buffer(
            "obs_buffer",
            torch.zeros(
                self.num_envs,
                self.obs_channels,
                self.height,
                self.width,
                dtype=self.dtype,
            ),
        )
        self.register_buffer("env_idx", torch.arange(self.num_envs, dtype=torch.int64))

        self.register_buffer("delta_x", torch.tensor([0, 0, 0, -1, 1], dtype=torch.int64))
        self.register_buffer("delta_y", torch.tensor([0, -1, 1, 0, 0], dtype=torch.int64))

        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.obs_channels, self.height, self.width),
            dtype=np.float32,
        )

    def assign_agents(self, agents: list[MazeRaceAgent]) -> None:
        self._assign_agents(agents, label="MazeRace")

    def reset(self) -> Dict[str, torch.Tensor]:
        for idx, (sx, sy) in enumerate(self.start_positions):
            self.agent_pos[:, idx, 0] = sx
            self.agent_pos[:, idx, 1] = sy
        self._reset_episode_state(winner_none=self.WINNER_NONE)
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = self._empty_rewards()
        active = ~self.done

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            action_idx = torch.clamp(action, min=0, max=4)
            has_action = action >= 0
            move_mask = active & has_action

            px = self.agent_pos[:, agent_id, 0]
            py = self.agent_pos[:, agent_id, 1]
            nx = torch.clamp(px + self.delta_x[action_idx], 0, self.width - 1)
            ny = torch.clamp(py + self.delta_y[action_idx], 0, self.height - 1)

            blocked = self.walls[ny, nx]
            can_move = move_mask & (~blocked)

            self.agent_pos[:, agent_id, 0] = torch.where(can_move, nx, px)
            self.agent_pos[:, agent_id, 1] = torch.where(can_move, ny, py)

        self.steps[active] += 1

        reached: list[torch.Tensor] = []
        for idx, (gx, gy) in enumerate(self.goal_positions):
            px = self.agent_pos[:, idx, 0]
            py = self.agent_pos[:, idx, 1]
            reached.append(active & (px == gx) & (py == gy))

        if self.num_agents == 1:
            reached0 = reached[0]
            if torch.any(reached0):
                rewards[reached0, 0] += float(self.config.win_reward)
                self.winner[reached0] = 0
            finished = reached0
        else:
            reached0, reached1 = reached
            both = reached0 & reached1
            only0 = reached0 & (~reached1)
            only1 = reached1 & (~reached0)

            if torch.any(only0):
                rewards[only0, 0] += float(self.config.win_reward)
                rewards[only0, 1] -= float(self.config.lose_penalty)
                self.winner[only0] = 0
            if torch.any(only1):
                rewards[only1, 1] += float(self.config.win_reward)
                rewards[only1, 0] -= float(self.config.lose_penalty)
                self.winner[only1] = 1
            if torch.any(both):
                rewards[both, :] += float(self.config.draw_reward)
                self.winner[both] = self.WINNER_DRAW

            finished = only0 | only1 | both

        timed_out = active & (self.steps >= int(self.config.max_steps))
        draw_timeout = timed_out & (~finished)
        if torch.any(draw_timeout):
            rewards[draw_timeout, :] += float(self.config.draw_reward)
            self.winner[draw_timeout] = self.WINNER_DRAW

        self.done |= finished | timed_out

        obs = self._get_observation()
        info = self._build_info()
        return obs, rewards, self.done.clone(), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        return self._normalize_action_matrix(actions)

    def _build_maze(self) -> torch.Tensor:
        walls = torch.zeros((self.height, self.width), dtype=torch.bool)

        walls[0, :] = True
        walls[self.height - 1, :] = True
        walls[:, 0] = True
        walls[:, self.width - 1] = True

        for x in range(3, self.width - 2, 3):
            walls[1 : self.height - 1, x] = True
            gap0 = 1 + ((2 * x + 1) % (self.height - 2))
            gap1 = 1 + ((3 * x + 2) % (self.height - 2))
            walls[gap0, x] = False
            walls[gap1, x] = False

        walls[self.start0[1], self.start0[0]] = False
        walls[self.start1[1], self.start1[0]] = False
        walls[self.goal0[1], self.goal0[0]] = False
        walls[self.goal1[1], self.goal1[0]] = False
        return walls

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        self.obs_buffer.zero_()

        self.obs_buffer[:, 0].copy_(self.wall_map.expand(self.num_envs, -1, -1))
        for idx in range(self.num_agents):
            goal_map = self.goal_maps[idx].unsqueeze(0).expand(self.num_envs, -1, -1)
            self.obs_buffer[:, 1 + idx].copy_(goal_map)
            self.obs_buffer[
                self.env_idx,
                1 + self.num_agents + idx,
                self.agent_pos[:, idx, 1],
                self.agent_pos[:, idx, 0],
            ] = 1.0

        return self._pack_observation_dict(self.obs_buffer)


MazeRaceEnv = MazeRaceTorchEnv


def create_env(
    config: MazeRaceConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> MazeRaceTorchEnv:
    return MazeRaceTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
