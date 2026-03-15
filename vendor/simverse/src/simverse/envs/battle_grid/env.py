from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from simverse.core.env import SimEnv
from simverse.envs.battle_grid.agent import BattleGridAgent
from simverse.envs.battle_grid.config import BattleGridConfig


class BattleGridTorchEnv(SimEnv):
    """Two-agent PvP grid world with movement and melee attacks."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_ATTACK = 5
    ACTION_SPACE = gym.spaces.Discrete(6)

    WINNER_NONE = -1
    WINNER_DRAW = -2

    def __init__(
        self,
        config: BattleGridConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if int(self.config.num_agents) != 2:
            raise ValueError("BattleGridTorchEnv requires exactly 2 agents")

        self.num_envs = self._resolve_num_envs(num_envs, config)
        self.num_agents = 2
        self.width = int(self.config.width)
        self.height = int(self.config.height)
        if self.width < 3 or self.height < 3:
            raise ValueError("BattleGrid requires width/height >= 3")

        self.max_steps = max(1, int(self.config.max_steps))
        self.max_health = max(1, int(self.config.max_health))
        self.attack_damage = max(1, int(self.config.attack_damage))
        self.attack_range = max(1, int(self.config.attack_range))

        self.agents: list[BattleGridAgent] = []

        self.register_buffer(
            "agent_pos",
            torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64),
        )
        self.register_buffer(
            "health",
            torch.full((self.num_envs, self.num_agents), self.max_health, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), self.WINNER_NONE, dtype=torch.int64),
        )
        self.register_buffer("env_idx", torch.arange(self.num_envs, dtype=torch.int64))

        self.obs_channels = 2
        self.feature_dim = 3
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
        self.register_buffer(
            "feature_buffer",
            torch.zeros(self.num_envs, self.feature_dim, dtype=self.dtype),
        )
        self.register_buffer(
            "last_obs_agent_pos",
            torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64),
        )
        self.register_buffer(
            "last_obs_agent_visible",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.bool),
        )

        self.register_buffer("delta_x", torch.tensor([0, 0, 0, -1, 1, 0], dtype=torch.int64))
        self.register_buffer("delta_y", torch.tensor([0, -1, 1, 0, 0, 0], dtype=torch.int64))

        self._observation_space = gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.obs_channels, self.height, self.width),
                    dtype=np.float32,
                ),
                "features": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.feature_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return self._observation_space

    def assign_agents(self, agents: list[BattleGridAgent]) -> None:
        self._assign_agents(agents, label="BattleGrid")

    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_episode_state(winner_none=self.WINNER_NONE)
        self.health.fill_(self.max_health)
        self._spawn_unique_positions()
        self._reset_observation_cache()
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = self._empty_rewards()
        active = ~self.done

        alive_before0 = self.health[:, 0] > 0
        alive_before1 = self.health[:, 1] > 0
        current_x = [self.agent_pos[:, agent_id, 0] for agent_id in range(self.num_agents)]
        current_y = [self.agent_pos[:, agent_id, 1] for agent_id in range(self.num_agents)]
        next_x: list[torch.Tensor] = []
        next_y: list[torch.Tensor] = []

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            action_idx = torch.clamp(action, min=0, max=self.ACTION_ATTACK)
            alive_before = alive_before0 if agent_id == 0 else alive_before1
            can_act = active & alive_before & (action >= 0)
            can_move = can_act & (action <= self.ACTION_RIGHT)

            px = current_x[agent_id]
            py = current_y[agent_id]
            nx = torch.clamp(px + self.delta_x[action_idx], 0, self.width - 1)
            ny = torch.clamp(py + self.delta_y[action_idx], 0, self.height - 1)
            next_x.append(torch.where(can_move, nx, px))
            next_y.append(torch.where(can_move, ny, py))

        same_destination = (
            active
            & alive_before0
            & alive_before1
            & (next_x[0] == next_x[1])
            & (next_y[0] == next_y[1])
        )
        swapped_positions = (
            active
            & alive_before0
            & alive_before1
            & (next_x[0] == current_x[1])
            & (next_y[0] == current_y[1])
            & (next_x[1] == current_x[0])
            & (next_y[1] == current_y[0])
        )
        blocked = same_destination | swapped_positions
        if torch.any(blocked):
            next_x[0] = torch.where(blocked, current_x[0], next_x[0])
            next_y[0] = torch.where(blocked, current_y[0], next_y[0])
            next_x[1] = torch.where(blocked, current_x[1], next_x[1])
            next_y[1] = torch.where(blocked, current_y[1], next_y[1])

        self.agent_pos[:, 0, 0] = next_x[0]
        self.agent_pos[:, 0, 1] = next_y[0]
        self.agent_pos[:, 1, 0] = next_x[1]
        self.agent_pos[:, 1, 1] = next_y[1]

        rewards[active, :] -= float(self.config.step_penalty)
        self.steps[active] += 1

        x0 = self.agent_pos[:, 0, 0]
        y0 = self.agent_pos[:, 0, 1]
        x1 = self.agent_pos[:, 1, 0]
        y1 = self.agent_pos[:, 1, 1]
        distance = torch.abs(x0 - x1) + torch.abs(y0 - y1)
        in_range = distance <= self.attack_range

        attack0 = active & (self.health[:, 0] > 0) & (action_tensor[:, 0] == self.ACTION_ATTACK)
        attack1 = active & (self.health[:, 1] > 0) & (action_tensor[:, 1] == self.ACTION_ATTACK)
        hit0 = attack0 & (self.health[:, 1] > 0) & in_range
        hit1 = attack1 & (self.health[:, 0] > 0) & in_range

        damage_to_1 = hit0.to(torch.int64) * self.attack_damage
        damage_to_0 = hit1.to(torch.int64) * self.attack_damage

        self.health[:, 1] = torch.clamp(
            self.health[:, 1] - damage_to_1,
            min=0,
            max=self.max_health,
        )
        self.health[:, 0] = torch.clamp(
            self.health[:, 0] - damage_to_0,
            min=0,
            max=self.max_health,
        )

        damage_reward = float(self.config.damage_reward)
        if damage_reward != 0.0:
            rewards[:, 0] += damage_to_1.to(self.dtype) * damage_reward
            rewards[:, 1] -= damage_to_1.to(self.dtype) * damage_reward
            rewards[:, 1] += damage_to_0.to(self.dtype) * damage_reward
            rewards[:, 0] -= damage_to_0.to(self.dtype) * damage_reward

        death0 = active & alive_before0 & (self.health[:, 0] <= 0)
        death1 = active & alive_before1 & (self.health[:, 1] <= 0)
        finished = death0 | death1

        kill_reward = float(self.config.kill_reward)
        death_penalty = float(self.config.death_penalty)
        kill_by0 = death1 & hit0
        kill_by1 = death0 & hit1
        if kill_reward != 0.0:
            rewards[kill_by0, 0] += kill_reward
            rewards[kill_by1, 1] += kill_reward
        if death_penalty != 0.0:
            rewards[kill_by0, 1] -= death_penalty
            rewards[kill_by1, 0] -= death_penalty

        only0 = death1 & (~death0)
        only1 = death0 & (~death1)
        both = death0 & death1
        if torch.any(only0):
            self.winner[only0] = 0
        if torch.any(only1):
            self.winner[only1] = 1
        if torch.any(both):
            self.winner[both] = self.WINNER_DRAW

        timed_out = active & (self.steps >= self.max_steps) & (~finished)
        if torch.any(timed_out):
            hp0 = self.health[:, 0]
            hp1 = self.health[:, 1]
            timeout0 = timed_out & (hp0 > hp1)
            timeout1 = timed_out & (hp1 > hp0)
            timeout_draw = timed_out & (~timeout0) & (~timeout1)

            win_reward = float(self.config.timeout_win_reward)
            lose_penalty = float(self.config.timeout_lose_penalty)
            draw_reward = float(self.config.draw_reward)

            if win_reward != 0.0:
                rewards[timeout0, 0] += win_reward
                rewards[timeout1, 1] += win_reward
            if lose_penalty != 0.0:
                rewards[timeout0, 1] -= lose_penalty
                rewards[timeout1, 0] -= lose_penalty
            if draw_reward != 0.0:
                rewards[timeout_draw, :] += draw_reward

            self.winner[timeout0] = 0
            self.winner[timeout1] = 1
            self.winner[timeout_draw] = self.WINNER_DRAW

        self.done |= finished | timed_out

        obs = self._get_observation()
        info = self._build_info(extra={"health": self.health})
        return obs, rewards, self._payload_value(self.done), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        return self._normalize_action_matrix(actions)

    def _spawn_unique_positions(self) -> None:
        x0 = torch.randint(0, self.width, (self.num_envs,), dtype=torch.int64, device=self.device)
        y0 = torch.randint(0, self.height, (self.num_envs,), dtype=torch.int64, device=self.device)

        x1 = torch.randint(0, self.width, (self.num_envs,), dtype=torch.int64, device=self.device)
        y1 = torch.randint(0, self.height, (self.num_envs,), dtype=torch.int64, device=self.device)

        overlap = (x0 == x1) & (y0 == y1)
        while torch.any(overlap):
            count = int(overlap.sum().item())
            x1[overlap] = torch.randint(
                0, self.width, (count,), dtype=torch.int64, device=self.device
            )
            y1[overlap] = torch.randint(
                0, self.height, (count,), dtype=torch.int64, device=self.device
            )
            overlap = (x0 == x1) & (y0 == y1)

        self.agent_pos[:, 0, 0] = x0
        self.agent_pos[:, 0, 1] = y0
        self.agent_pos[:, 1, 0] = x1
        self.agent_pos[:, 1, 1] = y1

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        previous_env_idx, previous_agent_idx = torch.nonzero(
            self.last_obs_agent_visible,
            as_tuple=True,
        )
        if previous_env_idx.numel() > 0:
            previous_y = self.last_obs_agent_pos[previous_env_idx, previous_agent_idx, 1]
            previous_x = self.last_obs_agent_pos[previous_env_idx, previous_agent_idx, 0]
            self.obs_buffer[previous_env_idx, previous_agent_idx, previous_y, previous_x] = 0.0

        alive = self.health > 0
        visible_env_idx, visible_agent_idx = torch.nonzero(alive, as_tuple=True)
        if visible_env_idx.numel() > 0:
            current_y = self.agent_pos[visible_env_idx, visible_agent_idx, 1]
            current_x = self.agent_pos[visible_env_idx, visible_agent_idx, 0]
            self.obs_buffer[visible_env_idx, visible_agent_idx, current_y, current_x] = 1.0

        self.last_obs_agent_pos.copy_(self.agent_pos)
        self.last_obs_agent_visible.copy_(alive)

        self.feature_buffer[:, 0] = self.health[:, 0].to(self.dtype) / float(self.max_health)
        self.feature_buffer[:, 1] = self.health[:, 1].to(self.dtype) / float(self.max_health)
        self.feature_buffer[:, 2] = self.steps.to(self.dtype) / float(self.max_steps)

        return self._pack_observation_dict(
            self.obs_buffer,
            extra={
                "features": self.feature_buffer,
                "health": self.health,
            },
        )

    def _reset_observation_cache(self) -> None:
        self.obs_buffer.zero_()
        self.last_obs_agent_pos.zero_()
        self.last_obs_agent_visible.zero_()


BattleGridEnv = BattleGridTorchEnv


def create_env(
    config: BattleGridConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BattleGridTorchEnv:
    return BattleGridTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
