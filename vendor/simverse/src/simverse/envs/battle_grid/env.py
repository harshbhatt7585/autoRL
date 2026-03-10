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

        self.obs_channels = 5
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

        self.register_buffer("delta_x", torch.tensor([0, 0, 0, -1, 1, 0], dtype=torch.int64))
        self.register_buffer("delta_y", torch.tensor([0, -1, 1, 0, 0, 0], dtype=torch.int64))

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

    def assign_agents(self, agents: list[BattleGridAgent]) -> None:
        self._assign_agents(agents, label="BattleGrid")

    def reset(self) -> Dict[str, torch.Tensor]:
        self._reset_episode_state(winner_none=self.WINNER_NONE)
        self.health.fill_(self.max_health)
        self._spawn_unique_positions()
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = self._empty_rewards()
        active = ~self.done

        old_pos = self.agent_pos.clone()
        proposed_pos = self.agent_pos.clone()

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            action_idx = torch.clamp(action, min=0, max=self.ACTION_ATTACK)
            has_action = action >= 0
            alive = self.health[:, agent_id] > 0
            can_act = active & alive & has_action
            can_move = can_act & (action <= self.ACTION_RIGHT)

            px = old_pos[:, agent_id, 0]
            py = old_pos[:, agent_id, 1]
            nx = torch.clamp(px + self.delta_x[action_idx], 0, self.width - 1)
            ny = torch.clamp(py + self.delta_y[action_idx], 0, self.height - 1)

            proposed_pos[:, agent_id, 0] = torch.where(can_move, nx, px)
            proposed_pos[:, agent_id, 1] = torch.where(can_move, ny, py)

        alive0 = self.health[:, 0] > 0
        alive1 = self.health[:, 1] > 0
        p0x = proposed_pos[:, 0, 0]
        p0y = proposed_pos[:, 0, 1]
        p1x = proposed_pos[:, 1, 0]
        p1y = proposed_pos[:, 1, 1]
        o0x = old_pos[:, 0, 0]
        o0y = old_pos[:, 0, 1]
        o1x = old_pos[:, 1, 0]
        o1y = old_pos[:, 1, 1]

        same_destination = active & alive0 & alive1 & (p0x == p1x) & (p0y == p1y)
        swapped_positions = (
            active & alive0 & alive1 & (p0x == o1x) & (p0y == o1y) & (p1x == o0x) & (p1y == o0y)
        )
        blocked = same_destination | swapped_positions
        if torch.any(blocked):
            proposed_pos[blocked, 0, :] = old_pos[blocked, 0, :]
            proposed_pos[blocked, 1, :] = old_pos[blocked, 1, :]

        self.agent_pos.copy_(proposed_pos)

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

        health_before = self.health.clone()
        self.health[:, 1] = torch.clamp(self.health[:, 1] - damage_to_1, min=0, max=self.max_health)
        self.health[:, 0] = torch.clamp(self.health[:, 0] - damage_to_0, min=0, max=self.max_health)

        damage_reward = float(self.config.damage_reward)
        if damage_reward != 0.0:
            rewards[:, 0] += damage_to_1.to(self.dtype) * damage_reward
            rewards[:, 1] -= damage_to_1.to(self.dtype) * damage_reward
            rewards[:, 1] += damage_to_0.to(self.dtype) * damage_reward
            rewards[:, 0] -= damage_to_0.to(self.dtype) * damage_reward

        death0 = active & (health_before[:, 0] > 0) & (self.health[:, 0] <= 0)
        death1 = active & (health_before[:, 1] > 0) & (self.health[:, 1] <= 0)
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
        info = self._build_info(extra={"health": self.health.clone()})
        return obs, rewards, self.done.clone(), info

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
        self.obs_buffer.zero_()

        alive0 = self.health[:, 0] > 0
        alive1 = self.health[:, 1] > 0

        if torch.any(alive0):
            idx0 = self.env_idx[alive0]
            self.obs_buffer[
                idx0,
                0,
                self.agent_pos[alive0, 0, 1],
                self.agent_pos[alive0, 0, 0],
            ] = 1.0
        if torch.any(alive1):
            idx1 = self.env_idx[alive1]
            self.obs_buffer[
                idx1,
                1,
                self.agent_pos[alive1, 1, 1],
                self.agent_pos[alive1, 1, 0],
            ] = 1.0

        hp0 = (self.health[:, 0].to(self.dtype) / float(self.max_health)).view(self.num_envs, 1, 1)
        hp1 = (self.health[:, 1].to(self.dtype) / float(self.max_health)).view(self.num_envs, 1, 1)
        step_progress = (self.steps.to(self.dtype) / float(self.max_steps)).view(
            self.num_envs, 1, 1
        )

        self.obs_buffer[:, 2].copy_(hp0.expand(-1, self.height, self.width))
        self.obs_buffer[:, 3].copy_(hp1.expand(-1, self.height, self.width))
        self.obs_buffer[:, 4].copy_(step_progress.expand(-1, self.height, self.width))

        return self._pack_observation_dict(self.obs_buffer, extra={"health": self.health.clone()})


BattleGridEnv = BattleGridTorchEnv


def create_env(
    config: BattleGridConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> BattleGridTorchEnv:
    return BattleGridTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
