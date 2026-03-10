from __future__ import annotations

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
import torch
from simverse.core.env import SimEnv
from simverse.envs.farmtila.agent import FarmtilaAgent
from simverse.envs.farmtila.config import FarmtilaConfig


class FarmtilaTorchEnv(SimEnv):
    HARVEST_ACTION = 4
    PICKUP_ACTION = 5
    ACTION_SPACE = gym.spaces.Discrete(6)
    LAND_EMPTY = 0
    LAND_OWNED = 1

    def __init__(
        self,
        config: FarmtilaConfig,
        num_envs: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(device=device, dtype=dtype)
        self.config = config
        if self.config.num_agents != 2:
            raise ValueError("Competitive Farmtila requires exactly 2 agents")

        self.num_envs = self._resolve_num_envs(num_envs, config)
        self.width = config.width
        self.height = config.height
        self.num_agents = config.num_agents
        self.agents: list[FarmtilaAgent] = []

        self.register_buffer(
            "seed_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "owner_grid",
            torch.full(
                (self.num_envs, self.width, self.height),
                -1,
                dtype=torch.int64,
            ),
        )
        self.register_buffer(
            "farm_grid",
            torch.zeros(self.num_envs, self.width, self.height, dtype=torch.int64),
        )
        self.register_buffer(
            "agent_pos",
            torch.zeros(self.num_envs, self.num_agents, 2, dtype=torch.int64),
        )
        self.register_buffer(
            "inventory",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer(
            "harvested_tiles",
            torch.zeros(self.num_envs, self.num_agents, dtype=torch.int64),
        )
        self.register_buffer(
            "seeds_spawned",
            torch.zeros(self.num_envs, dtype=torch.int64),
        )
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer(
            "winner",
            torch.full((self.num_envs,), -1, dtype=torch.int64),
        )
        self.register_buffer(
            "delta_x",
            torch.tensor([0, 0, -1, 1, 0, 0], dtype=torch.int64),
        )
        self.register_buffer(
            "delta_y",
            torch.tensor([-1, 1, 0, 0, 0, 0], dtype=torch.int64),
        )
        self.register_buffer(
            "env_idx",
            torch.arange(self.num_envs, dtype=torch.int64),
        )
        self.to(self.device)

    @property
    def action_space(self):
        return self.ACTION_SPACE

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-1,
            high=max(self.num_agents, self.LAND_OWNED, 1),
            shape=(5, self.width, self.height),
            dtype=np.float32,
        )

    def assign_agents(self, agents: list[FarmtilaAgent]) -> None:
        self._assign_agents(agents, expected_count=2, label="Competitive Farmtila")

    def reset(self) -> Dict[str, torch.Tensor]:
        self.seed_grid.zero_()
        self.owner_grid.fill_(-1)
        self.farm_grid.zero_()
        self.inventory.zero_()
        self.harvested_tiles.zero_()
        self.seeds_spawned.zero_()
        self._reset_episode_state(winner_none=-1)
        self._spawn_agents()
        self._spawn_seeds_if_due(force=True)
        return self._get_observation()

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        action_tensor = self._normalize_actions(actions)
        rewards = self._empty_rewards()
        active_mask = ~self.done
        env_idx = self.env_idx

        step_cost = float(getattr(self.config, "step_cost", 0.0))
        if step_cost != 0.0:
            rewards += (-step_cost) * active_mask.unsqueeze(1).to(self.dtype)

        prev_score_delta = (self.harvested_tiles[:, 0] - self.harvested_tiles[:, 1]).to(self.dtype)

        for agent_id in range(self.num_agents):
            action = action_tensor[:, agent_id]
            has_action = action >= 0
            active_action = has_action & active_mask
            action_index = torch.clamp(action, min=0, max=5)

            dx = self.delta_x[action_index] * active_action.to(self.delta_x.dtype)
            dy = self.delta_y[action_index] * active_action.to(self.delta_y.dtype)

            pos_x = self.agent_pos[:, agent_id, 0]
            pos_y = self.agent_pos[:, agent_id, 1]
            new_x = torch.clamp(pos_x + dx, 0, self.width - 1)
            new_y = torch.clamp(pos_y + dy, 0, self.height - 1)

            pos_x = torch.where(active_action, new_x, pos_x)
            pos_y = torch.where(active_action, new_y, pos_y)
            self.agent_pos[:, agent_id, 0] = pos_x
            self.agent_pos[:, agent_id, 1] = pos_y

            pickup = active_action & (self.seed_grid[env_idx, pos_x, pos_y] > 0)
            pickup_idx = env_idx[pickup]
            self.seed_grid[pickup_idx, pos_x[pickup], pos_y[pickup]] = 0
            self.inventory[pickup, agent_id] += 1

            harvest_action = (action == self.HARVEST_ACTION) & active_action
            can_spend = harvest_action & (self.inventory[:, agent_id] > 0)
            owner = self.owner_grid[env_idx, pos_x, pos_y]
            target_is_other = can_spend & (owner != agent_id)
            target_idx = env_idx[target_is_other]
            if target_idx.numel() > 0:
                prev_owner = owner[target_is_other]
                self.inventory[target_is_other, agent_id] -= 1
                self.owner_grid[target_idx, pos_x[target_is_other], pos_y[target_is_other]] = (
                    agent_id
                )
                self.farm_grid[target_idx, pos_x[target_is_other], pos_y[target_is_other]] = (
                    self.LAND_OWNED
                )
                self.harvested_tiles[target_is_other, agent_id] += 1

                had_prev_owner = prev_owner >= 0
                if torch.any(had_prev_owner):
                    prev_owner_envs = target_idx[had_prev_owner]
                    prev_owner_ids = prev_owner[had_prev_owner]
                    self.harvested_tiles[prev_owner_envs, prev_owner_ids] = torch.clamp(
                        self.harvested_tiles[prev_owner_envs, prev_owner_ids] - 1,
                        min=0,
                    )

        self.steps[active_mask] += 1
        self._spawn_seeds_if_due()
        self._check_episode_end(rewards)

        score_delta_reward = float(getattr(self.config, "score_delta_reward", 1.0))
        if score_delta_reward != 0.0:
            score_delta = (self.harvested_tiles[:, 0] - self.harvested_tiles[:, 1]).to(self.dtype)
            delta_change = (score_delta - prev_score_delta) * score_delta_reward
            rewards[:, 0] += delta_change
            rewards[:, 1] -= delta_change

        obs = self._get_observation()
        info = self._build_info()
        return obs, rewards, self.done.clone(), info

    def _normalize_actions(self, actions: torch.Tensor | None) -> torch.Tensor:
        return self._normalize_action_matrix(actions)

    def _spawn_agents(self) -> None:
        positions = torch.stack(
            (
                torch.randint(
                    0,
                    self.width,
                    (self.num_envs, self.num_agents),
                    device=self.device,
                ),
                torch.randint(
                    0,
                    self.height,
                    (self.num_envs, self.num_agents),
                    device=self.device,
                ),
            ),
            dim=-1,
        )
        self.agent_pos.copy_(positions)

    def _spawn_seeds_if_due(self, *, force: bool = False) -> None:
        if self.config.spawn_seed_every <= 0 and not force:
            return
        due_mask = (force | ((self.steps % self.config.spawn_seed_every) == 0)) & (~self.done)
        due_env_indices = torch.nonzero(due_mask, as_tuple=True)[0]
        if due_env_indices.numel() == 0:
            return
        total_cells = self.width * self.height
        spawn_cap = min(int(self.config.seeds_per_spawn), total_cells)
        if spawn_cap <= 0:
            return
        budgets = torch.clamp(
            self.config.total_seeds_per_episode - self.seeds_spawned[due_env_indices],
            min=0,
        )
        spawn_counts = torch.clamp(budgets, max=spawn_cap)

        random_scores = torch.rand(
            (due_env_indices.shape[0], total_cells),
            device=self.device,
        )
        flat_indices = torch.topk(
            random_scores,
            k=spawn_cap,
            dim=1,
            largest=False,
        ).indices

        xs = flat_indices // self.height
        ys = flat_indices % self.height

        due_env_grid = due_env_indices.unsqueeze(1).expand(-1, spawn_cap)
        existing_seed = self.seed_grid[due_env_grid, xs, ys]
        existing_farm = self.farm_grid[due_env_grid, xs, ys]
        within_budget = torch.arange(spawn_cap, device=self.device).unsqueeze(
            0
        ) < spawn_counts.unsqueeze(1)
        place_mask = within_budget & (existing_seed == 0) & (existing_farm == 0)

        self.seed_grid[due_env_grid, xs, ys] = torch.where(
            place_mask,
            torch.ones_like(existing_seed),
            existing_seed,
        )
        self.seeds_spawned[due_env_indices] += place_mask.sum(dim=1)

    def _check_episode_end(self, rewards: torch.Tensor) -> None:
        max_steps_mask = self.steps >= self.config.max_steps

        budgets = self.config.total_seeds_per_episode - self.seeds_spawned
        no_budget = budgets <= 0
        no_seed_on_map = self.seed_grid.view(self.num_envs, -1).sum(dim=1) == 0
        no_inventory = self.inventory.sum(dim=1) == 0
        exhausted_mask = no_budget & no_seed_on_map & no_inventory

        end_mask = (~self.done) & (max_steps_mask | exhausted_mask)
        if not torch.any(end_mask):
            return

        score0 = self.harvested_tiles[:, 0]
        score1 = self.harvested_tiles[:, 1]
        winner_ids = torch.where(score0 > score1, 0, torch.where(score1 > score0, 1, -1))

        self.done |= end_mask
        self.winner = torch.where(end_mask, winner_ids, self.winner)

        terminal = float(getattr(self.config, "terminal_win_reward", 1.0))
        if terminal != 0.0:
            envs0 = self.env_idx[end_mask & (winner_ids == 0)]
            envs1 = self.env_idx[end_mask & (winner_ids == 1)]
            rewards[envs0, 0] += terminal
            rewards[envs0, 1] -= terminal
            rewards[envs1, 1] += terminal
            rewards[envs1, 0] -= terminal

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        agent_grid = torch.zeros(
            (self.num_envs, self.width, self.height),
            dtype=self.dtype,
            device=self.device,
        )
        inventory_grid = torch.zeros_like(agent_grid)

        env_idx = self.env_idx
        for agent_id in range(self.num_agents):
            x = self.agent_pos[:, agent_id, 0]
            y = self.agent_pos[:, agent_id, 1]
            agent_grid[env_idx, x, y] = float(agent_id + 1)
            inventory_grid[env_idx, x, y] = self.inventory[:, agent_id].to(self.dtype)

        obs = torch.stack(
            [
                self.seed_grid.to(self.dtype),
                self.owner_grid.to(self.dtype),
                self.farm_grid.to(self.dtype),
                agent_grid,
                inventory_grid,
            ],
            dim=1,
        )

        return self._pack_observation_dict(obs)


FarmtilaEnv = FarmtilaTorchEnv


def create_env(
    config: FarmtilaConfig,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> FarmtilaTorchEnv:
    return FarmtilaTorchEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
