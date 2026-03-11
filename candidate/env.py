from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import torch

from simverse.core.env import SimEnv


BUY = 0
SELL = 1
REST = 2

COMPANY_COUNT = 5
CHANNEL_COUNT = 12
OBS_HEIGHT = 1
OBS_WIDTH = 25
HISTORY_WINDOW = OBS_WIDTH
INITIAL_CASH = 500.0
PRICE_SCALE = 40.0
HOLDINGS_SCALE = 24.0
TRADE_PENALTY = 0.0000
FLIP_PENALTY = 0.0005
RECENT_FLIP_WINDOW = 4
INVALID_ACTION_PENALTY = 0.0100
DENSE_REWARD_CLIP = 0.1200
TERMINAL_REWARD_CLIP = 1.0000
SUCCESS_MARGIN = 5.0


class TradingEnv(SimEnv):
    """Single-company trading task over five synthetic price regimes."""

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
        self.series_length = self.max_steps + HISTORY_WINDOW
        self.base_seed = int(getattr(config, "seed", 0) or 0)

        self._action_space = gym.spaces.Discrete(3)
        self._observation_space = gym.spaces.Box(
            low=-2.0,
            high=2.0,
            shape=(CHANNEL_COUNT, OBS_HEIGHT, OBS_WIDTH),
            dtype=float,
        )

        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer(
            "episode_return",
            torch.zeros(self.num_envs, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "episode_length",
            torch.zeros(self.num_envs, dtype=torch.int64, device=self.device),
        )
        self.register_buffer("cash", torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer("holdings", torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer(
            "avg_entry_price",
            torch.zeros(self.num_envs, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "last_trade_direction",
            torch.zeros(self.num_envs, dtype=torch.int64, device=self.device),
        )
        self.register_buffer(
            "steps_since_trade",
            torch.full(
                (self.num_envs,),
                fill_value=self.max_steps,
                dtype=torch.int64,
                device=self.device,
            ),
        )
        self.register_buffer(
            "episode_counter",
            torch.zeros(self.num_envs, dtype=torch.int64, device=self.device),
        )
        self.register_buffer("company_id", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer(
            "price_paths",
            torch.zeros(self.num_envs, self.series_length, dtype=self.dtype, device=self.device),
        )
        self.register_buffer(
            "history_offsets",
            torch.arange(HISTORY_WINDOW, dtype=torch.int64, device=self.device),
        )

        self.reset()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def describe(self) -> str:
        return (
            "A 100-hour synthetic trading task over five fictional company regimes "
            "with buy, sell, and rest actions."
        )

    def assign_agents(self, agents: list[Any]) -> None:
        self._assign_agents(agents, expected_count=1, label="TradingEnv")

    def reset(self) -> dict[str, torch.Tensor]:
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.done.zero_()
        self.steps.zero_()
        self.episode_return.zero_()
        self.episode_length.zero_()
        self.last_trade_direction.zero_()
        self._reset_subset(all_envs)
        return self._pack_observation_dict(self._build_observation())

    def step(
        self,
        actions: torch.Tensor | list[int] | tuple[int, ...] | dict[int, int] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        action_tensor = self._normalize_single_agent_actions(
            actions,
            missing_action=REST,
            dict_default=REST,
        )

        self.done.zero_()

        price_window = self._price_window()
        current_price = price_window[:, -1].clamp(min=1.0)
        next_price = self._next_price().clamp(min=1.0)

        valid_buy = (action_tensor == BUY) & (self.cash >= current_price)
        valid_sell = (action_tensor == SELL) & (self.holdings >= 1.0)
        valid_trade = valid_buy | valid_sell
        invalid_buy = (action_tensor == BUY) & (~valid_buy)
        invalid_sell = (action_tensor == SELL) & (~valid_sell)
        recent_trade = self.steps_since_trade <= RECENT_FLIP_WINDOW
        flip_trade = recent_trade & ((valid_buy & (self.last_trade_direction < 0)) | (
            valid_sell & (self.last_trade_direction > 0)
        ))

        buy_delta = valid_buy.to(dtype=self.dtype)
        sell_delta = valid_sell.to(dtype=self.dtype)
        updated_cash = self.cash - (buy_delta * current_price) + (sell_delta * current_price)
        updated_holdings = self.holdings + buy_delta - sell_delta
        buy_basis = torch.where(
            updated_holdings > 0.0,
            ((self.avg_entry_price * self.holdings) + (buy_delta * current_price))
            / updated_holdings.clamp(min=1.0),
            torch.zeros_like(self.avg_entry_price),
        )
        updated_avg_entry = torch.where(valid_buy, buy_basis, self.avg_entry_price)
        updated_avg_entry = torch.where(
            valid_sell & (updated_holdings <= 0.0),
            torch.zeros_like(updated_avg_entry),
            updated_avg_entry,
        )
        self.cash.copy_(updated_cash)
        self.holdings.copy_(updated_holdings)
        self.avg_entry_price.copy_(updated_avg_entry)

        portfolio_now = self.cash + (self.holdings * current_price)
        portfolio_next = self.cash + (self.holdings * next_price)

        reward = torch.clamp(
            (portfolio_next - portfolio_now) / INITIAL_CASH,
            min=-DENSE_REWARD_CLIP,
            max=DENSE_REWARD_CLIP,
        )
        reward -= valid_trade.to(dtype=self.dtype) * TRADE_PENALTY
        reward -= flip_trade.to(dtype=self.dtype) * FLIP_PENALTY
        reward -= invalid_buy.to(dtype=self.dtype) * INVALID_ACTION_PENALTY
        reward -= invalid_sell.to(dtype=self.dtype) * INVALID_ACTION_PENALTY

        next_trade_direction = self.last_trade_direction.clone()
        next_trade_direction = torch.where(
            valid_buy,
            torch.ones_like(next_trade_direction),
            next_trade_direction,
        )
        next_trade_direction = torch.where(
            valid_sell,
            -torch.ones_like(next_trade_direction),
            next_trade_direction,
        )
        self.last_trade_direction.copy_(next_trade_direction)
        next_steps_since_trade = torch.clamp(self.steps_since_trade + 1, max=self.max_steps)
        next_steps_since_trade = torch.where(
            valid_trade,
            torch.zeros_like(next_steps_since_trade),
            next_steps_since_trade,
        )
        self.steps_since_trade.copy_(next_steps_since_trade)

        self.steps.add_(1)
        self.episode_length.add_(1)
        done = self.steps >= self.max_steps
        self.done.copy_(done)

        final_value = self.cash + (self.holdings * next_price)
        terminal_reward = torch.clamp(
            (final_value - INITIAL_CASH) / INITIAL_CASH,
            min=-TERMINAL_REWARD_CLIP,
            max=TERMINAL_REWARD_CLIP,
        )
        reward = reward + torch.where(done, terminal_reward, torch.zeros_like(reward))
        reward = torch.clamp(reward, min=-1.0, max=1.0)

        success = done & (final_value >= (INITIAL_CASH + SUCCESS_MARGIN))

        self.episode_return.add_(reward)
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
            self.last_trade_direction.copy_(
                torch.where(
                    done,
                    torch.zeros_like(self.last_trade_direction),
                    self.last_trade_direction,
                )
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
        price_window = self._price_window()
        current_price = price_window[:, -1].clamp(min=1.0)
        price_history = torch.tanh(((price_window / current_price.unsqueeze(1)) - 1.0) * 6.0)
        return_history = torch.zeros_like(price_window)
        price_delta = (price_window[:, 1:] - price_window[:, :-1]) / price_window[:, :-1].clamp(min=1.0)
        return_history[:, 1:] = torch.tanh(price_delta * 12.0)

        obs = torch.zeros(
            (self.num_envs, CHANNEL_COUNT, OBS_HEIGHT, OBS_WIDTH),
            dtype=self.dtype,
            device=self.device,
        )
        obs[:, 0, 0] = price_history
        obs[:, 1, 0] = return_history

        recent_min = price_window.min(dim=1).values
        recent_max = price_window.max(dim=1).values
        recent_span = (recent_max - recent_min).clamp(min=1.0)
        short_trend = price_window[:, -6:].mean(dim=1)
        long_trend = price_window[:, -18:].mean(dim=1)

        price_plane = torch.clamp((current_price / PRICE_SCALE) - 0.75, min=-1.0, max=1.25)
        cash_plane = torch.clamp(self.cash / INITIAL_CASH, min=0.0, max=2.0)
        exposure_plane = torch.clamp((self.holdings * current_price) / INITIAL_CASH, min=0.0, max=2.0)
        holdings_plane = torch.clamp(self.holdings / HOLDINGS_SCALE, min=0.0, max=2.0)
        unrealized_plane = torch.clamp(
            (self.holdings * (current_price - self.avg_entry_price)) / INITIAL_CASH,
            min=-1.0,
            max=1.0,
        )
        buying_power_plane = torch.clamp(
            (self.cash / current_price) / HOLDINGS_SCALE,
            min=0.0,
            max=2.0,
        )
        range_position_plane = torch.clamp(
            (((current_price - recent_min) / recent_span) * 2.0) - 1.0,
            min=-1.0,
            max=1.0,
        )
        momentum_plane = torch.tanh(((short_trend / long_trend.clamp(min=1.0)) - 1.0) * 12.0)
        drawdown_plane = torch.clamp(
            ((current_price / recent_max.clamp(min=1.0)) - 1.0) * 6.0,
            min=-1.0,
            max=0.0,
        )
        time_plane = torch.clamp(
            (self.max_steps - self.steps).to(dtype=self.dtype) / float(self.max_steps),
            min=0.0,
            max=1.0,
        )
        obs[:, 2] = price_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 3] = cash_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 4] = exposure_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 5] = holdings_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 6] = unrealized_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 7] = buying_power_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 8] = range_position_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 9] = momentum_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 10] = drawdown_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        obs[:, 11] = time_plane.view(-1, 1, 1).expand(-1, OBS_HEIGHT, OBS_WIDTH)
        return obs

    def _price_window(self) -> torch.Tensor:
        indices = self.steps.unsqueeze(1) + self.history_offsets.unsqueeze(0)
        return torch.gather(self.price_paths, 1, indices)

    def _next_price(self) -> torch.Tensor:
        next_index = self.steps + HISTORY_WINDOW
        return torch.gather(self.price_paths, 1, next_index.unsqueeze(1)).squeeze(1)

    def _reset_subset(self, mask: torch.Tensor) -> None:
        indices = torch.nonzero(mask, as_tuple=False).flatten().cpu().tolist()
        if not indices:
            return

        for env_index in indices:
            self.episode_counter[env_index] += 1
            episode_seed = (
                self.base_seed
                + (env_index * 10007)
                + (int(self.episode_counter[env_index].item()) * 7919)
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(episode_seed)

            company_id = int((env_index + int(self.episode_counter[env_index].item())) % COMPANY_COUNT)
            price_path = self._generate_price_path(company_id, generator).to(
                device=self.device,
                dtype=self.dtype,
            )

            self.company_id[env_index] = company_id
            self.price_paths[env_index].copy_(price_path)
            self.cash[env_index] = INITIAL_CASH
            self.holdings[env_index] = 0.0
            self.avg_entry_price[env_index] = 0.0
            self.last_trade_direction[env_index] = 0
            self.steps_since_trade[env_index] = self.max_steps

    def _generate_price_path(self, company_id: int, generator: torch.Generator) -> torch.Tensor:
        # Each company family mixes one clean regime with mild seeded variation.
        t = torch.arange(self.series_length, dtype=torch.float32)
        phase = float(torch.rand((), generator=generator).item()) * (2.0 * math.pi)
        amplitude = 0.85 + (0.30 * float(torch.rand((), generator=generator).item()))
        drift = 0.85 + (0.30 * float(torch.rand((), generator=generator).item()))
        phase_jitter = float(torch.rand((), generator=generator).item()) - 0.5
        noise = torch.randn(self.series_length, generator=generator, dtype=torch.float32)

        if company_id == 0:
            base = 14.6 + (0.24 * drift * t) + (0.85 * amplitude * torch.sin((t / 8.6) + phase))
            price = base + (0.18 * noise)
        elif company_id == 1:
            price = torch.empty(self.series_length, dtype=torch.float32)
            family_phase = (0.30 * math.pi) + (0.45 * phase_jitter)
            mean_level = 18.8
            family_amplitude = 0.98 + (0.16 * amplitude)
            family_drift = 0.98 + (0.12 * drift)
            price[0] = mean_level - 0.4 + (0.28 * float(torch.randn((), generator=generator).item()))
            for index in range(1, self.series_length):
                anchor = mean_level + (0.052 * family_drift * index)
                anchor += 2.6 * family_amplitude * math.sin((index / 7.0) + family_phase)
                reversion = 0.54 * (anchor - float(price[index - 1].item()))
                shock = 0.12 * float(noise[index].item())
                price[index] = max(8.0, float(price[index - 1].item()) + reversion + shock)
            return price.clamp(min=6.0, max=80.0)
        elif company_id == 2:
            boom_shift = 1.5 * phase_jitter
            surge = 25.0 / (1.0 + torch.exp(-((t - (22.0 + boom_shift)) / 3.5)))
            fade = 22.5 / (1.0 + torch.exp(-((t - (61.0 + boom_shift)) / 4.5)))
            crest = 4.8 * torch.exp(-((t - (43.0 + boom_shift)) / 6.0) ** 2)
            base = 12.6 + surge - fade + crest
            base += 0.22 * amplitude * torch.sin((t / 6.4) + (0.20 * phase_jitter))
            price = base + (0.14 * noise)
        elif company_id == 3:
            cycle_phase = (0.72 * math.pi) + (0.30 * phase_jitter)
            family_amplitude = 0.98 + (0.14 * amplitude)
            family_drift = 0.98 + (0.12 * drift)
            base = 17.6 + (0.042 * family_drift * t)
            base += 7.4 * family_amplitude * torch.sin((t / 9.4) + cycle_phase)
            base += 2.4 * torch.sin((t / 18.8) + (0.4 * cycle_phase))
            price = base + (0.14 * noise)
        else:
            trap_shift = 2.6 * phase_jitter
            fakeouts = 1.1 * amplitude * torch.sin((t / 5.0) + (0.25 * math.pi) + (0.40 * phase_jitter))
            fakeouts += 0.45 * torch.sin((t / 2.3) + (0.18 * phase_jitter))
            traps = 1.9 * torch.exp(-((t - (34.0 + trap_shift)) / 5.6) ** 2)
            traps -= 3.0 * torch.exp(-((t - (58.0 + trap_shift)) / 6.8) ** 2)
            traps += 2.8 * torch.exp(-((t - (85.0 + trap_shift)) / 5.2) ** 2)
            base = 20.2 - (0.010 * drift * t) + fakeouts + traps
            price = base + (0.14 * noise)

        smoothed = price.clone()
        for index in range(1, self.series_length):
            smoothed[index] = (0.78 * smoothed[index - 1]) + (0.22 * price[index])
        return smoothed.clamp(min=6.0, max=80.0)

    def close(self) -> None:
        return None


CandidateEnv = TradingEnv


def create_env(
    config: Any,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TradingEnv:
    return TradingEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
