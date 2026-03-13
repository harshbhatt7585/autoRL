from __future__ import annotations

from typing import Any

import gymnasium as gym
import torch

from simverse.core.env import SimEnv


BUY = 0
SELL = 1
REST = 2

COMPANY_COUNT = 5
OBS_CHANNELS = 8
OBS_HEIGHT = 5
OBS_WIDTH = 5
HISTORY_LEN = OBS_HEIGHT * OBS_WIDTH
INITIAL_CASH = 500.0
INVALID_ACTION_PENALTY = 0.02
CHURN_PENALTY = 0.003
TRADE_FEE = 0.001


class TradingPatternEnv(SimEnv):
    """Synthetic trading simulator over five company regime families."""

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
        self.seed = int(getattr(config, "seed", 0) or 0)

        self._action_space = gym.spaces.Discrete(3)
        self._observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH),
            dtype=float,
        )

        self._episode_counter = 0
        self._rng = torch.Generator(device="cpu")
        self._rng.manual_seed(self.seed)

        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool, device=self.device))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer("cash", torch.full((self.num_envs,), INITIAL_CASH, dtype=self.dtype, device=self.device))
        self.register_buffer("holdings", torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer("avg_cost", torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer("last_action", torch.full((self.num_envs,), REST, dtype=torch.int64, device=self.device))
        self.register_buffer("company_id", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))
        self.register_buffer(
            "price_paths",
            torch.zeros((self.num_envs, self.max_steps + 1), dtype=self.dtype, device=self.device),
        )
        self.register_buffer("price", torch.ones(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer(
            "buy_hold_final", torch.full((self.num_envs,), INITIAL_CASH, dtype=self.dtype, device=self.device)
        )
        self.register_buffer("episode_return", torch.zeros(self.num_envs, dtype=self.dtype, device=self.device))
        self.register_buffer("episode_length", torch.zeros(self.num_envs, dtype=torch.int64, device=self.device))

        self.reset()

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def describe(self) -> str:
        return (
            "Synthetic trading task with five fictional company regimes, "
            "actions buy/sell/rest, and 100-hour episodes."
        )

    def assign_agents(self, agents: list[Any]) -> None:
        self._assign_agents(agents, expected_count=1, label="TradingPatternEnv")

    def reset(self) -> dict[str, torch.Tensor]:
        mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_subset(mask)
        self.done.zero_()
        self.steps.zero_()
        self.episode_return.zero_()
        self.episode_length.zero_()
        return self._pack_observation_dict(self._build_observation())

    def step(
        self,
        actions: torch.Tensor | list[int] | tuple[int, ...] | dict[int, int] | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        action_tensor = self._normalize_single_agent_actions(actions, missing_action=REST, dict_default=REST)

        self.done.zero_()
        current_step = self.steps.clone()
        price_t = self._gather_price_at(current_step)
        price_next = self._gather_price_at(torch.clamp(current_step + 1, max=self.max_steps))
        portfolio_before = self.cash + self.holdings * price_t

        reward = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)

        can_buy = self.cash >= price_t
        buy_mask = action_tensor == BUY
        valid_buy = buy_mask & can_buy
        invalid_buy = buy_mask & ~can_buy

        can_sell = self.holdings > 0.0
        sell_mask = action_tensor == SELL
        valid_sell = sell_mask & can_sell
        invalid_sell = sell_mask & ~can_sell

        if bool(valid_buy.any().item()):
            buy_price = price_t[valid_buy]
            old_holdings = self.holdings[valid_buy]
            old_avg = self.avg_cost[valid_buy]
            self.cash[valid_buy] -= buy_price
            self.holdings[valid_buy] = old_holdings + 1.0
            self.avg_cost[valid_buy] = (old_avg * old_holdings + buy_price) / (old_holdings + 1.0)

        if bool(valid_sell.any().item()):
            self.cash[valid_sell] += price_t[valid_sell]
            self.holdings[valid_sell] -= 1.0
            sold_out = valid_sell & (self.holdings <= 0.0)
            self.avg_cost = torch.where(sold_out, torch.zeros_like(self.avg_cost), self.avg_cost)

        invalid_mask = invalid_buy | invalid_sell
        reward = reward - invalid_mask.to(self.dtype) * INVALID_ACTION_PENALTY

        traded = (action_tensor == BUY) | (action_tensor == SELL)
        reward = reward - traded.to(self.dtype) * TRADE_FEE

        churn_mask = traded & (self.last_action != REST) & (action_tensor != self.last_action)
        reward = reward - churn_mask.to(self.dtype) * CHURN_PENALTY

        trend = price_next - price_t
        trend_up = trend > 0.01
        trend_down = trend < -0.01
        trend_flat = ~(trend_up | trend_down)

        reward = reward + (trend_up & (action_tensor == BUY)).to(self.dtype) * 0.012
        reward = reward + (trend_down & (action_tensor == SELL)).to(self.dtype) * 0.012
        reward = reward + (trend_flat & (action_tensor == REST)).to(self.dtype) * 0.004
        reward = reward + (trend_up & (action_tensor == REST) & (self.holdings > 0.0)).to(self.dtype) * 0.006
        reward = reward + (trend_down & (action_tensor == REST) & (self.holdings <= 0.0)).to(self.dtype) * 0.004
        wrong_way = (trend_up & (action_tensor == SELL)) | (trend_down & (action_tensor == BUY))
        reward = reward - wrong_way.to(self.dtype) * 0.006

        portfolio_after = self.cash + self.holdings * price_next
        delta = (portfolio_after - portfolio_before) / INITIAL_CASH
        reward = reward + torch.clamp(delta * 3.0, min=-0.08, max=0.08)

        self.last_action.copy_(action_tensor)
        self.steps.add_(1)
        self.episode_length.add_(1)
        self.price.copy_(price_next)
        self.episode_return.add_(reward)

        done = self.steps >= self.max_steps
        self.done.copy_(done)

        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if bool(done.any().item()):
            terminal_value = self.cash + self.holdings * self.price
            terminal_bonus = torch.clamp((terminal_value - INITIAL_CASH) / INITIAL_CASH, min=-1.5, max=1.5)
            beat_hold = terminal_value > (self.buy_hold_final + 2.0)
            beat_cash = terminal_value > (INITIAL_CASH + 5.0)
            terminal_bonus = terminal_bonus + beat_hold.to(self.dtype) * 0.28 + beat_cash.to(self.dtype) * 0.08
            reward = reward + done.to(self.dtype) * terminal_bonus
            self.episode_return.add_(done.to(self.dtype) * terminal_bonus)
            success = done & (terminal_value > torch.maximum(self.buy_hold_final, torch.full_like(terminal_value, INITIAL_CASH)))

        finished_return = torch.where(done, self.episode_return, torch.zeros_like(self.episode_return))
        finished_length = torch.where(done, self.episode_length, torch.zeros_like(self.episode_length))

        if bool(done.any().item()):
            self._reset_subset(done)
            self.steps = torch.where(done, torch.zeros_like(self.steps), self.steps)
            self.episode_return = torch.where(done, torch.zeros_like(self.episode_return), self.episode_return)
            self.episode_length = torch.where(done, torch.zeros_like(self.episode_length), self.episode_length)
            self.last_action = torch.where(done, torch.full_like(self.last_action, REST), self.last_action)

        observation = self._pack_observation_dict(self._build_observation())
        info = self._build_info(
            extra={
                "success": success,
                "episode_return": finished_return,
                "episode_length": finished_length,
            }
        )
        return observation, reward.unsqueeze(1), done.clone(), info

    def _gather_price_at(self, step_idx: torch.Tensor) -> torch.Tensor:
        gather_idx = torch.clamp(step_idx, min=0, max=self.max_steps).unsqueeze(1)
        return torch.gather(self.price_paths, dim=1, index=gather_idx).squeeze(1)

    def _build_observation(self) -> torch.Tensor:
        obs = torch.zeros(
            (self.num_envs, OBS_CHANNELS, OBS_HEIGHT, OBS_WIDTH),
            dtype=self.dtype,
            device=self.device,
        )

        time_idx = torch.arange(HISTORY_LEN, device=self.device).unsqueeze(0)
        start_idx = torch.clamp(self.steps.unsqueeze(1) - (HISTORY_LEN - 1), min=0)
        hist_idx = torch.clamp(start_idx + time_idx, max=self.max_steps)
        price_hist = torch.gather(self.price_paths, 1, hist_idx)

        current_price = self.price.unsqueeze(1)
        rel = torch.clamp((price_hist / torch.clamp(current_price, min=1.0)) - 1.0, min=-0.2, max=0.2)
        rel = (rel + 0.2) / 0.4
        obs[:, 0] = rel.view(self.num_envs, OBS_HEIGHT, OBS_WIDTH)

        diff = torch.diff(price_hist, dim=1, prepend=price_hist[:, :1])
        momentum = torch.clamp(diff / torch.clamp(current_price, min=1.0), min=-0.08, max=0.08)
        momentum = (momentum + 0.08) / 0.16
        obs[:, 1] = momentum.view(self.num_envs, OBS_HEIGHT, OBS_WIDTH)

        vol = torch.clamp(torch.abs(diff) / torch.clamp(current_price, min=1.0), min=0.0, max=0.12) / 0.12
        obs[:, 2] = vol.view(self.num_envs, OBS_HEIGHT, OBS_WIDTH)

        cash_plane = torch.clamp(self.cash / INITIAL_CASH, min=0.0, max=2.0) / 2.0
        holdings_plane = torch.clamp(self.holdings / 10.0, min=0.0, max=1.0)
        time_remaining = torch.clamp((self.max_steps - self.steps).to(self.dtype) / float(self.max_steps), 0.0, 1.0)
        unrealized = (self.price - self.avg_cost) * self.holdings / INITIAL_CASH
        unrealized_plane = (torch.tanh(unrealized * 4.0) + 1.0) / 2.0
        company_plane = self.company_id.to(self.dtype) / float(max(1, COMPANY_COUNT - 1))

        obs[:, 3] = cash_plane.view(-1, 1, 1)
        obs[:, 4] = holdings_plane.view(-1, 1, 1)
        obs[:, 5] = time_remaining.view(-1, 1, 1)
        obs[:, 6] = unrealized_plane.view(-1, 1, 1)
        obs[:, 7] = company_plane.view(-1, 1, 1)
        return obs

    def _reset_subset(self, mask: torch.Tensor) -> None:
        reset_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not reset_indices:
            return

        for env_idx in reset_indices:
            company = int(torch.randint(0, COMPANY_COUNT, (1,), generator=self._rng).item())
            prices = self._generate_company_path(company=company, env_idx=env_idx)
            self.company_id[env_idx] = company
            self.price_paths[env_idx] = prices
            self.price[env_idx] = prices[0]
            self.cash[env_idx] = INITIAL_CASH
            self.holdings[env_idx] = 0.0
            self.avg_cost[env_idx] = 0.0
            self.last_action[env_idx] = REST

            init_price = prices[0]
            buy_units = torch.floor(torch.tensor(INITIAL_CASH, dtype=self.dtype, device=self.device) / init_price)
            rem_cash = INITIAL_CASH - buy_units * init_price
            self.buy_hold_final[env_idx] = rem_cash + buy_units * prices[self.max_steps]
            self._episode_counter += 1

    def _generate_company_path(self, *, company: int, env_idx: int) -> torch.Tensor:
        time = torch.arange(self.max_steps + 1, device=self.device, dtype=self.dtype)
        phase = ((self._episode_counter + env_idx * 11 + company * 7) % 31) / 31.0

        if company == 0:
            base = 35.0 + 0.26 * time + 2.0 * torch.sin((time / 14.0) + phase)
        elif company == 1:
            center = 62.0 + 1.3 * torch.sin((time / 24.0) + phase * 2.5)
            base = center + 5.0 * torch.sin((time / 5.0) + phase * 4.0) * torch.exp(-time / 180.0)
        elif company == 2:
            boom = 16.0 * torch.exp(-((time - 34.0) ** 2) / 260.0)
            bust = -20.0 * torch.exp(-((time - 71.0) ** 2) / 180.0)
            drift = 52.0 + 0.06 * time
            base = drift + boom + bust + 2.5 * torch.sin((time / 3.6) + phase)
        elif company == 3:
            base = 55.0 + 8.5 * torch.sin((time / 8.3) + phase) + 2.4 * torch.sin((time / 18.0) + phase * 1.8)
        else:
            base = 48.0 + 0.03 * time + 2.8 * torch.sin((time / 6.0) + phase * 6.0)

        noise_scale = 1.2 if company in (2, 4) else 0.8
        noise = torch.randn(self.max_steps + 1, generator=self._rng, dtype=self.dtype)
        if self.device.type != "cpu":
            noise = noise.to(self.device)
        smooth_noise = torch.cumsum(noise * noise_scale * 0.12, dim=0)
        prices = torch.clamp(base + smooth_noise, min=5.0)
        return prices

    def close(self) -> None:
        return None


CandidateEnv = TradingPatternEnv


def create_env(
    config: Any,
    *,
    num_envs: int | None = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> TradingPatternEnv:
    return TradingPatternEnv(config=config, num_envs=num_envs, device=device, dtype=dtype)
