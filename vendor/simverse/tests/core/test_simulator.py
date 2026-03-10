from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from simverse.core.agent import SimAgent
from simverse.core.env import SimEnv
from simverse.core.simulator import Simulator


class NoOpTrainer:
    def train(self, *args, **kwargs) -> None:
        del args, kwargs


class DummyAgent(SimAgent):
    def __init__(self, agent_id: int, action_space: np.ndarray, policy: nn.Module | None) -> None:
        self.agent_id = agent_id
        super().__init__(name=f"agent_{agent_id}", action_space=action_space, policy=policy)

    def action(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return np.zeros((1,), dtype=np.int64)

    def info(self) -> dict:
        return {}

    def reset(self) -> None:
        return None

    def get_action_space(self) -> np.ndarray:
        return self.action_space

    def get_memory(self) -> dict:
        return {}

    def current_state(self) -> np.ndarray:
        return np.zeros((1,), dtype=np.float32)

    def get_policy(self) -> nn.Module | None:
        return self.policy

    def set_policy(self, policy: nn.Module | None) -> None:
        self.policy = policy


class ConstantPolicy(nn.Module):
    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(obs.shape[0]) if obs.ndim > 0 else 1
        logits = torch.zeros((batch_size, 1), dtype=torch.float32, device=obs.device)
        value = torch.zeros((batch_size, 1), dtype=torch.float32, device=obs.device)
        return logits, value


class TorchMultiAgentEnv(SimEnv):
    def __init__(self) -> None:
        super().__init__(device="cpu", dtype=torch.float32)
        self.num_envs = 2
        self.num_agents = 2
        self.config = SimpleNamespace(max_steps=4)
        self._observation_space = SimpleNamespace(shape=(1, 2, 2))
        self._action_space = SimpleNamespace(n=1)
        self.agents = []
        self.actions_seen: list[torch.Tensor] = []
        self.register_buffer("done", torch.zeros(self.num_envs, dtype=torch.bool))
        self.register_buffer("steps", torch.zeros(self.num_envs, dtype=torch.int64))

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    def assign_agents(self, agents: list[DummyAgent]) -> None:
        self.agents = list(agents)

    def reset(self) -> dict[str, torch.Tensor]:
        self.done.zero_()
        self.steps.zero_()
        return {"obs": torch.zeros((self.num_envs, 1, 2, 2), dtype=self.dtype, device=self.device)}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, int]]:
        self.actions_seen.append(actions.clone())
        self.steps += 1
        self.done = self.steps >= 2
        obs = {"obs": torch.ones((self.num_envs, 1, 2, 2), dtype=self.dtype, device=self.device)}
        reward = torch.zeros((self.num_envs, self.num_agents), dtype=self.dtype, device=self.device)
        return obs, reward, self.done.clone(), {"step": int(self.steps.max().item())}


class LegacyVectorEnv:
    def __init__(self) -> None:
        self.num_envs = 2
        self.num_agents = 2
        self.config = SimpleNamespace(max_steps=4)
        self.observation_space = SimpleNamespace(shape=(4,))
        self.action_space = SimpleNamespace(n=1)
        self.agents = []
        self.actions_seen: list[list[dict[int, int]]] = []
        self.step_calls = 0

    def assign_agents(self, agents: list[DummyAgent]) -> None:
        self.agents = list(agents)

    def reset(self):
        self.step_calls = 0
        obs = {"obs": np.zeros((self.num_envs, 4), dtype=np.float32)}
        return obs, {"reset": True}

    def step(self, actions):
        self.actions_seen.append(actions)
        self.step_calls += 1
        obs = {"obs": np.full((self.num_envs, 4), self.step_calls, dtype=np.float32)}
        reward = np.zeros((self.num_envs, self.num_agents), dtype=np.float32)
        terminated = np.array([self.step_calls >= 2, self.step_calls >= 2], dtype=np.bool_)
        truncated = np.array([False, False], dtype=np.bool_)
        return obs, reward, terminated, truncated, {"step": self.step_calls}


def make_agent(agent_id: int, policy: nn.Module, env) -> DummyAgent:
    del env
    return DummyAgent(agent_id=agent_id, action_space=np.array([0], dtype=np.int64), policy=policy)


def test_simulator_run_uses_tensor_action_matrix_for_torch_multi_agent_env() -> None:
    env = TorchMultiAgentEnv()
    simulator = Simulator(
        env=env,
        num_agents=2,
        policies=[ConstantPolicy(), ConstantPolicy()],
        loss_trainer=NoOpTrainer(),
        agent_factory=make_agent,
    )

    simulator.run()

    assert len(env.actions_seen) == 2
    first_actions = env.actions_seen[0]
    assert isinstance(first_actions, torch.Tensor)
    assert first_actions.shape == (2, 2)
    assert first_actions.dtype == torch.int64
    assert torch.equal(first_actions, torch.zeros((2, 2), dtype=torch.int64))


def test_simulator_run_falls_back_to_legacy_action_dicts() -> None:
    env = LegacyVectorEnv()
    simulator = Simulator(
        env=env,
        num_agents=2,
        policies=[ConstantPolicy(), ConstantPolicy()],
        loss_trainer=NoOpTrainer(),
        agent_factory=make_agent,
    )

    simulator.run()

    assert len(env.actions_seen) == 2
    assert env.actions_seen[0] == [{0: 0, 1: 0}, {0: 0, 1: 0}]
