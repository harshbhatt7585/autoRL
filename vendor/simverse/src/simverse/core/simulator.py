from __future__ import annotations

import random
from typing import Any, Callable, List, Mapping, Optional, Protocol, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from simverse.core.agent import SimAgent
from simverse.core.env import SimEnv
from simverse.core.trainer import Trainer
from simverse.training.checkpoints import Checkpointer

AgentFactory = Callable[[int, nn.Module, Any], SimAgent]


class Renderer(Protocol):
    """Protocol for environment renderers."""

    def draw(self, env: Any) -> None: ...
    def handle_events(self) -> None: ...
    def close(self) -> None: ...


class Simulator:
    """High-level orchestrator that spawns agents and delegates training."""

    def __init__(
        self,
        env: Any,
        num_agents: int,
        policies: List[nn.Module],
        loss_trainer: Trainer,
        agent_factory: AgentFactory,
    ) -> None:
        if not policies:
            raise ValueError("Simulator requires at least one policy instance")
        self.env = env
        self.num_agents = num_agents
        self.policies = policies
        self.loss_trainer = loss_trainer
        self.agent_factory = agent_factory

        self.checkpointer = Checkpointer(self.env)

    def _attach_agents(self, agents: List[SimAgent]) -> None:
        if hasattr(self.env, "assign_agents") and callable(self.env.assign_agents):
            self.env.assign_agents(agents)
        elif hasattr(self.env, "agents"):
            self.env.agents = agents

    @staticmethod
    def _unwrap_reset_result(reset_result: Any) -> Any:
        if (
            isinstance(reset_result, tuple)
            and len(reset_result) == 2
            and isinstance(reset_result[1], Mapping)
        ):
            return reset_result[0]
        return reset_result

    def _extract_observation_payload(self, obs: Any) -> Any:
        obs = self._unwrap_reset_result(obs)
        if isinstance(obs, Mapping):
            if "obs" in obs:
                return obs
            raise KeyError("Expected observation payload under the 'obs' key")
        return obs

    def _prepare_policy_inputs(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor | None]:
        payload = self._extract_observation_payload(obs)
        if isinstance(payload, Mapping):
            obs_payload = payload["obs"]
            feature_payload = payload.get("features")
        else:
            obs_payload = payload
            feature_payload = None

        obs_tensor = (
            obs_payload if isinstance(obs_payload, torch.Tensor) else torch.as_tensor(obs_payload)
        )
        if obs_tensor.ndim == 0:
            obs_tensor = obs_tensor.unsqueeze(0)

        observation_space = getattr(self.env, "observation_space", None)
        if hasattr(observation_space, "spaces"):
            single_obs_shape = getattr(observation_space["obs"], "shape", None)
            feature_shape = getattr(observation_space.spaces.get("features"), "shape", None)
        else:
            single_obs_shape = getattr(observation_space, "shape", None)
            feature_shape = None
        if single_obs_shape is not None:
            expected_shape = tuple(int(dim) for dim in single_obs_shape)
            if tuple(obs_tensor.shape) == expected_shape:
                obs_tensor = obs_tensor.unsqueeze(0)
        elif obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        expected_batch = getattr(self.env, "num_envs", None)
        if expected_batch is not None and obs_tensor.ndim > 0:
            expected_batch = int(expected_batch)
            if obs_tensor.shape[0] != expected_batch and expected_batch == 1:
                obs_tensor = obs_tensor.unsqueeze(0)

        feature_tensor: torch.Tensor | None = None
        if feature_payload is not None:
            feature_tensor = (
                feature_payload
                if isinstance(feature_payload, torch.Tensor)
                else torch.as_tensor(feature_payload)
            )
            if feature_tensor.ndim == 0:
                feature_tensor = feature_tensor.unsqueeze(0)
            elif feature_shape is not None:
                expected_feature_shape = tuple(int(dim) for dim in feature_shape)
                if tuple(feature_tensor.shape) == expected_feature_shape:
                    feature_tensor = feature_tensor.unsqueeze(0)
            if feature_tensor.shape[0] != obs_tensor.shape[0] and obs_tensor.shape[0] == 1:
                feature_tensor = feature_tensor.unsqueeze(0)
            feature_tensor = feature_tensor.to(dtype=torch.float32)

        return obs_tensor.to(dtype=torch.float32), feature_tensor

    @staticmethod
    def _all_done(done: Any) -> bool:
        if isinstance(done, torch.Tensor):
            return bool(done.to(dtype=torch.bool).all().item())
        if isinstance(done, np.ndarray):
            return bool(np.all(done))
        if isinstance(done, Sequence) and not isinstance(done, (str, bytes)):
            return bool(np.all(np.asarray(done, dtype=np.bool_)))
        return bool(done)

    @staticmethod
    def _merge_done_flags(terminated: Any, truncated: Any) -> Any:
        if isinstance(terminated, torch.Tensor) or isinstance(truncated, torch.Tensor):
            if isinstance(terminated, torch.Tensor):
                device = terminated.device
            else:
                device = truncated.device
            terminated_tensor = (
                terminated
                if isinstance(terminated, torch.Tensor)
                else torch.as_tensor(terminated, dtype=torch.bool, device=device)
            )
            truncated_tensor = (
                truncated
                if isinstance(truncated, torch.Tensor)
                else torch.as_tensor(truncated, dtype=torch.bool, device=device)
            )
            return terminated_tensor.to(dtype=torch.bool) | truncated_tensor.to(dtype=torch.bool)

        return np.asarray(terminated, dtype=np.bool_) | np.asarray(truncated, dtype=np.bool_)

    def _step_env(
        self,
        env_actions: torch.Tensor | Sequence[dict[int, int]] | dict[int, int],
    ) -> tuple[Any, Any, Any, Any]:
        step_result = self.env.step(env_actions)
        if not isinstance(step_result, tuple):
            raise TypeError("Environment step() must return a tuple")
        if len(step_result) == 4:
            return step_result
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = self._merge_done_flags(terminated, truncated)
            return obs, reward, done, info
        raise ValueError(f"Unsupported step() return arity: {len(step_result)}")

    def _sample_env_actions(
        self,
        *,
        agents: List[SimAgent],
        obs_tensor: torch.Tensor,
        feature_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor | Sequence[dict[int, int]] | dict[int, int]:
        batch_envs = int(obs_tensor.shape[0]) if obs_tensor.ndim > 0 else 1

        if isinstance(self.env, SimEnv):
            env_actions = torch.full(
                (batch_envs, self.num_agents),
                -1,
                dtype=torch.int64,
                device=self.env.device,
            )
        else:
            env_actions = [{} for _ in range(batch_envs)]

        for agent in agents:
            if agent.policy is None:
                continue
            agent.policy.eval()
            with torch.no_grad():
                if feature_tensor is None:
                    logits, _ = agent.policy(obs_tensor)
                else:
                    logits, _ = agent.policy(obs_tensor, feature_tensor)
                logits_f32 = logits.to(dtype=torch.float32)
                if not bool(torch.isfinite(logits_f32).all().item()):
                    raise RuntimeError("Non-finite logits detected during simulator run")
                action = Categorical(logits=logits_f32).sample()

            if action.ndim == 0:
                action = action.unsqueeze(0)
            if action.shape[0] == 1 and batch_envs > 1:
                action = action.expand(batch_envs)
            if action.shape[0] != batch_envs:
                raise ValueError(
                    f"Agent {agent.agent_id} produced {int(action.shape[0])} actions for "
                    f"{batch_envs} environments"
                )

            if isinstance(env_actions, torch.Tensor):
                env_actions[:, agent.agent_id] = action.to(
                    device=env_actions.device,
                    dtype=torch.int64,
                )
                continue

            action_cpu = action.detach().cpu()
            for env_idx in range(batch_envs):
                env_actions[env_idx][agent.agent_id] = int(action_cpu[env_idx].item())

        if isinstance(env_actions, list) and batch_envs == 1:
            return env_actions[0]
        return env_actions

    def _build_agents(self) -> List[SimAgent]:
        agents: List[SimAgent] = []
        assign_by_index = len(self.policies) == self.num_agents
        single_policy = len(self.policies) == 1
        for idx in range(self.num_agents):
            if assign_by_index:
                policy = self.policies[idx]
            elif single_policy:
                policy = self.policies[0]
            else:
                policy = random.choice(self.policies)
            agent = self.agent_factory(idx, policy, self.env)
            agents.append(agent)
        return agents

    def train(self, *args, **kwargs) -> None:
        agents = self._build_agents()
        self._attach_agents(agents)
        self.loss_trainer.train(self.env, agents, *args, **kwargs)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        self.checkpointer.load(checkpoint_path)

    def run(
        self,
        checkpoint_path: Optional[str] = None,
        max_steps: Optional[int] = None,
        renderer: Optional[Renderer] = None,
    ) -> None:
        """
        Run inference with trained policies.

        Args:
            checkpoint_path: Path to load model checkpoint from
            max_steps: Maximum steps to run (defaults to env.config.max_steps)
            renderer: Optional renderer instance for visualization
        """
        agents = self._build_agents()
        self._attach_agents(agents)
        previous_clone_payload_tensors = getattr(self.env, "clone_payload_tensors", None)

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

        try:
            if isinstance(self.env, SimEnv) and hasattr(self.env, "set_fast_payload_mode"):
                self.env.set_fast_payload_mode(False)

            obs = self._unwrap_reset_result(self.env.reset())
            max_steps = max_steps or getattr(getattr(self.env, "config", None), "max_steps", None)
            done = False
            step = 0

            while not done and (max_steps is None or step < max_steps):
                if renderer:
                    renderer.handle_events()

                obs_tensor, feature_tensor = self._prepare_policy_inputs(obs)
                actions = self._sample_env_actions(
                    agents=agents,
                    obs_tensor=obs_tensor,
                    feature_tensor=feature_tensor,
                )

                obs, reward, done, info = self._step_env(actions)
                obs = self._unwrap_reset_result(obs)

                if renderer:
                    renderer.draw(self.env)

                del reward, info
                step += 1
                done = self._all_done(done)
        finally:
            if isinstance(self.env, SimEnv) and previous_clone_payload_tensors is not None:
                self.env.clone_payload_tensors = bool(previous_clone_payload_tensors)
            if renderer:
                renderer.close()
