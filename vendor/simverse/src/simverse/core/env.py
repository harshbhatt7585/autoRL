from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn


class SimEnv(nn.Module, ABC):
    """Torch-native, batched simulation environment base class."""

    def __init__(self, device: str | torch.device = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.device = torch.device(device)
        self.dtype = dtype

    @property
    @abstractmethod
    def action_space(self) -> Any: ...

    @property
    @abstractmethod
    def observation_space(self) -> Any: ...

    @abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]: ...

    @abstractmethod
    def step(
        self, actions: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        NotImplementedError("Subclasses must implement this method")

    def get_observation(self) -> Dict[str, torch.Tensor]:
        return self.reset()

    def _resolve_num_envs(self, num_envs: int | None, config: Any, *, default: int = 1) -> int:
        resolved = num_envs
        if resolved is None:
            resolved = getattr(config, "num_envs", default)
        return max(1, int(resolved))

    def _normalize_action_matrix(
        self,
        actions: torch.Tensor | Sequence[int] | None,
        *,
        num_agents: int | None = None,
        missing_action: int = -1,
    ) -> torch.Tensor:
        env_count = int(self.num_envs)
        agent_count = int(self.num_agents if num_agents is None else num_agents)

        if actions is None:
            return torch.full(
                (env_count, agent_count),
                int(missing_action),
                dtype=torch.int64,
                device=self.device,
            )

        action_tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        if action_tensor.ndim == 1:
            if env_count == 1 and action_tensor.numel() == agent_count:
                action_tensor = action_tensor.unsqueeze(0)
            elif agent_count == 1 and action_tensor.numel() == env_count:
                action_tensor = action_tensor.unsqueeze(1)

        expected_shape = (env_count, agent_count)
        if tuple(action_tensor.shape) != expected_shape:
            raise ValueError(
                f"Expected actions shape {expected_shape}, got {tuple(action_tensor.shape)}"
            )
        return action_tensor.to(device=self.device, dtype=torch.int64)

    def _normalize_single_agent_actions(
        self,
        actions: torch.Tensor | Sequence[int] | Mapping[int, int] | None,
        *,
        missing_action: int = -1,
        dict_default: int | None = None,
    ) -> torch.Tensor:
        env_count = int(self.num_envs)

        if actions is None:
            return torch.full(
                (env_count,),
                int(missing_action),
                dtype=torch.int64,
                device=self.device,
            )

        if isinstance(actions, Mapping):
            if env_count != 1:
                raise ValueError("Dict actions are supported only when num_envs == 1")
            fallback = missing_action if dict_default is None else dict_default
            return torch.as_tensor(
                [int(actions.get(0, fallback))],
                dtype=torch.int64,
                device=self.device,
            )

        action_tensor = actions if isinstance(actions, torch.Tensor) else torch.as_tensor(actions)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        elif action_tensor.ndim == 2 and action_tensor.shape[1] == 1:
            action_tensor = action_tensor[:, 0]
        if action_tensor.ndim != 1:
            raise ValueError(
                "Expected actions with shape [num_envs] or [num_envs, 1], "
                f"got {tuple(action_tensor.shape)}"
            )

        if action_tensor.shape[0] == 1 and env_count > 1:
            action_tensor = action_tensor.repeat(env_count)

        if action_tensor.shape[0] != env_count:
            raise ValueError(f"Expected {env_count} actions, got {int(action_tensor.shape[0])}")

        return action_tensor.to(device=self.device, dtype=torch.int64)

    def _assign_agents(
        self,
        agents: Sequence[Any],
        *,
        expected_count: int | None = None,
        label: str = "Environment",
    ) -> None:
        expected = int(self.num_agents if expected_count is None else expected_count)
        if len(agents) != expected:
            raise ValueError(f"{label} requires exactly {expected} agents")
        self.agents = list(agents)

    def _empty_rewards(self, *, num_agents: int | None = None) -> torch.Tensor:
        agent_count = int(self.num_agents if num_agents is None else num_agents)
        return torch.zeros((self.num_envs, agent_count), dtype=self.dtype, device=self.device)

    def _reset_episode_state(self, *, winner_none: int | None = None) -> None:
        if hasattr(self, "done"):
            self.done.zero_()
        if hasattr(self, "steps"):
            self.steps.zero_()
        if winner_none is not None and hasattr(self, "winner"):
            self.winner.fill_(int(winner_none))

    def _pack_observation_dict(
        self,
        obs: torch.Tensor,
        *,
        clone_obs: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"obs": obs.clone() if clone_obs else obs}
        if hasattr(self, "done"):
            payload["done"] = self.done.clone()
        if hasattr(self, "winner"):
            payload["winner"] = self.winner.clone()
        if hasattr(self, "steps"):
            payload["steps"] = self.steps.clone()
        if extra:
            payload.update(dict(extra))
        return payload

    def _build_info(self, *, extra: Mapping[str, Any] | None = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if hasattr(self, "winner"):
            info["winner"] = self.winner.clone()
        if hasattr(self, "steps"):
            info["steps"] = self.steps.clone()
        if extra:
            info.update(dict(extra))
        return info

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if args:
            self.device = torch.device(args[0])
        elif "device" in kwargs:
            self.device = torch.device(kwargs["device"])
        return self
