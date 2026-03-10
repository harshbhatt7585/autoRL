from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch.nn as nn


class SimAgent(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        action_space: np.ndarray,
        policy: Optional[nn.Module] = None,
        **kwargs,
    ) -> None:
        self.name = name
        self.action_space = action_space
        self.policy = policy if policy is not None else None

    @abstractmethod
    def action(self, obs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def info(self) -> dict:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_action_space(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_memory(self) -> dict:
        pass

    @abstractmethod
    def current_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_policy(self) -> nn.Module | None:
        pass

    @abstractmethod
    def set_policy(self, policy: nn.Module | None) -> None:
        pass
