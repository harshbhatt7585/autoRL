from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simverse.training.checkpoints import Checkpointer


class Trainer(ABC):
    def __init__(self, *args, **kwargs) -> None:
        self._checkpointer: Checkpointer | None = None

    @abstractmethod
    def train(self, *args, **kwargs) -> None:
        pass

    def save_checkpoint(self, checkpoint_path: str) -> None:
        from simverse.training.checkpoints import Checkpointer

        if self._checkpointer is None or self._checkpointer.env is not self.env:
            self._checkpointer = Checkpointer(self.env)
        checkpointer = self._checkpointer
        checkpointer.save(checkpoint_path)
