"""Simple reference policy that samples uniformly from the action space."""

from __future__ import annotations


class RandomPolicy:
    """Randomly samples valid actions from the bound environment."""

    def __init__(self, action_space):
        self._action_space = action_space

    def act(self, _obs):
        return self._action_space.sample()


__all__ = ["RandomPolicy"]
