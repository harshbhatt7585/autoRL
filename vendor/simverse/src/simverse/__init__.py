"""Public package exports for the current Simverse surface."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BattleGridConfig": ("simverse.envs", "BattleGridConfig"),
    "BattleGridEnv": ("simverse.envs", "BattleGridEnv"),
    "BattleGridTorchEnv": ("simverse.envs", "BattleGridTorchEnv"),
    "CentralizedCritic": ("simverse.policies", "CentralizedCritic"),
    "FarmtilaConfig": ("simverse.envs", "FarmtilaConfig"),
    "FarmtilaEnv": ("simverse.envs", "FarmtilaEnv"),
    "FarmtilaTorchEnv": ("simverse.envs", "FarmtilaTorchEnv"),
    "GymEnv": ("simverse.envs", "GymEnv"),
    "GymTorchConfig": ("simverse.envs", "GymTorchConfig"),
    "GymTorchEnv": ("simverse.envs", "GymTorchEnv"),
    "MazeRaceConfig": ("simverse.envs", "MazeRaceConfig"),
    "MazeRaceEnv": ("simverse.envs", "MazeRaceEnv"),
    "MazeRaceTorchEnv": ("simverse.envs", "MazeRaceTorchEnv"),
    "RandomPolicy": ("simverse.policies", "RandomPolicy"),
    "SimAgent": ("simverse.core.agent", "SimAgent"),
    "SimEnv": ("simverse.core.env", "SimEnv"),
    "SimplePolicy": ("simverse.policies", "SimplePolicy"),
    "Simulator": ("simverse.core.simulator", "Simulator"),
    "SnakeAgent": ("simverse.envs", "SnakeAgent"),
    "SnakeConfig": ("simverse.envs", "SnakeConfig"),
    "SnakeEnv": ("simverse.envs", "SnakeEnv"),
    "SnakeTorchEnv": ("simverse.envs", "SnakeTorchEnv"),
    "Trainer": ("simverse.core.trainer", "Trainer"),
    "quicktrain": ("simverse.recipes", "quicktrain"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
