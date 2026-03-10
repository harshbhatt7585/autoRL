"""Core runtime abstractions for Simverse."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "SimAgent": ("simverse.core.agent", "SimAgent"),
    "SimEnv": ("simverse.core.env", "SimEnv"),
    "Simulator": ("simverse.core.simulator", "Simulator"),
    "Trainer": ("simverse.core.trainer", "Trainer"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
