"""Training utilities and implementations for Simverse."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Checkpointer": ("simverse.training.checkpoints", "Checkpointer"),
    "DEFAULT_WANDB_PROJECT": ("simverse.training.wandb", "DEFAULT_WANDB_PROJECT"),
    "PPOTrainer": ("simverse.training.ppo", "PPOTrainer"),
    "TrainingStats": ("simverse.training.stats", "TrainingStats"),
    "build_adam_optimizers": ("simverse.training.utils", "build_adam_optimizers"),
    "build_ppo_training_config": ("simverse.training.utils", "build_ppo_training_config"),
    "compile_policy_models": ("simverse.training.utils", "compile_policy_models"),
    "configure_torch_backend": ("simverse.training.utils", "configure_torch_backend"),
    "configure_logging": ("simverse.training.logging", "configure_logging"),
    "get_logger": ("simverse.training.logging", "get_logger"),
    "resolve_rollout_dtype": ("simverse.training.utils", "resolve_rollout_dtype"),
    "resolve_torch_device": ("simverse.training.utils", "resolve_torch_device"),
    "run_ppo_training": ("simverse.training.utils", "run_ppo_training"),
    "training_logger": ("simverse.training.logging", "training_logger"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attribute_name)
