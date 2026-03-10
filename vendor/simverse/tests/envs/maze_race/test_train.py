from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch


def test_maze_race_train_defaults_to_two_agents(monkeypatch) -> None:
    train_module = importlib.import_module("simverse.envs.maze_race.train")
    captured: dict[str, object] = {}

    monkeypatch.setattr(train_module, "resolve_torch_device", lambda prefer_mps=True: "cpu")
    monkeypatch.setattr(
        train_module,
        "resolve_rollout_dtype",
        lambda device, cpu_dtype: torch.float32,
    )
    monkeypatch.setattr(train_module, "configure_torch_backend", lambda device: None)

    def fake_create_env(config, *, num_envs, device, dtype):
        captured["env_config"] = config
        captured["create_env_args"] = {
            "num_envs": num_envs,
            "device": device,
            "dtype": dtype,
        }
        return SimpleNamespace(
            observation_space=SimpleNamespace(shape=(5, 7, 7)),
            action_space=SimpleNamespace(n=5),
            config=config,
        )

    def fake_build_ppo_training_config(**kwargs):
        captured["training_config_kwargs"] = kwargs
        return kwargs

    def fake_run_ppo_training(**kwargs):
        captured["run_ppo_training_kwargs"] = kwargs

    monkeypatch.setattr(train_module, "create_env", fake_create_env)
    monkeypatch.setattr(train_module, "build_ppo_training_config", fake_build_ppo_training_config)
    monkeypatch.setattr(train_module, "run_ppo_training", fake_run_ppo_training)

    train_module.train(num_envs=16, episodes=3, use_wandb=False, use_compile=False)

    env_config = captured["env_config"]
    training_config_kwargs = captured["training_config_kwargs"]
    run_ppo_training_kwargs = captured["run_ppo_training_kwargs"]

    assert env_config.num_agents == 2
    assert training_config_kwargs["num_agents"] == 2
    assert run_ppo_training_kwargs["use_compile"] is False
