from __future__ import annotations

from types import SimpleNamespace

import simverse.cli as cli_module
from simverse.replay_server import replay_target_for_env
from typer.testing import CliRunner


def test_replay_target_mapping_includes_battle_grid() -> None:
    target = replay_target_for_env("battle-grid")

    assert target is not None
    assert target.route_prefix == "/battle-grid"
    assert target.frontend_game == "battle-grid"


def test_train_with_replay_starts_and_stops_services(monkeypatch) -> None:
    runner = CliRunner()
    calls: list[str] = []
    replay_events: list[str] = []

    def fake_trainer() -> None:
        calls.append("trainer")

    def fake_start_replay_services(env_name: str):
        replay_events.append(f"start:{env_name}")
        return SimpleNamespace(
            backend_url="http://127.0.0.1:8770",
            frontend_url="http://127.0.0.1:5173/render?game=snake",
            frontend_note=None,
            target=SimpleNamespace(
                route_prefix="/snake",
                display_name="Snake",
                replay_dir="recordings/snake",
            ),
        )

    def fake_stop_replay_services(_services) -> None:
        replay_events.append("stop")

    monkeypatch.setitem(
        cli_module.TRAINERS,
        "snake",
        cli_module.TrainEntry("Snake", fake_trainer),
    )
    monkeypatch.setattr(cli_module, "start_replay_services", fake_start_replay_services)
    monkeypatch.setattr(cli_module, "stop_replay_services", fake_stop_replay_services)

    result = runner.invoke(cli_module.app, ["train", "snake", "--replay"])

    assert result.exit_code == 0
    assert replay_events == ["start:snake", "stop"]
    assert calls == ["trainer"]
    assert "Replay API ready" in result.stdout
    assert "Replay UI ready" in result.stdout


def test_train_with_replay_rejects_unsupported_env(monkeypatch) -> None:
    runner = CliRunner()

    def fake_start_replay_services(_env_name: str):
        raise ValueError("Replay mode is only supported for: battle-grid, maze-race, snake.")

    monkeypatch.setattr(cli_module, "start_replay_services", fake_start_replay_services)

    result = runner.invoke(cli_module.app, ["train", "farmtila", "--replay"])

    assert result.exit_code != 0
    combined_output = f"{result.stdout}\n{getattr(result, 'stderr', '')}"
    assert "Replay mode is only supported" in combined_output
