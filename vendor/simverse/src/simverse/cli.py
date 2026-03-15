"""Command line helpers for Simverse."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import typer

from simverse.envs.battle_grid.train import train as train_battle_grid
from simverse.envs.farmtila.train import train as train_farmtila
from simverse.envs.gym_env.train import train as train_gym_env
from simverse.envs.maze_race.train import train as train_maze_race
from simverse.envs.snake.train import train as train_snake
from simverse.recipes import quicktrain
from simverse.replay_server import start_replay_services, stop_replay_services

app = typer.Typer(help="Simverse RL quickstart utilities")

TrainFn = Callable[[], None]


@dataclass(frozen=True)
class TrainEntry:
    display_name: str
    trainer: TrainFn


TRAINERS: dict[str, TrainEntry] = {
    "battle-grid": TrainEntry("Battle Grid", train_battle_grid),
    "farmtila": TrainEntry("Farmtila", train_farmtila),
    "gym-env": TrainEntry("Gym Env", train_gym_env),
    "maze-race": TrainEntry("Maze Race", train_maze_race),
    "snake": TrainEntry("Snake", train_snake),
}


@app.command()
def rollout(
    env_id: str = typer.Option("CartPole-v1", help="Gymnasium environment id"),
    episodes: int = typer.Option(5, help="Number of episodes to simulate"),
    max_steps: int = typer.Option(200, help="Max steps per episode"),
    render: bool = typer.Option(False, help="Enable human rendering"),
):
    """Runs the quicktrain recipe and prints summary stats."""

    stats = quicktrain(env_id=env_id, episodes=episodes, max_steps=max_steps, render=render)
    typer.echo(
        f"{stats['env_id']} | completed {stats['episodes']} episodes | "
        f"mean reward: {stats['mean_reward']:.2f} | "
        f"max reward: {stats['max_reward']:.2f}"
    )


@app.command()
def train(
    env_name: str = typer.Argument(..., help="Environment name, for example: battle-grid"),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Launch the replay backend and, when available, the replay UI during training.",
    ),
):
    """Run a built-in environment training entrypoint."""

    key = env_name.strip().lower()
    trainer_entry = TRAINERS.get(key)
    if trainer_entry is None:
        available = ", ".join(sorted(TRAINERS))
        raise typer.BadParameter(
            f"Unknown environment '{env_name}'. Available environments: {available}"
        )

    replay_services = None
    if replay:
        try:
            replay_services = start_replay_services(key)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        except RuntimeError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc

        replay_api_url = f"{replay_services.backend_url}{replay_services.target.route_prefix}"
        typer.echo(f"Replay API ready at {replay_api_url}")
        if replay_services.frontend_note is None and replay_services.frontend_url is not None:
            typer.echo(f"Replay UI ready at {replay_services.frontend_url}")
        elif replay_services.frontend_note is not None:
            typer.echo(replay_services.frontend_note)
            typer.echo(
                "If the frontend is already running, open "
                f"{replay_services.frontend_url} and select {replay_services.target.display_name}."
            )
        typer.echo(
            "New replay files appear after each episode finishes and are written to the "
            f"{replay_services.target.replay_dir} directory."
        )

    display_name = trainer_entry.display_name
    trainer = trainer_entry.trainer
    typer.echo(f"Starting {display_name} training")
    try:
        trainer()
    finally:
        if replay_services is not None:
            stop_replay_services(replay_services)


if __name__ == "__main__":
    app()
