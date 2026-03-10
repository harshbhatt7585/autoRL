"""Command line helpers for Simverse."""

from __future__ import annotations

from collections.abc import Callable

import typer

from simverse.envs.battle_grid.train import train as train_battle_grid
from simverse.envs.farmtila.train import train as train_farmtila
from simverse.envs.gym_env.train import train as train_gym_env
from simverse.envs.maze_race.train import train as train_maze_race
from simverse.envs.snake.train import train as train_snake
from simverse.recipes import quicktrain

app = typer.Typer(help="Simverse RL quickstart utilities")

TrainFn = Callable[[], None]

TRAINERS: dict[str, tuple[str, TrainFn]] = {
    "battle-grid": ("Battle Grid", train_battle_grid),
    "farmtila": ("Farmtila", train_farmtila),
    "gym-env": ("Gym Env", train_gym_env),
    "maze-race": ("Maze Race", train_maze_race),
    "snake": ("Snake", train_snake),
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
):
    """Run a built-in environment training entrypoint."""

    key = env_name.strip().lower()
    trainer_entry = TRAINERS.get(key)
    if trainer_entry is None:
        available = ", ".join(sorted(TRAINERS))
        raise typer.BadParameter(
            f"Unknown environment '{env_name}'. Available environments: {available}"
        )

    display_name, trainer = trainer_entry
    typer.echo(f"Starting {display_name} training")
    trainer()


if __name__ == "__main__":
    app()
