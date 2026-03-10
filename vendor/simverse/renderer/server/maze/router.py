from __future__ import annotations

from server.router_factory import create_game_router

router = create_game_router(
    prefix="/maze",
    tag="maze",
    replay_env_var="MAZE_REPLAY_DIR",
    default_replay_dir="recordings/maze_race",
)
