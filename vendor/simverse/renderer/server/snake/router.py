from __future__ import annotations

from server.router_factory import create_game_router

router = create_game_router(
    prefix="/snake",
    tag="snake",
    replay_env_var="SNAKE_REPLAY_DIR",
    default_replay_dir="recordings/snake",
)
