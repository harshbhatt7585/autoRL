from __future__ import annotations

from server.router_factory import create_game_router

router = create_game_router(
    prefix="/battle-grid",
    tag="battle-grid",
    replay_env_var="BATTLE_GRID_REPLAY_DIR",
    default_replay_dir="recordings/battle_grid",
)
