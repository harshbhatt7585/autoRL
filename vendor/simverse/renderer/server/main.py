from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI

if __package__ in {None, ""}:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from server.battle_grid.router import router as battle_grid_router
from server.maze.router import router as maze_router
from server.snake.router import router as snake_router

app = FastAPI(title="Simverse API")
app.include_router(snake_router)
app.include_router(maze_router)
app.include_router(battle_grid_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "simverse-api", "status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("SIMVERSE_API_HOST", os.getenv("SNAKE_RENDER_HOST", "127.0.0.1"))
    port = int(os.getenv("SIMVERSE_API_PORT", os.getenv("SNAKE_RENDER_PORT", "8770")))
    uvicorn.run("server.main:app", host=host, port=port, reload=False)
