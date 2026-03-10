from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_game_router(
    *,
    prefix: str,
    tag: str,
    replay_env_var: str,
    default_replay_dir: str,
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=[tag])

    def replay_dir() -> Path:
        configured = os.getenv(replay_env_var, default_replay_dir)
        return Path(configured)

    def all_replay_files() -> list[Path]:
        directory = replay_dir()
        if not directory.exists():
            return []
        return sorted(p for p in directory.glob("*.json") if p.is_file())

    @router.get("/")
    def game_root() -> dict[str, str]:
        return {"service": tag, "status": "ok"}

    @router.get("/snapshot")
    def get_snapshot() -> dict[str, Any]:
        files = all_replay_files()
        return {
            "replay_dir": str(replay_dir()),
            "replay_count": len(files),
            "latest_replay": files[-1].name if files else None,
        }

    @router.get("/replays")
    @router.get("/replays/")
    def get_replays() -> dict[str, list[dict[str, str]]]:
        episodes: list[dict[str, str]] = []
        for file in all_replay_files():
            episodes.append({"id": file.stem, "name": file.name})
        return {"episodes": episodes}

    @router.get("/replays/{replay_id}")
    def get_replay(replay_id: str) -> dict[str, Any]:
        candidates = [replay_id, f"{replay_id}.json"]
        directory = replay_dir()
        for name in candidates:
            path = directory / name
            if path.exists() and path.is_file():
                return {"id": path.stem, "name": path.name, "data": _read_json(path)}
        raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")

    return router
