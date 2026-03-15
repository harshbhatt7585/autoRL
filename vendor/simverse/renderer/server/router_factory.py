from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]


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

    def replay_dir(override_dir: str | None = None) -> Path:
        configured = override_dir if override_dir else os.getenv(replay_env_var, default_replay_dir)
        candidate = Path(configured).expanduser()
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate

    def all_replay_files(override_dir: str | None = None) -> list[Path]:
        directory = replay_dir(override_dir)
        if not directory.exists():
            return []
        single_replay_file = directory / "replay.json"
        if single_replay_file.is_file():
            return [single_replay_file]
        all_files = sorted(p for p in directory.rglob("*.json") if p.is_file())
        nested_files = [path for path in all_files if path.parent != directory]
        if not nested_files:
            return all_files
        latest_run_dir = max(
            {path.parent for path in nested_files}, key=lambda run_dir: run_dir.name
        )
        return sorted(path for path in nested_files if path.parent == latest_run_dir)

    def build_replay_event_payload(override_dir: str | None = None) -> dict[str, Any]:
        directory = replay_dir(override_dir)
        files = all_replay_files(override_dir)
        payload: dict[str, Any] = {
            "event": "replay_update",
            "replay_dir": str(directory),
            "replay_count": len(files),
            "latest_replay_id": None,
            "latest_replay_name": None,
            "latest_frame_index": -1,
        }
        if not files:
            return payload
        latest = files[-1]
        payload["latest_replay_id"] = replay_id_for_path(latest, directory)
        payload["latest_replay_name"] = replay_name_for_path(latest, directory)
        try:
            replay_data = _read_json(latest)
            frames = replay_data.get("frames") if isinstance(replay_data, dict) else None
            if isinstance(frames, list):
                payload["latest_frame_index"] = max(len(frames) - 1, 0)
        except Exception:
            payload["latest_frame_index"] = -1
        return payload

    def event_signature(payload: dict[str, Any]) -> tuple[Any, ...]:
        return (
            payload.get("replay_count"),
            payload.get("latest_replay_id"),
            payload.get("latest_frame_index"),
        )

    def replay_id_for_path(path: Path, directory: Path) -> str:
        relative = path.relative_to(directory).with_suffix("")
        return relative.as_posix()

    def replay_name_for_path(path: Path, directory: Path) -> str:
        return path.relative_to(directory).as_posix()

    def resolve_replay_path(
        replay_id: str, override_dir: str | None = None
    ) -> tuple[Path, Path] | None:
        directory = replay_dir(override_dir).resolve()
        for name in (replay_id, f"{replay_id}.json"):
            candidate = (directory / name).resolve()
            if directory not in candidate.parents and candidate != directory:
                continue
            if candidate.is_file() and candidate.suffix.lower() == ".json":
                return candidate, directory
        return None

    @router.get("/")
    def game_root() -> dict[str, str]:
        return {"service": tag, "status": "ok"}

    @router.get("/snapshot")
    def get_snapshot(dir: str | None = None) -> dict[str, Any]:
        directory = replay_dir(dir)
        files = all_replay_files(dir)
        return {
            "replay_dir": str(directory),
            "replay_count": len(files),
            "latest_replay": replay_name_for_path(files[-1], directory) if files else None,
        }

    @router.get("/replays")
    @router.get("/replays/")
    def get_replays(dir: str | None = None) -> dict[str, list[dict[str, str]]]:
        episodes: list[dict[str, str]] = []
        directory = replay_dir(dir)
        for file in all_replay_files(dir):
            episodes.append(
                {
                    "id": replay_id_for_path(file, directory),
                    "name": replay_name_for_path(file, directory),
                }
            )
        return {"episodes": episodes}

    @router.get("/replays/events")
    async def get_replay_events(dir: str | None = None) -> StreamingResponse:
        async def event_stream():
            last_sig: tuple[Any, ...] | None = None
            while True:
                payload = build_replay_event_payload(dir)
                signature = event_signature(payload)
                if signature != last_sig:
                    yield f"event: replay_update\ndata: {json.dumps(payload)}\n\n"
                    last_sig = signature
                else:
                    # Keep the SSE connection alive through proxies/load balancers.
                    yield "event: ping\ndata: {}\n\n"
                await asyncio.sleep(1.0)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.get("/replays/{replay_id:path}")
    def get_replay(replay_id: str, dir: str | None = None) -> dict[str, Any]:
        resolved = resolve_replay_path(replay_id, dir)
        if resolved is not None:
            path, directory = resolved
            return {
                "id": replay_id_for_path(path, directory),
                "name": replay_name_for_path(path, directory),
                "data": _read_json(path),
            }
        raise HTTPException(status_code=404, detail=f"Replay not found: {replay_id}")

    @router.get("/replay")
    @router.get("/replay/")
    def get_single_replay(dir: str | None = None) -> dict[str, Any]:
        files = all_replay_files(dir)
        if not files:
            raise HTTPException(status_code=404, detail="Replay not found")
        directory = replay_dir(dir)
        path = files[-1]
        return {
            "id": replay_id_for_path(path, directory),
            "name": replay_name_for_path(path, directory),
            "data": _read_json(path),
        }

    return router
