from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RENDERER_ROOT = PROJECT_ROOT / "renderer"
FRONTEND_ROOT = RENDERER_ROOT / "frontend"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8770
DEFAULT_FRONTEND_PORT = 5173


@dataclass(frozen=True)
class ReplayTarget:
    env_name: str
    route_prefix: str
    frontend_game: str
    replay_dir: str
    replay_env_var: str
    display_name: str


@dataclass
class ReplayServices:
    target: ReplayTarget
    backend_url: str
    frontend_url: str | None
    backend_process: subprocess.Popen[str] | None
    frontend_process: subprocess.Popen[str] | None
    frontend_note: str | None = None


REPLAY_TARGETS: dict[str, ReplayTarget] = {
    "battle-grid": ReplayTarget(
        env_name="battle-grid",
        route_prefix="/battle-grid",
        frontend_game="battle-grid",
        replay_dir="recordings/battle_grid",
        replay_env_var="BATTLE_GRID_REPLAY_DIR",
        display_name="Battle Grid",
    ),
    "maze-race": ReplayTarget(
        env_name="maze-race",
        route_prefix="/maze",
        frontend_game="maze",
        replay_dir="recordings/maze_race",
        replay_env_var="MAZE_REPLAY_DIR",
        display_name="Maze Race",
    ),
    "snake": ReplayTarget(
        env_name="snake",
        route_prefix="/snake",
        frontend_game="snake",
        replay_dir="recordings/snake",
        replay_env_var="SNAKE_REPLAY_DIR",
        display_name="Snake",
    ),
}


def replay_target_for_env(env_name: str) -> ReplayTarget | None:
    return REPLAY_TARGETS.get(env_name.strip().lower())


def supported_replay_env_names() -> tuple[str, ...]:
    return tuple(sorted(REPLAY_TARGETS))


def start_replay_services(
    env_name: str,
    *,
    host: str = DEFAULT_HOST,
    api_port: int = DEFAULT_API_PORT,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
) -> ReplayServices:
    target = replay_target_for_env(env_name)
    if target is None:
        supported = ", ".join(supported_replay_env_names())
        raise ValueError(
            f"Replay mode is only supported for: {supported}. " f"Received {env_name!r}."
        )

    backend_url = f"http://{host}:{int(api_port)}"
    backend_process: subprocess.Popen[str] | None = None
    frontend_process: subprocess.Popen[str] | None = None

    root_payload = _fetch_json(f"{backend_url}/")
    if root_payload is not None:
        if root_payload.get("service") != "simverse-api":
            raise RuntimeError(
                f"Port {api_port} is already in use by another service. "
                f"Expected Simverse replay API at {backend_url}."
            )
        if not _route_snapshot_ready(backend_url, target.route_prefix):
            raise RuntimeError(
                f"A Simverse replay server is already running at {backend_url}, "
                f"but it does not expose {target.route_prefix}. Restart it after updating."
            )
    else:
        _ensure_backend_dependencies()
        env = os.environ.copy()
        env["SIMVERSE_API_HOST"] = host
        env["SIMVERSE_API_PORT"] = str(int(api_port))
        env[target.replay_env_var] = str((PROJECT_ROOT / target.replay_dir).resolve())
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "server.main"],
            cwd=RENDERER_ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if not _wait_for(lambda: _route_snapshot_ready(backend_url, target.route_prefix), 10.0):
            _terminate_process(backend_process)
            raise RuntimeError(f"Replay API failed to start at {backend_url}.")

    frontend_url = f"http://{host}:{int(frontend_port)}/render?game={target.frontend_game}"
    frontend_note: str | None = None
    if _http_ready(frontend_url):
        pass
    else:
        frontend_note = (
            "Frontend was not auto-started. Run `cd renderer/frontend && npm run dev` "
            "to open the replay UI."
        )
        if shutil.which("npm") is not None and (FRONTEND_ROOT / "node_modules").exists():
            env = os.environ.copy()
            env["VITE_SNAKE_API_URL"] = backend_url
            env["VITE_MAZE_API_URL"] = backend_url
            env["VITE_BATTLE_GRID_API_URL"] = backend_url
            frontend_process = subprocess.Popen(
                [
                    "npm",
                    "run",
                    "dev",
                    "--",
                    "--host",
                    host,
                    "--port",
                    str(int(frontend_port)),
                ],
                cwd=FRONTEND_ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            if _wait_for(lambda: _http_ready(frontend_url), 15.0):
                frontend_note = None
            else:
                _terminate_process(frontend_process)
                frontend_process = None

    return ReplayServices(
        target=target,
        backend_url=backend_url,
        frontend_url=frontend_url,
        backend_process=backend_process,
        frontend_process=frontend_process,
        frontend_note=frontend_note,
    )


def stop_replay_services(services: ReplayServices) -> None:
    _terminate_process(services.frontend_process)
    _terminate_process(services.backend_process)


def _ensure_backend_dependencies() -> None:
    missing = [
        module for module in ("fastapi", "uvicorn") if importlib.util.find_spec(module) is None
    ]
    if missing:
        deps = ", ".join(missing)
        raise RuntimeError(
            f"Replay backend dependencies are missing: {deps}. "
            "Install them with `pip install -e .[renderer]`."
        )


def _terminate_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _wait_for(check: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.2)
    return False


def _fetch_json(url: str) -> dict[str, Any] | None:
    try:
        with urlopen(url, timeout=0.5) as response:
            data = response.read().decode("utf-8")
    except (HTTPError, URLError, TimeoutError, ValueError):
        return None
    try:
        import json

        payload = json.loads(data)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _http_ready(url: str) -> bool:
    try:
        with urlopen(url, timeout=0.5) as response:
            return int(getattr(response, "status", 200)) < 400
    except (HTTPError, URLError, TimeoutError, ValueError):
        return False


def _route_snapshot_ready(base_url: str, route_prefix: str) -> bool:
    payload = _fetch_json(f"{base_url}{route_prefix}/snapshot")
    return payload is not None and "replay_count" in payload
