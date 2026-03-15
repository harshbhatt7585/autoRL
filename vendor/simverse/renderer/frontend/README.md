# Simverse Renderer (React)

This app is the frontend for Simverse replay playback.

## Modes
- `Live`
  - Placeholder only for now.
- `Replay`
  - Supports `Snake`, `Maze Race`, and `Battle Grid`.
  - Fetches replay files from `/{game}/replays/` (`/snake/...`, `/maze/...`, or `/battle-grid/...`).
  - Uses a shared grid renderer with per-game adapters.
  - Polls for newly written replay JSON files during training.

## Run
```bash
cd renderer
npm run dev
```

Open `http://localhost:5173/render`.

## Dev Proxy Defaults
Vite proxies:
- `/snake` -> `http://127.0.0.1:8770`
- `/maze` -> `http://127.0.0.1:8770`
- `/battle-grid` -> `http://127.0.0.1:8770`

Override target with:
- `VITE_SNAKE_API_URL`
- `VITE_MAZE_API_URL`
- `VITE_BATTLE_GRID_API_URL`

Example:
```bash
VITE_SNAKE_API_URL=http://127.0.0.1:9000 npm run dev
# or for maze
VITE_MAZE_API_URL=http://127.0.0.1:9001 npm run dev
```

## Backend API
```bash
SNAKE_REPLAY_DIR=recordings/snake MAZE_REPLAY_DIR=recordings/maze_race BATTLE_GRID_REPLAY_DIR=recordings/battle_grid SIMVERSE_API_HOST=127.0.0.1 SIMVERSE_API_PORT=8770 python -m server.main
```
