# Server

Server-side modules used by the frontend replay UI live here.

Single server entrypoint:
- `server/main.py` - central FastAPI app.

Routers:
- `server/snake/router.py` - snake endpoints mounted under `/snake`:
  - `GET /snake/`
  - `GET /snake/snapshot`
  - `GET /snake/replays`
  - `GET /snake/replays/{replay_id}`
- `server/maze/router.py` - maze endpoints mounted under `/maze`:
  - `GET /maze/`
  - `GET /maze/snapshot`
  - `GET /maze/replays`
  - `GET /maze/replays/{replay_id}`

Compatibility:
- Use `server/main.py` as the single server entrypoint.
