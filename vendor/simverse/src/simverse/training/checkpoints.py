from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from simverse.core.env import SimEnv


class Checkpointer:
    def __init__(self, env: SimEnv, run_id: str | None = None):
        self.env = env
        self.run_id = run_id or self._generate_run_id()
        self.run_directory: Path | None = None

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"run_{timestamp}_{uuid4().hex[:8]}"

    def resolve_checkpoint_path(self, state_dict_path: str) -> Path:
        requested_path = Path(state_dict_path)
        checkpoint_name = requested_path.name
        base_directory = requested_path.parent if requested_path.parent != Path("") else Path(".")
        if self.run_directory is None:
            self.run_directory = base_directory / self.run_id
        return self.run_directory / checkpoint_name

    def save(self, state_dict_path: str) -> Path:
        agents = self.env.agents
        resolved_path = self.resolve_checkpoint_path(state_dict_path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        farmtila_state_dict = {
            "env_config": self.env.config,
            "agents": [
                {"agent_id": agent.agent_id, "policy_state_dict": agent.policy.state_dict()}
                for agent in agents
            ],
            "steps": self.env.steps,
        }
        with resolved_path.open("wb") as f:
            pickle.dump(farmtila_state_dict, f)
        return resolved_path

    def load(self, state_dict_path: str) -> None:
        with Path(state_dict_path).open("rb") as f:
            farmtila_state_dict = pickle.load(f)
            self.env.config = farmtila_state_dict["env_config"]
            for agent in farmtila_state_dict["agents"]:
                self.env.agents[agent["agent_id"]].policy.load_state_dict(
                    agent["policy_state_dict"]
                )
