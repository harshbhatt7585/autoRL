from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__.startswith("__main__"):
    _src = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_src))

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np

from simverse.utils.replay_buffer import Experience

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None
    _TORCH_AVAILABLE = False

try:
    import wandb  # type: ignore

    _WANDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    wandb = None
    _WANDB_AVAILABLE = False


def _json_default(obj: Any) -> Any:
    if _TORCH_AVAILABLE and isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


@dataclass
class TrainingStats:
    experiences: List[Experience] = field(default_factory=list)
    steps: int = 0
    episode_count: int = 0
    step_rewards: List[float] = field(default_factory=list)  # Per-step rewards
    episode_rewards: List[float] = field(default_factory=list)  # Total episode rewards
    episode_steps: List[int] = field(default_factory=list)
    episode_harvested_tiles: List[float] = field(default_factory=list)
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    agent_metrics: DefaultDict[int, DefaultDict[str, List[float]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )
    current_episode_frames: List[Dict[str, Any]] = field(default_factory=list)
    env_count: int = 1

    def push_experience(self, experience: Experience) -> None:
        self.experiences.append(experience)
        # Track per-step reward
        reward = experience.reward
        if isinstance(reward, dict):
            reward = sum(reward.values())
        numeric_reward = float(reward) if reward else 0.0
        self.step_rewards.append(numeric_reward)
        self._record_agent_metric(experience.agent_id, "reward", numeric_reward)

    def push_agent_losses(self, agent_id: int, policy_loss: float, value_loss: float) -> None:
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self._record_agent_metric(agent_id, "loss/policy", policy_loss)
        self._record_agent_metric(agent_id, "loss/value", value_loss)

    def _record_agent_metric(self, agent_id: int, metric: str, value: float) -> None:
        self.agent_metrics[agent_id][metric].append(value)

    def record_frame(self, frame: Dict[str, Any]) -> None:
        self.current_episode_frames.append(frame)

    def push_reward(self, reward: float, env_count: int | None = None) -> None:
        """Push total episode reward (normalized per environment)."""
        if env_count and env_count > 0:
            normalized_reward = reward / env_count
        else:
            normalized_reward = reward
        self.episode_rewards.append(normalized_reward)
        self.episode_count += 1

    def push_episode_metrics(
        self,
        *,
        steps: int | None = None,
        harvested_tiles: float | None = None,
    ) -> None:
        if steps is not None:
            self.episode_steps.append(int(steps))
        if harvested_tiles is not None:
            self.episode_harvested_tiles.append(float(harvested_tiles))

    def step(self, increment: int = 1) -> None:
        self.steps += max(int(increment), 1)

    def set_env_count(self, count: int) -> None:
        if count > 0:
            self.env_count = count

    def log_wandb(self, step: Optional[int] = None) -> None:
        if not _WANDB_AVAILABLE or getattr(wandb, "run", None) is None:
            return
        payload = {}
        payload["trainer/steps"] = self.steps
        payload["trainer/episodes"] = self.episode_count

        # Step-level rewards
        if self.step_rewards:
            payload["step/reward"] = self.step_rewards[-1]
        for agent_id, metrics in self.agent_metrics.items():
            reward_history = metrics.get("reward")
            if reward_history:
                payload[f"agent/{agent_id}/reward"] = reward_history[-1]

        # Episode-level rewards
        if self.episode_rewards:
            payload["episode/reward"] = self.episode_rewards[-1]
            payload["episode/reward_avg"] = sum(self.episode_rewards) / len(self.episode_rewards)
        if self.episode_steps:
            payload["episode/steps"] = self.episode_steps[-1]
        if self.episode_harvested_tiles:
            payload["episode/harvested_tiles"] = self.episode_harvested_tiles[-1]
        if self.experiences:
            last = self.experiences[-1]
            payload["episode/done"] = np.float32(last.done).item()

        if self.policy_losses:
            payload["loss/policy"] = self.policy_losses[-1]
            payload["loss/policy_avg"] = sum(self.policy_losses) / len(self.policy_losses)
        for agent_id, metrics in self.agent_metrics.items():
            policy_losses = metrics.get("loss/policy")
            if policy_losses:
                payload[f"agent/{agent_id}/loss/policy"] = policy_losses[-1]

        if self.value_losses:
            payload["loss/value"] = self.value_losses[-1]
            payload["loss/value_avg"] = sum(self.value_losses) / len(self.value_losses)
        for agent_id, metrics in self.agent_metrics.items():
            value_losses = metrics.get("loss/value")
            if value_losses:
                payload[f"agent/{agent_id}/loss/value"] = value_losses[-1]

        wandb.log(payload, step=self.steps if step is None else step)

    def reset_episode(self) -> None:
        """Reset episode-level stats."""
        self.step_rewards.clear()
        self.experiences.clear()
        self.current_episode_frames.clear()

    def dump_episode_recording(
        self,
        output_dir: str | Path,
        episode: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "episode": episode,
            "steps": self.steps,
            "agent_metrics": {
                agent_id: {metric: values for metric, values in metrics.items()}
                for agent_id, metrics in self.agent_metrics.items()
            },
            "frames": self.current_episode_frames,
        }
        if metadata:
            payload["metadata"] = metadata
        output_path = output_dir / f"episode_{episode:04d}.json"
        output_path.write_text(json.dumps(payload, indent=2, default=_json_default))
        self.current_episode_frames.clear()
        return output_path


if __name__ == "__main__":
    import time

    import numpy as np
    import wandb

    wandb.init(project="simverse", name="stats-test")
    stats = TrainingStats()
    total_steps = 100
    wait_time = 0.3
    for step in range(total_steps):
        time.sleep(wait_time)
        stats.step()
        stats.push_experience(
            Experience(
                agent_id=0,
                observation=np.array([1, 2, 3]),
                action=np.array([4, 5, 6]),
                log_prob=np.array([7, 8, 9]),
                value=np.array([10, 11, 12]),
                reward=0.5,  # Simple float reward
                done=False,
                info={},
            )
        )
        stats.push_agent_losses(agent_id=0, policy_loss=0.1 * step, value_loss=0.05 * step)
        stats.log_wandb(step=stats.steps)
    wandb.finish()
