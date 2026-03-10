from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class MazeRaceConfig:
    width: int = 15
    height: int = 15
    num_agents: int = 2
    num_envs: int = 256
    max_steps: int = 200
    step_penalty: float = 0.0
    win_reward: float = 1.0
    lose_penalty: float = 1.0
    draw_reward: float = 0.0
    policies: List[Any] = field(default_factory=list)
