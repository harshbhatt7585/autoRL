from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class SnakeConfig:
    width: int = 15
    height: int = 15
    num_agents: int = 1
    num_envs: int = 256
    max_steps: int = 300
    init_length: int = 3
    food_reward: float = 1.0
    crash_penalty: float = 1.0
    distance_reward_scale: float = 0.05
    survival_bonus: float = 1.0
    survival_bonus_every: int = 10
    auto_reset_done_envs: bool = True
    seed: int | None = None
    policies: List[Any] = field(default_factory=list)
