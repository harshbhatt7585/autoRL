from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class BattleGridConfig:
    width: int = 13
    height: int = 13
    num_agents: int = 2
    num_envs: int = 512
    max_steps: int = 200
    max_health: int = 3
    attack_damage: int = 1
    attack_range: int = 1
    step_penalty: float = 0.01
    damage_reward: float = 0.05
    kill_reward: float = 1.0
    death_penalty: float = 1.0
    timeout_win_reward: float = 0.5
    timeout_lose_penalty: float = 0.5
    draw_reward: float = 0.0
    policies: List[Any] = field(default_factory=list)
