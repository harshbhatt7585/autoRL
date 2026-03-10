import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Experience:
    agent_id: int
    observation: Any
    action: Any
    log_prob: Any
    value: Any
    reward: Any
    done: Any
    info: Dict[str, Any]


class ReplayBuffer:
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, experience: Experience) -> None:
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def sample_for_agent(self, agent_id: int, batch_size: int) -> List[Experience]:
        agent_experiences = [exp for exp in self.buffer if exp.agent_id == agent_id]
        if not agent_experiences:
            return []
        if len(agent_experiences) <= batch_size:
            return agent_experiences.copy()
        # PPO expects ordered on-policy trajectories for GAE; return most recent window.
        return agent_experiences[-batch_size:]

    def clear(self) -> None:
        self.buffer.clear()
