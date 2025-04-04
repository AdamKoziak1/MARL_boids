

from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    max_steps: int = 100
    n_agents: int = 5


