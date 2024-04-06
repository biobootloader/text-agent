from abc import ABC, abstractmethod


class AgentInterface(ABC):
    @abstractmethod
    def choose_next_action(
        self, last_observation: str, valid_actions: list[str], immediate_reward: int, score: int
    ) -> str: ...
