import random
from abc import ABC, abstractmethod

from termcolor import cprint


class AgentInterface(ABC):
    @abstractmethod
    def choose_next_action(
        self, last_observation: str, valid_actions: list[str], immediate_reward: int, score: int
    ) -> str: ...


class RandomAgent(AgentInterface):
    def choose_next_action(
        self, last_observation: str, valid_actions: list[str], immediate_reward: int, score: int
    ) -> str:
        return random.choice(valid_actions)


class HumanAgent(AgentInterface):
    def choose_next_action(
        self, last_observation: str, valid_actions: list[str], immediate_reward: int, score: int
    ) -> str:
        cprint(f"Observation: {last_observation}", "green")
        cprint(f"Reward and Score: {immediate_reward} {score}", "magenta")
        cprint(f"Valid Actions: {valid_actions}", "green")
        return input("Enter your action: ")


class RawHistoryAgent(AgentInterface):
    def __init__(self):
        self.history = []

    def choose_next_action(
        self, last_observation: str, valid_actions: list[str], immediate_reward: int, score: int
    ) -> str:
        self.history.append((last_observation, valid_actions, immediate_reward, score))
        return random.choice(valid_actions)

    def get_history(self):
        return self.history
