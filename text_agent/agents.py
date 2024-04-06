import random
from abc import ABC, abstractmethod

from termcolor import cprint

from utils.gpt import GPTModelManager
from utils.history import History


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
        return input("Enter your action: ")


class RawHistoryAgent(AgentInterface):
    def __init__(self):
        self.history = History()
        self.agent = GPTModelManager(
            system_message="You are an expert at text-based games. \
                                     You are trying to play a game, and \
                                     you are trying to figure out what the next best action should be based on context you are given."
        )

    def choose_next_action(self, last_observation: str, valid_actions: list[str]) -> str:
        model = "claude-3-sonnet-20240229"
        next_action = self.agent.get_response(
            model=model,
            prompt=self.history.get_formatted_history_for_next_action(
                last_observation, valid_actions
            ),
            response_model=str,
        )
        return next_action

    def update_history(
        self, last_observation: str, action_taken: str, immediate_reward: int, score: int
    ):
        self.history.add_to_history(last_observation, action_taken, immediate_reward, score)
