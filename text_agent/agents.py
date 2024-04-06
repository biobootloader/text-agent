import random
from abc import ABC, abstractmethod

from termcolor import cprint

from utils.gpt import GPTModelManager
from utils.history import History


class AgentInterface(ABC):
    @abstractmethod
    def show_state(
        chosen_action: str, observation: str, reward: int, score: int, valid_actions: list[str]
    ) -> None: ...

    @abstractmethod
    def choose_next_action(self) -> str: ...


class RandomAgent(AgentInterface):
    def __init__(self):
        self.valid_actions = []

    def show_state(
        self,
        chosen_action: str,
        observation: str,
        reward: int,
        score: int,
        valid_actions: list[str],
    ) -> None:
        self.valid_actions = valid_actions

    def choose_next_action(self) -> str:
        return random.choice(self.valid_actions)


class HumanAgent(AgentInterface):
    def show_state(
        self,
        chosen_action: str,
        observation: str,
        reward: int,
        score: int,
        valid_actions: list[str],
    ) -> None:
        pass

    def choose_next_action(
        self,
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

    def choose_next_action(self) -> str:
        model = "claude-3-sonnet-20240229"
        next_action = self.agent.get_response(
            model=model,
            prompt=self.history.get_formatted_history_for_next_action(),
            response_model=str,
        )
        return next_action

    def show_state(
        self,
        chosen_action: str,
        observation: str,
        reward: int,
        score: int,
        valid_actions: list[str],
    ) -> None:
        self.history.update_history(
            chosen_action=chosen_action,
            observation=observation,
            reward=reward,
            score=score,
            next_valid_actions=valid_actions,
        )
