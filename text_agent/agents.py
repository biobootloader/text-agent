import random
import textwrap
from abc import ABC, abstractmethod

import anthropic
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
        system_message = textwrap.dedent("""\
            You are an expert at text-based games. You are trying to play a game, and
            you are trying to figure out what the next best action should be based on
            context you are given. You will always be given a list of valid actions.
            Your response should always consist of only one of the valid actions,
            written exactly as it appears in the list. Do not guess at actions that
            you think should be possible. Only choose an action on the latest list.""")

        self.history = History(system_message=system_message)
        self.agent = GPTModelManager(system_message=system_message)

    def choose_next_action(self) -> str:
        model = "claude-3-haiku-20240307"
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
        self.history.export_history_to_file("history.txt")


class ThinkingAgent(AgentInterface):
    def __init__(self):
        self.history = History(system_message="")
        self.anthropic_client = anthropic.Anthropic()

    def choose_next_action(self) -> str:
        system = textwrap.dedent("""\
            You are an expert at text-based games. You are trying to play a game, and
            you are trying to figure out what the next best action should be based on
            context you are given. You will always be given a list of valid actions.

            Begin your response with a few sentences of thinking through your choices.
            Then output two
            newlines and then the exact text of your chosen action. Write nothing
            after outputing your chosen action. It should be the
            only text on the final line. Do not guess at actions you think should be
            possible. Only choose an action on the latest list.

            Example response:

            Thinking: As I've explored east already, I will go west now.

            west

            Another example response:

            Thinking: I will try to find a key to open the door.

            open box
            """)

        # model = "claude-3-haiku-20240307"
        model = "claude-3-sonnet-20240229"

        response = self.anthropic_client.messages.create(
            system=system,
            model=model,
            messages=[
                {"role": "user", "content": self.history.get_formatted_history_for_next_action()}
            ],
            max_tokens=1000,
        )

        text = response.content[0].text
        cprint(text, "light_blue")
        action = text.splitlines()[-1].strip()
        return action

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
        self.history.export_history_to_file("history.txt")
