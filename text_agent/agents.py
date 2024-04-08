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
        self.last_valid_actions = []

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

            ```Thinking: As I've explored east already, I will go west now.

            west```

            Another example response:

            ```Thinking: I will try to find a key to open the door.

            open box```
            """)

        # model = "claude-3-haiku-20240307"
        model = "claude-3-sonnet-20240229"
        # model = "claude-3-opus-20240229"

        response = self.anthropic_client.messages.create(
            system=system,
            model=model,
            messages=[
                {"role": "user", "content": self.history.get_formatted_history_for_next_action()}
            ],
            max_tokens=1000,
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cprint(f"Input and Output tokens: {input_tokens} {output_tokens}", "yellow")

        self.history.update_history_with_string(text)
        cprint(text, "light_blue")
        action = text.splitlines()[-1].strip()

        if action not in self.last_valid_actions:
            cprint(f"Invalid action: {action}", "red")
            return random.choice(self.last_valid_actions)

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
        self.last_valid_actions = valid_actions


class SummaryAgent(AgentInterface):
    def __init__(self):
        self.history = History(system_message="")
        self.anthropic_client = anthropic.Anthropic()
        self.last_valid_actions = []

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

            ```Thinking: As I've explored east already, I will go west now.

            west```

            Another example response:

            ```Thinking: I will try to find a key to open the door.

            open box```
            """)

        # model = "claude-3-haiku-20240307"
        model = "claude-3-sonnet-20240229"
        # model = "claude-3-opus-20240229"

        response = self.anthropic_client.messages.create(
            system=system,
            model=model,
            messages=[
                {"role": "user", "content": self.history.get_formatted_history_for_next_action()}
            ],
            max_tokens=1000,
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cprint(f"Input and Output tokens: {input_tokens} {output_tokens}", "yellow")

        self.history.update_history_with_string(text)
        cprint(text, "light_blue")
        action = text.splitlines()[-1].strip()

        if action not in self.last_valid_actions:
            cprint(f"Invalid action: {action}", "red")
            return random.choice(self.last_valid_actions)

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
        self.last_valid_actions = valid_actions


class MentalMapAgent(AgentInterface):
    def __init__(self):
        self.history = History(system_message="")
        self.anthropic_client = anthropic.Anthropic()
        self.last_valid_actions = []

    def choose_next_action(self) -> str:
        system = textwrap.dedent("""\
            You are an expert at text-based games. You are trying to play a game, and
            you are trying to figure out what the next best action should be based on
            context you are given. You will always be given a list of valid actions.
            YOUR GOAL IS TO MAXIMIZE YOUR SCORE.
            EXPLORING MORE REWARDING AREAS MIGHT HELP THAN LOOPING AROUND UNREWARDING AREAS.

            Begin your response re-iterate mental map and update last mental map if needed.
            then, output a few sentences of thinking through your choices.
            Then output two
            newlines and then the exact text of your chosen action. Write nothing
            after outputing your chosen action. It should be the
            only text on the final line. Do not guess at actions you think should be
            possible. Only choose an action on the latest list.

            FOLLOW THIS FORMAT STRICTLY:
            ---------
            Mental Map:
                - Nodes:
                - Current Location: Room A
                - Interactions:
                    - Explored east (Room B)
                    - Found a locked door in Room B
                - Objects:
                    - Box in Room A
                - Edges:
                - (Current Location: Room A) -- (Explored east) --> (Room B)
                - (Room B) -- (Found) --> (Locked Door)
                - (Current Location: Room A) -- (Contains) --> (Box)
            -------
            Thinking: Based on the mental map, I have explored Room B to the east and found a locked door. Since I'm back in Room A and there is a box here, I will search the box for a potential key to unlock the door in Room B.
            
            NEXT BEST ACTION FROM VALID NEXT ACTIONS:
            open box""")

        model = "claude-3-haiku-20240307"
        # model = "claude-3-sonnet-20240229"
        # model = "claude-3-opus-20240229"

        response = self.anthropic_client.messages.create(
            system=system,
            model=model,
            messages=[
                {"role": "user", "content": self.history.get_latest_history_entry()}
            ],
            max_tokens=1000,
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cprint(f"Input and Output tokens: {input_tokens} {output_tokens}", "yellow")

        self.history.update_history_with_string(text)
        cprint(text, "light_blue")
        action = text.splitlines()[-1].strip()

        if action not in self.last_valid_actions:
            cprint(f"Invalid action: {action}", "red")
            return random.choice(self.last_valid_actions)

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
        self.last_valid_actions = valid_actions


class MentalMapAgentB(AgentInterface):
    def __init__(self):
        self.history = History(system_message="")
        self.anthropic_client = anthropic.Anthropic()
        self.last_valid_actions = []

    def choose_next_action(self) -> str:
        system = textwrap.dedent("""\
            You are an expert AI agent navigating a text-based game environment. Your goal is to maximize your
            score by making strategic decisions based on the information provided to you.

            You are given the current observation of the game state and the list of valid actions you can take.
            You also have access to your mental map and your reasoning process from last action.
            
            Begin by analyzing the current game state and updating your mental map of the environment in the
            following format:

            <mental_map>
            Mental Map:
            - Nodes:
            - Current Location: [Room/Area]
            - [Other Rooms/Areas]
            - Interactions:
            - [Interaction 1]
            - [Interaction 2]
            - ...
            - Objects:
            - [Object 1]
            - [Object 2]
            - ...
            - Edges:
            - (Current Location) -- [Interaction/Connection] --> [Connected Room/Area]
            - [Room/Area 1] -- [Interaction/Connection] --> [Room/Area 2]
            - ...
            </mental_map>

            Next, think through potential strategies and evaluate the best next action. Consider factors such
            as:
            - Exploring new areas that may lead to higher rewards
            - Investigating objects or interactions that could provide useful items or information
            - Avoiding actions that may lead to negative consequences or looping behavior

            Write your thought process inside <strategy_thinking> tags.

            Do not guess at actions you think should be possible. Only choose an action from the latest list of
            valid actions.

            Remember, your ultimate goal is to MAXIMIZE YOUR SCORE by making strategic decisions and exploring
            rewarding areas of the game environment.
            Finally, choose the next best action from the list of valid actions. Output your chosen action in
            the following strict format:
                                 
            NEXT BEST ACTION FROM VALID NEXT ACTIONS:
            open box""")

        model = "claude-3-haiku-20240307"
        # model = "claude-3-sonnet-20240229"
        # model = "claude-3-opus-20240229"

        response = self.anthropic_client.messages.create(
            system=system,
            model=model,
            messages=[
                {"role": "user", "content": self.history.get_latest_history_entry()}
            ],
            max_tokens=1000,
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cprint(f"Input and Output tokens: {input_tokens} {output_tokens}", "yellow")

        self.history.update_history_with_string(text)
        cprint(text, "light_blue")
        action = text.splitlines()[-1].strip()

        if action not in self.last_valid_actions:
            cprint(f"Invalid action: {action}", "red")
            return random.choice(self.last_valid_actions)

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
        self.last_valid_actions = valid_actions

