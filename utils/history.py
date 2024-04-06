import textwrap
from abc import ABC, abstractmethod


class History(ABC):
    def __init__(self, system_message):
        self.system_message = system_message
        self.history = []

    def update_history(self, chosen_action, observation, reward, score, next_valid_actions):
        history_entry = textwrap.dedent(f"""\
Chosen Action: {chosen_action}
Led to this Observation: {observation}
with Reward: {reward}
with Total Score: {score}
Valid Actions: {next_valid_actions}""")

        self.history.append(f"\n\n{history_entry}\n\n")

    def update_history_with_string(self, string):
        self.history.append(string)

    def get_formatted_history_for_next_action(self):
        return self.system_message + "\n" + "\n".join(self.history)

    def export_history_to_file(self, filename):
        with open(filename, "w") as f:
            f.write(self.get_formatted_history_for_next_action())
