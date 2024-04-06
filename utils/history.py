from abc import ABC, abstractmethod


class History(ABC):
    def __init__(self):
        self.history = []

    def update_history(self, chosen_action, observation, reward, score, next_valid_actions):
        # format all of the data into a string for efficient prompting
        history_entry = f"Chosen Action: {chosen_action}\n \
        Led to this Observation: {observation}\n \
        with Reward: {reward}\n \
        with Total Score: {score}\n \
        Valid Actions: {next_valid_actions}"

        self.history.append(history_entry)

    def get_formatted_history_for_next_action(self):
        # pass all of the history to the agent
        return "\n".join(self.history) + "\n\nAction:"

    def export_history_to_file(self, filename):
        with open(filename, "w") as f:
            f.write("\n".join(self.history))
