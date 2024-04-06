from abc import ABC, abstractmethod

class History(ABC):
    def __init__(self):
        self.history = []

    def update_history(self, last_observation, action_taken, immediate_reward, score):
        # format all of the data into a string for efficient prompting
        history_entry = f"Observation: {last_observation}\n \
        Action: {action_taken}\n \
        Reward: {immediate_reward}\n \
        Score: {score}"
        
        self.history.append(history_entry)

    def get_formatted_history_for_next_action(self, last_observation, valid_actions):
        return f"Observation: {last_observation}\n \
        Valid Actions: {valid_actions}\n \
        Action:"
    
    def export_history_to_file(self, filename):
        with open(filename, "w") as f:
            f.write("\n".join(self.history))

