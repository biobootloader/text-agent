import instructor

from pydantic import BaseModel
import anthropic
import json

class GPTModelManager:
    def __init__(self, system_message: str = "You're a helpful AI assistant here to navigate game environment you're in."):
        """
        Initializes a new instance of the GPTModelManager.
        :param use_local: Determines whether to use a local model or an OpenAI model.
        """
        self.client = None
        self.initialize_client()
        self.system_message = system_message

    def initialize_client(self):
        """
        Initializes the GPT client based on the configuration.
        """
        client = instructor.from_anthropic(anthropic.Anthropic())
    
    def get_response(self, prompt: str, response_model: BaseModel, model: str = 'claude-3-sonnet-20240229') -> str:
        """
        Gets a response from the GPT model.
        :param prompt: The prompt to send to the GPT model.
        :return: The response from the GPT model.
        """
        response = self.client.messages.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt},
            ],
            response_model=response_model
        )
        # extract string response from response object
        return response

