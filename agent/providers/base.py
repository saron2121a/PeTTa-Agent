from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):

    @abstractmethod
    def generate(self, messages, tools):
        """
        Send messages and available tool descriptions
        to an LLM provider.

        Return a normalized response:
        - final answer
        - tool call
        """
        pass
