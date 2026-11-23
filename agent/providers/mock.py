
from agent.providers.base import BaseLLMProvider


class MockProvider(BaseLLMProvider):

    def __init__(self):
        self.call_count = 0
        self.received_messages = []

    def generate(self, messages, tools):

        self.call_count += 1

        # Keep a copy of the messages received by the provider.
        self.received_messages.append(messages.copy())

        user_content = messages[0]["content"]

        # Test direct answers.
        if "What is the capital of France?" in user_content:
            return "The capital of France is Paris."

        # Test memory retrieval and delivery.
        if "Continue working on the PeTTa project" in user_content:
            if "PeTTa symbolic reasoning" in user_content:
                return "Memory successfully received by the provider."

            return "Relevant memory was not received."
        # If both tool results already exist, finish the task.
        existing_results = [
            message["content"]
            for message in messages
            if message["role"] == "tool"
        ]

        if existing_results == ["8", "8"]:
            return (
                "Both tools executed successfully. "
                "Addition: 8. Multiplication: 8."
            )
        # First response: request two tools at once.
        if self.call_count == 1:
            return {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "name": "add",
                        "arguments": {
                            "a": 5,
                            "b": 3
                        }
                    },
                    {
                        "name": "multiply",
                        "arguments": {
                            "a": 4,
                            "b": 2
                        }
                    }
                ]
            }

        # Collect all tool results.
        tool_results = [
            message["content"]
            for message in messages
            if message["role"] == "tool"
        ]

        if tool_results == ["8", "8"]:
            return (
                "Both tools executed successfully. "
                "Addition: 8. Multiplication: 8."
            )

        return f"Unexpected tool results: {tool_results}"
