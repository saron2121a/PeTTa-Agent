from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def test_agent_executes_multiple_tools():

    tools = ToolRegistry()

    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    tools.register(
        name="add",
        description="Add two numbers",
        parameters={
            "a": {"description": "First number", "type": int},
            "b": {"description": "Second number", "type": int}
        },
        function=add
    )

    tools.register(
        name="multiply",
        description="Multiply two numbers",
        parameters={
            "a": {"description": "First number", "type": int},
            "b": {"description": "Second number", "type": int}
        },
        function=multiply
    )

    agent = AgentLoop(
        llm=MockProvider(),
        tool_registry=tools,
        parser=ToolCallParser()
    )

    result = agent.run("Calculate 5 + 3 and 4 * 2.")

    assert result == (
        "Both tools executed successfully. "
        "Addition: 8. Multiplication: 8."
    )