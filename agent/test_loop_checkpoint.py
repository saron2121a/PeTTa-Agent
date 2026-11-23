import pytest

from agent.checkpoint import CheckpointManager
from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def test_agent_saves_completed_checkpoint(tmp_path):

    db_path = tmp_path / "checkpoints.db"
    checkpoints = CheckpointManager(db_path=str(db_path))

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
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    task_id = "checkpoint-test-001"

    result = agent.run(
        "Calculate 5 + 3 and 4 * 2.",
        task_id=task_id
    )

    assert result == (
        "Both tools executed successfully. "
        "Addition: 8. Multiplication: 8."
    )

    saved = checkpoints.load_checkpoint(task_id)

    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["current_step"] == 2

    tool_results = [
        message["content"]
        for message in saved["messages"]
        if message["role"] == "tool"
    ]

    assert tool_results == ["8", "8"]