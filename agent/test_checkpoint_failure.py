import pytest

from agent.checkpoint import CheckpointManager
from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def create_tools():
    tools = ToolRegistry()

    def add(a, b):
        return a + b

    def multiply(a, b):
        return a * b

    for name, description, function in [
        ("add", "Add two numbers", add),
        ("multiply", "Multiply two numbers", multiply)
    ]:
        tools.register(
            name=name,
            description=description,
            parameters={
                "a": {"description": "First number", "type": int},
                "b": {"description": "Second number", "type": int}
            },
            function=function
        )

    return tools


class FailingProvider(MockProvider):

    def generate(self, messages, available_tools):
        raise RuntimeError("Simulated provider failure")


def test_agent_recovers_after_provider_failure(tmp_path):
    checkpoints = CheckpointManager(
        db_path=str(tmp_path / "failure.db")
    )

    tools = create_tools()
    task_id = "failure-test-001"
    task = "Calculate 5 + 3."

    failing_agent = AgentLoop(
        llm=FailingProvider(),
        tool_registry=tools,
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    with pytest.raises(RuntimeError, match="Simulated provider failure"):
        failing_agent.run(task, task_id=task_id)

    checkpoint = checkpoints.load_checkpoint(task_id)

    assert checkpoint is not None
    assert checkpoint["status"] == "failed"

    resumed_agent = AgentLoop(
        llm=MockProvider(),
        tool_registry=tools,
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    result = resumed_agent.run(
        task="",
        task_id=task_id,
        resume=True
    )

    assert result == (
        "Both tools executed successfully. "
        "Addition: 8. Multiplication: 8."
    )

    final_checkpoint = checkpoints.load_checkpoint(task_id)

    assert final_checkpoint["status"] == "completed"
