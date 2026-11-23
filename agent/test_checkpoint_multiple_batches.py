from agent.checkpoint import CheckpointManager
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser


class MultiBatchProvider:

    def __init__(self):
        self.call_count = 0

    def generate(self, messages, tools):
        self.call_count += 1

        if self.call_count == 1:
            return {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "name": "add",
                        "arguments": {"a": 5, "b": 3}
                    },
                    {
                        "name": "multiply",
                        "arguments": {"a": 4, "b": 2}
                    }
                ]
            }

        if self.call_count == 2:
            return {
                "type": "tool_call",
                "tool_calls": [
                    {
                        "name": "subtract",
                        "arguments": {"a": 10, "b": 3}
                    },
                    {
                        "name": "divide",
                        "arguments": {"a": 20, "b": 4}
                    }
                ]
            }

        return "All four tools completed successfully."


def create_tools(calls, interrupt):
    registry = ToolRegistry()

    def add(a, b):
        calls["add"] += 1
        return a + b

    def multiply(a, b):
        calls["multiply"] += 1
        return a * b

    def subtract(a, b):
        calls["subtract"] += 1
        return a - b

    def divide(a, b):
        calls["divide"] += 1

        if interrupt["enabled"]:
            interrupt["enabled"] = False
            raise KeyboardInterrupt("Simulated interruption")

        return a / b

    functions = [
        ("add", add),
        ("multiply", multiply),
        ("subtract", subtract),
        ("divide", divide)
    ]

    for name, function in functions:
        registry.register(
            name=name,
            description=f"Test {name}",
            parameters={
                "a": {
                    "description": "First number",
                    "type": int
                },
                "b": {
                    "description": "Second number",
                    "type": int
                }
            },
            function=function
        )

    return registry


def test_multiple_batches_resume_without_repeating_tools(tmp_path):

    checkpoints = CheckpointManager(
        db_path=str(tmp_path / "multiple_batches.db")
    )

    calls = {
        "add": 0,
        "multiply": 0,
        "subtract": 0,
        "divide": 0
    }

    interrupt = {"enabled": True}

    provider = MultiBatchProvider()

    registry = create_tools(calls, interrupt)

    agent = AgentLoop(
        llm=provider,
        tool_registry=registry,
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    task_id = "multiple-batches-resume-001"

    # First execution: simulate interruption during batch two.
    try:
        agent.run(
            "Calculate using four operations.",
            task_id=task_id
        )
        assert False, "Expected simulated interruption"

    except KeyboardInterrupt:
        pass

    # Verify the first batch completed exactly once.
    assert calls["add"] == 1
    assert calls["multiply"] == 1

    # Verify subtract completed before interruption.
    assert calls["subtract"] == 1

    # Divide was attempted but did not complete.
    assert calls["divide"] == 1

    saved = checkpoints.load_checkpoint(task_id)

    assert saved["status"] == "running"

    saved_tool_results = [
        message["name"]
        for message in saved["messages"]
        if message["role"] == "tool"
    ]

    assert saved_tool_results == [
        "add",
        "multiply",
        "subtract"
    ]

    # Resume the interrupted task.
    result = agent.run(
        task="",
        task_id=task_id,
        resume=True
    )

    assert result == "All four tools completed successfully."

    # Verify that completed tools were not repeated.
    assert calls == {
        "add": 1,
        "multiply": 1,
        "subtract": 1,
        "divide": 2
    }

    # The provider was called twice before interruption
    # and once after successful recovery.
    assert provider.call_count == 3

    final_checkpoint = checkpoints.load_checkpoint(task_id)

    assert final_checkpoint["status"] == "completed"

    final_tool_results = [
        message["name"]
        for message in final_checkpoint["messages"]
        if message["role"] == "tool"
    ]

    assert final_tool_results == [
        "add",
        "multiply",
        "subtract",
        "divide"
    ]