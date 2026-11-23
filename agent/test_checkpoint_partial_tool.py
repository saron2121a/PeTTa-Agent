from agent.checkpoint import CheckpointManager
from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def test_agent_resumes_after_partial_tool_execution(tmp_path):
    checkpoints = CheckpointManager(
        db_path=str(tmp_path / "partial.db")
    )

    tools = ToolRegistry()

    calls = {
        "add": 0,
        "multiply": 0,
        "interrupt": True
    }

    def add(a, b):
        calls["add"] += 1
        return a + b

    def multiply(a, b):
        if calls["interrupt"]:
            calls["interrupt"] = False
            raise KeyboardInterrupt("Simulated interruption")

        calls["multiply"] += 1
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

    task_id = "partial-tool-test-001"
    task = "Calculate 5 + 3 and 4 * 2."

    agent = AgentLoop(
        llm=MockProvider(),
        tool_registry=tools,
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    try:
        agent.run(task, task_id=task_id)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Expected simulated interruption")

    checkpoint = checkpoints.load_checkpoint(task_id)

    assert checkpoint is not None
    assert checkpoint["status"] == "running"

    completed_tools = [
        message["name"]
        for message in checkpoint["messages"]
        if message["role"] == "tool"
    ]

    assert completed_tools == ["add"]
    assert calls["add"] == 1
    assert calls["multiply"] == 0

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

    assert calls["add"] == 1
    assert calls["multiply"] == 1

    final_checkpoint = checkpoints.load_checkpoint(task_id)

    assert final_checkpoint["status"] == "completed"
