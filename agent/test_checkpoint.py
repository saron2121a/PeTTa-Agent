from agent.checkpoint import CheckpointManager
from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def test_agent_resumes_interrupted_task(tmp_path):

    db_path = tmp_path / "resume.db"
    checkpoints = CheckpointManager(db_path=str(db_path))

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

    task_id = "resume-test-001"

    saved_messages = [
        {
            "role": "user",
            "content": "Calculate 5 + 3 and 4 * 2."
        },
        {
            "role": "assistant",
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
        },
        {
            "role": "tool",
            "name": "add",
            "content": "8"
        },
        {
            "role": "tool",
            "name": "multiply",
            "content": "8"
        }
    ]

    checkpoints.save_checkpoint(
        task_id=task_id,
        task="Calculate 5 + 3 and 4 * 2.",
        messages=saved_messages,
        current_step=1,
        status="running"
    )

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
    assert final_checkpoint["current_step"] == 2


def test_completed_task_cannot_be_resumed(tmp_path):

    checkpoints = CheckpointManager(
        db_path=str(tmp_path / "completed.db")
    )

    tools = ToolRegistry()

    agent = AgentLoop(
        llm=MockProvider(),
        tool_registry=tools,
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints
    )

    task_id = "completed-test-001"

    checkpoints.save_checkpoint(
        task_id=task_id,
        task="A completed task",
        messages=[{"role": "user", "content": "A completed task"}],
        current_step=1,
        status="completed"
    )

    import pytest

    with pytest.raises(ValueError, match="Task is already completed"):
        agent.run(
            task="",
            task_id=task_id,
            resume=True
        )