from agent.checkpoint import CheckpointManager
from agent.loop import AgentLoop
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser


class MaxStepsProvider:

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
                    }
                ]
            }

        return "Task completed after resuming."


def create_registry():
    registry = ToolRegistry()

    def add(a, b):
        return a + b

    registry.register(
        name="add",
        description="Add two numbers",
        parameters={
            "a": {"description": "First number", "type": int},
            "b": {"description": "Second number", "type": int}
        },
        function=add
    )

    return registry


def test_agent_resumes_after_max_steps(tmp_path):

    checkpoints = CheckpointManager(
        db_path=str(tmp_path / "max_steps.db")
    )

    provider = MaxStepsProvider()

    agent = AgentLoop(
        llm=provider,
        tool_registry=create_registry(),
        parser=ToolCallParser(),
        checkpoint_manager=checkpoints,
        max_steps=1
    )

    task_id = "max-steps-001"

    # First execution should reach the maximum step limit.
    result = agent.run(
        "Add 5 and 3, then finish the task.",
        task_id=task_id
    )

    assert result == "Maximum agent steps reached."

    saved = checkpoints.load_checkpoint(task_id)

    assert saved["status"] == "failed"
    assert saved["current_step"] == 1

    # Resume the task.
    resumed_result = agent.run(
        task="",
        task_id=task_id,
        resume=True
    )

    assert resumed_result == "Task completed after resuming."

    final_checkpoint = checkpoints.load_checkpoint(task_id)

    assert final_checkpoint["status"] == "completed"
    assert provider.call_count == 2
