from agent.memory import AgentMemory
from agent.providers.mock import MockProvider
from agent.tools.registry import ToolRegistry
from agent.tools.parser import ToolCallParser
from agent.loop import AgentLoop


def test_agent_retrieves_and_uses_relevant_memory(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    memory.save_memory(
        "semantic",
        "project name",
        "uses",
        "PeTTa symbolic reasoning"
    )

    provider = MockProvider()

    tools = ToolRegistry()

    agent = AgentLoop(
        llm=provider,
        tool_registry=tools,
        parser=ToolCallParser(),
        memory=memory
    )

    result = agent.run(
        "Continue working on the PeTTa project"
    )

    assert result == "Memory successfully received by the provider."

    # Confirm that the provider received the stored memory.
    received_content = provider.received_messages[0][0]["content"]

    assert "PeTTa symbolic reasoning" in received_content
    assert "Current task:" in received_content
