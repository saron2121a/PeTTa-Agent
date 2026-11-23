import pytest

from agent.memory import AgentMemory


def test_memory_is_saved_to_sqlite(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    saved = memory.save_memory(
        "semantic",
        "project",
        "uses",
        "PeTTa"
    )

    assert saved is True

    results = memory.search_memories("PeTTa")

    assert len(results) == 1
    assert results[0]["memory_type"] == "semantic"
    assert results[0]["subject"] == "project"
    assert results[0]["object"] == "PeTTa"


def test_duplicate_memory_is_not_saved_twice(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    first = memory.save_memory(
        "semantic", "agent", "uses", "Gemini"
    )

    second = memory.save_memory(
        "semantic", "agent", "uses", "Gemini"
    )

    assert first is True
    assert second is False

    results = memory.search_memories("Gemini")

    assert len(results) == 1


def test_memory_is_restored_into_petta(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    memory.save_memory(
        "semantic",
        "agent",
        "uses",
        "Gemini"
    )

    count = memory.restore_to_petta()

    assert count == 1

    result = memory.query_memory("semantic", "agent")

    assert result


def test_repeated_restore_does_not_duplicate_memories(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    memory.save_memory(
        "semantic",
        "agent",
        "uses",
        "Gemini"
    )

    first_count = memory.restore_to_petta()
    second_count = memory.restore_to_petta()

    assert first_count == 1
    assert second_count == 1

    result = memory.query_memory("semantic", "agent")

    assert result


def test_invalid_memory_type_is_rejected(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "memory.db")
    )

    with pytest.raises(ValueError, match="Invalid memory type"):
        memory.save_memory(
            "unknown",
            "agent",
            "uses",
            "Gemini"
        )


def test_all_five_memory_categories_are_restored(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "all_categories.db")
    )

    categories = {
        "context": ("project", "name", "PeTTa Agent"),
        "skill": ("agent", "can", "calculate"),
        "episodic": ("task_1", "completed", "successfully"),
        "semantic": ("agent", "uses", "MeTTa"),
        "procedural": ("recovery", "requires", "checkpoint"),
    }

    for memory_type, (subject, relation, obj) in categories.items():
        assert memory.save_memory(
            memory_type, subject, relation, obj
        ) is True

    restored_count = memory.restore_to_petta()

    assert restored_count == 5

    for memory_type, (subject, relation, obj) in categories.items():
        result = memory.query_memory(memory_type, subject)
        assert result, f"{memory_type} memory was not restored"


def test_relevant_memory_retrieval_ranks_matching_memories(tmp_path):
    memory = AgentMemory(db_path=str(tmp_path / "retrieval.db"))

    memory.save_memory("semantic", "agent", "uses", "MeTTa")
    memory.save_memory("skill", "agent", "can", "calculate")
    memory.save_memory("context", "project", "uses", "Redis cache")

    results = memory.get_relevant_memories(
        "agent MeTTa integration", limit=5
    )

    assert results
    assert results[0]["subject"] == "agent"
    assert results[0]["object"] == "MeTTa"
    assert all(
        result["object"] != "Redis cache" for result in results
    )


def test_relevant_memory_retrieval_respects_limit(tmp_path):
    memory = AgentMemory(db_path=str(tmp_path / "retrieval_limit.db"))

    for i in range(5):
        memory.save_memory("semantic", f"agent_{i}", "uses", "MeTTa")

    results = memory.get_relevant_memories(
        "agent uses MeTTa", limit=2
    )

    assert len(results) == 2


def test_relevant_memory_retrieval_rejects_invalid_input(tmp_path):
    memory = AgentMemory(
        db_path=str(tmp_path / "retrieval_validation.db")
    )

    with pytest.raises(ValueError):
        memory.get_relevant_memories("")

    with pytest.raises(ValueError):
        memory.get_relevant_memories("agent", limit=0)
