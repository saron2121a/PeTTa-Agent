import pytest

from agent.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    tools = ToolRegistry()

    tools.register(
        name="add",
        description="Add two numbers",
        parameters={
            "a": {"description": "First number", "type": int},
            "b": {"description": "Second number", "type": int}
        },
        function=lambda a, b: a + b
    )

    return tools


def test_register_and_execute_tool(registry):
    result = registry.execute(
        "add",
        {"a": 5, "b": 3}
    )

    assert result == 8


def test_unknown_tool_is_rejected(registry):
    with pytest.raises(ValueError, match="Unknown tool"):
        registry.execute("divide", {"a": 8, "b": 2})


def test_missing_argument_is_rejected(registry):
    with pytest.raises(ValueError, match="Missing arguments"):
        registry.execute("add", {"a": 5})


def test_unexpected_argument_is_rejected(registry):
    with pytest.raises(ValueError, match="Unexpected arguments"):
        registry.execute("add", {"a": 5, "b": 3, "c": 2})


def test_wrong_argument_type_is_rejected(registry):
    with pytest.raises(TypeError, match="must be int"):
        registry.execute("add", {"a": "five", "b": 3})


def test_boolean_is_rejected_as_integer(registry):
    with pytest.raises(TypeError, match="must be int"):
        registry.execute("add", {"a": True, "b": 3})


def test_non_dictionary_arguments_are_rejected(registry):
    with pytest.raises(ValueError, match="must be a dictionary"):
        registry.execute("add", [5, 3])


def test_duplicate_tool_registration_is_rejected(registry):
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            name="add",
            description="Another addition tool",
            parameters={},
            function=lambda: 0
        )


def test_non_callable_function_is_rejected():
    tools = ToolRegistry()

    with pytest.raises(ValueError, match="must be callable"):
        tools.register(
            name="invalid",
            description="Invalid tool",
            parameters={},
            function="not a function"
        )


def test_describe_tools_returns_public_information(registry):
    descriptions = registry.describe_tools()

    assert len(descriptions) == 1
    assert descriptions[0]["name"] == "add"
    assert descriptions[0]["description"] == "Add two numbers"
    assert "function" not in descriptions[0]
