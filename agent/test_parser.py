import pytest

from agent.tools.parser import ToolCallParser


@pytest.fixture
def parser():
    return ToolCallParser()


def test_plain_text_is_final_answer(parser):
    result = parser.parse("Hello, how can I help?")

    assert result == {
        "type": "final",
        "content": "Hello, how can I help?"
    }


def test_json_final_answer(parser):
    result = parser.parse('{"type": "final", "content": "The answer is 42."}')

    assert result == {
        "type": "final",
        "content": "The answer is 42."
    }


def test_multiple_tool_calls(parser):
    response = {
        "type": "tool_call",
        "tool_calls": [
            {"name": "add", "arguments": {"a": 5, "b": 3}},
            {"name": "multiply", "arguments": {"a": 4, "b": 2}}
        ]
    }

    result = parser.parse(response)

    assert result["type"] == "tool_call"
    assert len(result["tool_calls"]) == 2
    assert result["tool_calls"][0]["name"] == "add"
    assert result["tool_calls"][1]["name"] == "multiply"


def test_single_tool_call_legacy_format(parser):
    response = {
        "type": "tool_call",
        "name": "add",
        "arguments": {"a": 5, "b": 3}
    }

    result = parser.parse(response)

    assert result == {
        "type": "tool_call",
        "tool_calls": [
            {"name": "add", "arguments": {"a": 5, "b": 3}}
        ]
    }


def test_json_string_tool_call(parser):
    response = (
        '{"type": "tool_call", '
        '"tool_calls": [{"name": "add", "arguments": {"a": 5, "b": 3}}]}'
    )

    result = parser.parse(response)

    assert result["type"] == "tool_call"
    assert result["tool_calls"][0]["name"] == "add"


def test_empty_tool_calls_are_rejected(parser):
    with pytest.raises(ValueError, match="non-empty list"):
        parser.parse({
            "type": "tool_call",
            "tool_calls": []
        })


def test_non_list_tool_calls_are_rejected(parser):
    with pytest.raises(ValueError, match="non-empty list"):
        parser.parse({
            "type": "tool_call",
            "tool_calls": "add"
        })


def test_invalid_tool_call_object_is_rejected(parser):
    with pytest.raises(ValueError, match="Each tool call"):
        parser.parse({
            "type": "tool_call",
            "tool_calls": ["not an object"]
        })


def test_missing_tool_name_is_rejected(parser):
    with pytest.raises(ValueError, match="Tool name must be a string"):
        parser.parse({
            "type": "tool_call",
            "tool_calls": [
                {"arguments": {"a": 5, "b": 3}}
            ]
        })


def test_invalid_tool_arguments_are_rejected(parser):
    with pytest.raises(ValueError, match="Tool arguments must be an object"):
        parser.parse({
            "type": "tool_call",
            "tool_calls": [
                {"name": "add", "arguments": "invalid"}
            ]
        })


def test_malformed_json_becomes_final_text(parser):
    response = '{"type": "tool_call"'

    result = parser.parse(response)

    assert result == {
        "type": "final",
        "content": response
    }


def test_non_dictionary_json_becomes_final_text(parser):
    response = '["one", "two"]'

    result = parser.parse(response)

    assert result == {
        "type": "final",
        "content": response
    }
