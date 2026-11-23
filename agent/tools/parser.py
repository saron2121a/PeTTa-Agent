import json


class ToolCallParser:

    def parse(self, response):

        if isinstance(response, dict):
            data = response
        else:
            try:
                data = json.loads(response)
            except (json.JSONDecodeError, TypeError):
                return {
                    "type": "final",
                    "content": response
                }

        if not isinstance(data, dict):
            return {
                "type": "final",
                "content": response
            }

        if data.get("type") == "tool_call":

            # Support the new multiple-tool format.
            if "tool_calls" in data:
                tool_calls = data["tool_calls"]

            # Also support the old single-tool format.
            else:
                tool_calls = [{
                    "name": data.get("name"),
                    "arguments": data.get("arguments")
                }]

            if not isinstance(tool_calls, list) or not tool_calls:
                raise ValueError("Tool calls must be a non-empty list")

            for call in tool_calls:
                if not isinstance(call, dict):
                    raise ValueError("Each tool call must be an object")

                if not isinstance(call.get("name"), str):
                    raise ValueError("Tool name must be a string")

                if not isinstance(call.get("arguments"), dict):
                    raise ValueError("Tool arguments must be an object")

            return {
                "type": "tool_call",
                "tool_calls": tool_calls
            }

        return {
            "type": "final",
            "content": data.get("content", response)
        }
