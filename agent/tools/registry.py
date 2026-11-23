
class ToolRegistry:

    def __init__(self):
        self.tools = {}

    def register(
        self,
        name,
        description,
        parameters,
        function
    ):
        if name in self.tools:
            raise ValueError(f"Tool already registered: {name}")

        if not callable(function):
            raise ValueError("Tool function must be callable")

        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "function": function
        }

    def execute(self, name, arguments):
        if name not in self.tools:
            raise ValueError(f"Unknown tool: {name}")

        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a dictionary")

        tool = self.tools[name]
        expected = set(tool["parameters"])
        received = set(arguments)

        missing = expected - received
        unexpected = received - expected

        if missing:
            raise ValueError(
                f"Missing arguments: {', '.join(sorted(missing))}"
            )

        if unexpected:
            raise ValueError(
                f"Unexpected arguments: {', '.join(sorted(unexpected))}"
            )

        for name, value in arguments.items():
            expected_type = tool["parameters"][name]["type"]

            if expected_type is int:
                valid_type = type(value) is int
            else:
                valid_type = isinstance(value, expected_type)

            if not valid_type:
                raise TypeError(
                    f"Argument '{name}' must be "
                    f"{expected_type.__name__}, "
                    f"not {type(value).__name__}"
                )

        return tool["function"](**arguments)

    def describe_tools(self):
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
            for tool in self.tools.values()
        ]
