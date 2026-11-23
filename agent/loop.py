import uuid


class AgentLoop:

    def __init__(
        self,
        llm,
        tool_registry,
        parser,
        memory=None,
        max_steps=10,
        checkpoint_manager=None
    ):
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")

        self.llm = llm
        self.tools = tool_registry
        self.parser = parser
        self.memory = memory
        self.max_steps = max_steps
        self.checkpoint_manager = checkpoint_manager
        self.last_task_id = None

    def _save_checkpoint(
        self,
        task_id,
        task,
        messages,
        current_step,
        status
    ):
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.save_checkpoint(
                task_id,
                task,
                messages,
                current_step,
                status
            )

    def _get_pending_tool_batch(self, messages):
        """
        Find the latest assistant tool-call batch
        and its associated results.
        """

        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]

            if message.get("role") == "assistant":

                if "tool_calls" not in message:
                    return None

                tool_calls = message["tool_calls"]

                results = [
                    item
                    for item in messages[index + 1:]
                    if item.get("role") == "tool"
                ]

                return tool_calls, results

        return None

    def _execute_pending_tools(
        self,
        tool_calls,
        messages,
        task_id,
        task,
        step
    ):
        pending_batch = self._get_pending_tool_batch(messages)

        if pending_batch is None:
            raise ValueError("No pending tool-call batch found")

        batch_calls, existing_results = pending_batch

        if len(existing_results) > len(batch_calls):
            raise ValueError("More tool results than tool calls")

        # Verify that saved results belong to this batch.
        for index, result in enumerate(existing_results):
            if result.get("name") != batch_calls[index].get("name"):
                raise ValueError("Tool result does not match its call")

        # Execute only tools without saved results.
        for index, call in enumerate(tool_calls):

            if index < len(existing_results):
                continue

            try:
                result = self.tools.execute(
                    call["name"],
                    call["arguments"]
                )

                result_content = str(result)

            except Exception as error:
                result_content = (
                    f"Tool execution failed: "
                    f"{type(error).__name__}: {error}"
                )

            messages.append({
                "role": "tool",
                "name": call["name"],
                "content": result_content
            })

            # Save immediately after each tool result.
            self._save_checkpoint(
                task_id,
                task,
                messages,
                step + 1,
                "running"
            )

    def run(self, task, task_id=None, resume=False):

        if resume:

            if self.checkpoint_manager is None:
                raise ValueError(
                    "Checkpoint manager is not configured"
                )

            if not task_id:
                raise ValueError("Task ID is required to resume")

            checkpoint = self.checkpoint_manager.load_checkpoint(
                task_id
            )

            if checkpoint is None:
                raise ValueError("Checkpoint not found")

            if checkpoint["status"] == "completed":
                raise ValueError("Task is already completed")

            task = checkpoint["task"]
            messages = checkpoint["messages"]
            start_step = checkpoint["current_step"]

            # Give a resumed execution a fresh step allowance
            # if the previous execution exhausted its allowance.
            if start_step >= self.max_steps:
                start_step = 0

        else:

            if not isinstance(task, str) or not task.strip():
                raise ValueError("Task must be a non-empty string")

            if task_id is None:
                task_id = uuid.uuid4().hex

            relevant_memories = []

            if self.memory is not None:
                relevant_memories = (
                    self.memory.get_relevant_memories(task)
                )

            if relevant_memories:
                memory_context = "\n".join(
                    f"- [{item['memory_type']}] "
                    f"{item['subject']} {item['relation']} "
                    f"{item['object']}"
                    for item in relevant_memories
                )

                user_content = (
                    "Relevant stored memories:\n"
                    f"{memory_context}\n\n"
                    "Current task:\n"
                    f"{task}"
                )
            else:
                user_content = task

            messages = [
                {
                    "role": "user",
                    "content": user_content
                }
            ]

            start_step = 0

        self.last_task_id = task_id

        self._save_checkpoint(
            task_id,
            task,
            messages,
            start_step,
            "running"
        )

        # If resuming, finish an interrupted tool batch first.
        if resume:
            pending_batch = self._get_pending_tool_batch(messages)

            if pending_batch is not None:
                pending_calls, _ = pending_batch

                try:
                    self._execute_pending_tools(
                        pending_calls,
                        messages,
                        task_id,
                        task,
                        max(start_step - 1, 0)
                    )

                except Exception:
                    self._save_checkpoint(
                        task_id,
                        task,
                        messages,
                        start_step,
                        "failed"
                    )
                    raise

        for step in range(start_step, self.max_steps):

            try:
                response = self.llm.generate(
                    messages,
                    self.tools.describe_tools()
                )

                parsed = self.parser.parse(response)

            except Exception:
                self._save_checkpoint(
                    task_id,
                    task,
                    messages,
                    step,
                    "failed"
                )
                raise

            if parsed["type"] == "final":

                messages.append({
                    "role": "assistant",
                    "content": parsed["content"]
                })

                self._save_checkpoint(
                    task_id,
                    task,
                    messages,
                    step + 1,
                    "completed"
                )

                return parsed["content"]

            if parsed["type"] == "tool_call":

                tool_calls = parsed["tool_calls"]

                # Save the entire batch before execution.
                messages.append({
                    "role": "assistant",
                    "tool_calls": tool_calls
                })

                self._save_checkpoint(
                    task_id,
                    task,
                    messages,
                    step,
                    "running"
                )

                try:
                    self._execute_pending_tools(
                        tool_calls,
                        messages,
                        task_id,
                        task,
                        step
                    )

                except Exception:
                    self._save_checkpoint(
                        task_id,
                        task,
                        messages,
                        step,
                        "failed"
                    )
                    raise

        self._save_checkpoint(
            task_id,
            task,
            messages,
            self.max_steps,
            "failed"
        )

        return "Maximum agent steps reached."