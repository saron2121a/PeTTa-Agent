# NOTES: design of the MeTTa (PeTTa) agent

**What it is.** A command-line agent that takes tasks from the user, works on them turn by turn, and keeps running between tasks. The loop, the state and every decision live in MeTTa (`agent/agent.metta`). Python (`agent_bridge.py`, called with `py-call`) only does what MeTTa cannot: the HTTP call to the LLM, running tools, text parsing, SQLite.

**Each turn.** Build the prompt (task + relevant memories + recent transcript) -> ask the model -> classify the reply (`FINAL`, `TOOL`+`ARGS`, `REMEMBER`) -> run the tool -> append the result -> repeat. Plain text is exchanged, so any OpenAI-compatible provider works (Gemini by default).

**Tools.** `read_file`, `write_file`, `run_shell`, `send_message`, `ask_user` (required), plus `list_files`, `calculator`, `current_datetime`, `wait`, `web_fetch`. Every tool returns text. A tool that raises, a missing file, bad arguments, a timeout and an unknown tool name all become an `ERROR: ...` message in the transcript; the unknown-tool message lists the real tools. Nothing outside the registry can run.

**Safety.** Shell commands use the workspace as their working directory and have a timeout (30 s); they are not restricted to that directory. Commands matching a destructive pattern (rm, mv, `>` overwrite, sudo, ...) pause for a y/N approval; a denial goes back to the model as `DENIED` and it is told not to retry. This is a cautious filter, not a sandbox.

**Staying on track in long tasks.** The task text is in every prompt, tool output is cut to ~4000 chars, older history is summarised when the transcript passes a size limit (recent entries are kept), and the agent is told to keep a `plan.md` checklist for long jobs. Progress is checkpointed in SQLite after every step, so an interrupted task resumes.

**Failure handling.** The prompt tells the model to read errors and change approach. A turn limit (default 40) and two stuck checks (same action and result three times; five errors in a row) end the task with a report to the user: reason, turns used, last problem, last actions.

**Asking instead of guessing.** The prompt requires `ask_user` when information is missing. With no user available (stdin closed) the tool returns an error and the model must state its assumption or stop.

**Memory.** Five memory types stored in SQLite and restored into an AtomSpace (`&agent_memories`); recall is keyword matching, not embeddings. The newest completed tasks are always offered to the model so follow-up tasks in one session have context.

**Logs.** `logs/task_<id>.log` records every turn, tool result and the final status.

**Trade-offs and limits.**
- One action per model turn: simple and robust to parse, but long tasks need many calls.
- The model decides when it is done; there is no separate verifier.
- Keyword memory can miss paraphrases.
- Tests replay the bridge calls in Python because the MeTTa loop needs PeTTa; the MeTTa loop itself was checked with the smoke tests and real runs.

**Next steps (not done).** Sub-agents for independent sub-tasks, embedding-based memory, a real sandbox for shell commands.

**Live demonstration.** See [Live CLI Demonstration](docs/live_demo.md) for a record of actual interactive CLI tests, including sequential tool use, asking the user for missing information, continuing after the user's response, and handling multiple tasks in one session.
