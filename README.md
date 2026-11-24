# MeTTa Agent

A persistent AI agent built with **MeTTa and Python**. MeTTa controls the agent's task loop and decisions, while Python provides supporting services such as LLM communication, tool execution, memory retrieval, context handling, and SQLite persistence.

## 1. Overview

The agent accepts a task, sends relevant context to a language model, interprets the response, and decides what to do next. It can call tools, store useful information in memory, continue working through multiple steps, and save checkpoints so interrupted tasks can be resumed.

The project combines MeTTa's symbolic control flow with Python's integration capabilities.

## 2. Architecture

| Component                           | Responsibility                                                     |
| ----------------------------------- | ------------------------------------------------------------------ |
| MeTTa (`agent/agent.metta`)         | Main agent loop, task flow, response classification, and decisions |
| Python bridge (`agent_bridge.py`)   | Connects MeTTa to Python services                                  |
| LLM providers                       | Send prompts to language models and return text responses          |
| Tool system (`agent/tools/`)        | Parse, validate, and execute tool requests                         |
| Memory (`agent/memory.py`)          | Store and retrieve persistent memories                             |
| Checkpoints (`agent/checkpoint.py`) | Save task progress and support recovery                            |
| SQLite                              | Persist memories and task checkpoints                              |

MeTTa controls the workflow. Python supports it with external communication, data handling, persistence, and other operations.

## 3. Features

- **Multiple LLM providers:** Gemini, OpenAI, OpenRouter, Ollama, mock, and custom OpenAI-compatible endpoints.
- **Multi-step task execution:** The agent can reason through a task, use tools, and continue until it reaches a final response or stopping condition.
- **Tool execution:** read file, write file, **run shell command**, **send message to the user**, **ask the user a question**, plus calculator, date and time, file listing, waiting and web fetching. Every tool returns text; errors, missing files and timeouts come back as messages the agent can read and never crash it. Unknown tool names are returned to the model as errors.
- **Safe shell use:** shell commands have a timeout, and destructive ones (rm, mv, overwriting with `>`, sudo, ...) need the user's approval first.
- **Long tasks:** the goal is repeated in every prompt, long tool output is cut, old history is compacted, and a turn limit stops runaway tasks. When the agent is stuck (turn limit, or the same failing step repeated) it tells the user why.
- **Asks instead of guessing:** when information is missing the agent uses `ask_user`.
- **Task logs:** every task is written to `logs/task_<id>.log` with numbered turns.
- **Persistent memory:** Stores information in SQLite and restores it into the MeTTa AtomSpace.
- **Checkpoint recovery:** Saves task progress and can resume an interrupted task.
- **Context compaction:** Summarizes older conversation content when the transcript becomes too large.
- **Provider reliability:** Supports retries, retry delays, fallback models, and minimum request spacing.
- **Trace mode:** Prints internal progress information for debugging.

## 4. Built-in Tools

| Tool               | Purpose                                                     |
| ------------------ | ----------------------------------------------------------- |
| `calculator`       | Evaluates supported arithmetic and mathematical expressions |
| `current_datetime` | Returns the current local date and time                     |
| `read_file`        | Reads a file from the agent workspace                       |
| `write_file`       | Creates or overwrites a file in the agent workspace         |
| `list_files`       | Lists files in the agent workspace                          |
| `wait`             | Waits for a specified duration                              |
| `web_fetch`        | Fetches and extracts basic text from an HTTP or HTTPS page  |
| `run_shell`        | Runs a shell command in the workspace and returns its output (timeout; destructive commands need approval) |
| `send_message`     | Shows a progress note or result to the user without ending the task |
| `ask_user`         | Asks the user a question and returns the typed answer      |

File tools are restricted to the agent workspace. Shell commands start in the workspace folder. The web-fetch tool is a basic HTTP text extractor, not a full browser.

### Shell safety

- **Timeout:** every command is stopped after `AGENT_SHELL_TIMEOUT` seconds (default 30). The agent receives a "timed out" message and can try something smaller.
- **Approval:** before a command runs it is checked against a list of risky patterns (`rm`, `rmdir`, `mv`, `dd`, `sudo`, `chmod`, `kill`, `git reset --hard`, `sed -i`, overwriting a file with `>`, piping into a shell, and similar). A match pauses the agent and asks you `Allow? [y/N]`. Anything other than `y` or `yes` returns a `DENIED` message to the agent, which is told not to retry that command.
- **No user available** (for example stdin is closed): approval is treated as denied and `ask_user` returns an error message instead of crashing.
- **Limit:** this is a cautious pattern check, not a security sandbox. It can ask about harmless commands (a false alarm only costs one question) and it cannot catch every possible way to damage files.

### Long tasks, bounded prompts and getting stuck

- The task text is placed at the top of every prompt, so the goal is never lost.
- A single tool result is cut to about 4,000 characters (head and tail are kept, with a `truncated` marker).
- When the transcript exceeds `AGENT_MAX_CONTEXT_CHARS`, older entries are summarised and only recent history is kept.
- The agent is told to keep a `plan.md` checklist in the workspace for long tasks.
- `AGENT_MAX_STEPS` (default 40) limits the number of model turns.
- The agent also stops early if it repeats the same action with the same result three times, or gets five errors in a row.
- In both cases the user receives a report: the reason, the turns used, the last problem and the last actions.

## 5. Persistent Memory

The agent uses five memory types:

| Memory Type  | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `context`    | General information useful for the current context |
| `skill`      | Reusable skills or ways of performing tasks        |
| `episodic`   | Records of past tasks and experiences              |
| `semantic`   | Facts and general knowledge                        |
| `procedural` | Instructions and procedures                        |

Memories are stored in SQLite and restored into a dedicated MeTTa AtomSpace memory space when the agent starts.

Memory retrieval currently uses MeTTa matching together with Python keyword matching and ranking. It does **not** use embedding models or vector search.

## 6. Checkpointing and Recovery

The agent saves task information, conversation messages, the current step, and task status in SQLite.

When the application starts, it checks for an unfinished task and can resume it.

Interactive commands:

| Command          | Purpose                                                  |
| ---------------- | -------------------------------------------------------- |
| `/resume`        | Resume the most recently updated unfinished task         |
| `/abandon`       | Mark the most recently updated unfinished task as failed |
| `exit` or `quit` | Leave the interactive session                            |

The database can retain checkpoints for multiple tasks, but the recovery commands select one unfinished task at a time: the most recently updated task with a `running` status.

## 7. Requirements

- Python 3.10 or a compatible Python environment.
- The PeTTa Python runtime must be installed and importable in the active Python environment.
- The `openai` Python package is needed for the OpenAI-compatible provider.
- `pytest` is needed to run the automated tests.
- An API key is needed for hosted providers such as Gemini, OpenAI, and OpenRouter. Ollama can be used locally.

Run project commands from the repository root, using the Python environment in which `petta` is available.

Check the environment:

```bash
python --version
python -c "from petta import PeTTa; print('PeTTa is available')"
python -c "import openai; print('OpenAI SDK is available')"
```

If required packages are missing, install them into that same Python environment.

## 8. Configure an LLM Provider

The provider is selected with `LLM_PROVIDER`. Gemini is the default when no provider is specified.

### Gemini

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your-api-key"
export LLM_MODEL=gemini-3.5-flash
```

### OpenAI

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
export LLM_MODEL=gpt-4o-mini
```

### OpenRouter

```bash
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY="your-api-key"
export LLM_MODEL=openai/gpt-4o-mini
```

### Ollama

Start Ollama and make sure the model you select is available locally.

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1
```

### Mock Provider

The mock provider is useful for local tests and demonstrations that do not require a hosted LLM.

```bash
export LLM_PROVIDER=mock
```

### Custom OpenAI-Compatible Endpoint

```bash
export LLM_PROVIDER=custom
export LLM_BASE_URL="http://localhost:8000/v1"
export LLM_API_KEY="your-api-key-if-required"
export LLM_MODEL="your-model-name"
```

Optional provider configuration:

| Variable              | Purpose                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `LLM_BASE_URL`        | Overrides the provider's base URL                                              |
| `LLM_API_KEY`         | Supplies an API key directly                                                   |
| `LLM_MODEL`           | Selects the model                                                              |
| `LLM_FALLBACK_MODELS` | Comma-separated fallback models; set to `none` to disable configured fallbacks |
| `LLM_MIN_INTERVAL`    | Minimum spacing between requests, in seconds                                   |

**Security:** Never commit API keys or other secrets to the repository.

## 9. Run the Agent

Run commands from the project root after activating the Python environment that has PeTTa installed.

### Interactive Mode

```bash
python agent_main.py
```

Enter a task when prompted. Type `exit` or `quit` to leave.

### Run One Task

```bash
python agent_main.py --task "What is 12*7? Use the calculator."
```

### Load a MeTTa Smoke-Test File

```bash
LLM_PROVIDER=mock python agent_main.py --file smoke/01_pycall.metta
```

Other smoke tests:

```bash
LLM_PROVIDER=mock python agent_main.py --file smoke/02_memory.metta
LLM_PROVIDER=mock python agent_main.py --file smoke/03_loop_mock.metta
```

## 10. Configuration and Local Data

The following environment variables configure the agent:

| Variable                  | Default                 | Purpose                                                |
| ------------------------- | ----------------------- | ------------------------------------------------------ |
| `AGENT_TRACE`             | `0`                     | Enables trace output when set to `1`, `true`, or `yes` |
| `AGENT_CHECKPOINT_DB`     | `agent_checkpoints.db`  | Checkpoint database path                               |
| `AGENT_MEMORY_DB`         | `agent_memory_store.db` | Memory database path                                   |
| `AGENT_MAX_STEPS`         | `40`                    | Turn limit for one task                                |
| `AGENT_SHELL_TIMEOUT`     | `30`                    | Seconds before a shell command is stopped              |
| `AGENT_LOG_DIR`           | `logs`                  | Folder for per-task logs (set empty to disable)        |
| `AGENT_MAX_CONTEXT_CHARS` | `12000`                 | Transcript size threshold for context compaction       |

The agent workspace defaults to `agent_workspace/`.

Example:

```bash
export AGENT_TRACE=1
export AGENT_MAX_STEPS=30
```

SQLite databases, generated workspace files, and logs are excluded from Git by default. The three completed task logs required for this submission are added explicitly; other runtime and debug logs remain untracked.

## 10b. Task logs

Each task creates `logs/task_<id>.log` with the task, the start time, every model action (numbered `--- turn N ---`), every tool result, and a final line such as `===== COMPLETED after 23 turns =====`. Example of producing the three required logs in one session:

```bash
export AGENT_TRACE=1 LLM_MIN_INTERVAL=12      # spacing for free-tier rate limits
python agent_main.py
agent> <task 1: short>
agent> <task 2: uses the shell>
agent> <task 3: long, 20+ turns>
```

Suggested tasks:

1. *Short:* "Create hello.txt containing a greeting, then read it back and tell me what it says."
2. *Shell:* "Use the shell to count the lines and words in every .txt file in the workspace and write the results to report.txt."
3. *Long (20+ turns):* "Create ten text files named solar_01_mercury.txt through solar_10_ceres.txt, each containing a distinct fact about a Solar System object. Read each file back, then write and read a summary listing every filename and its first line."

Also try (not required as logs): an ambiguous request ("summarise the file") to see it ask you; a read of a missing file to see it recover; `rm` of a file to see the approval prompt.

## 11. Testing

Run the automated test suite from the repository root:

```bash
python -m pytest agent/ -v
```

`agent/test_requirements.py` covers the assignment requirements without needing PeTTa or a real LLM: the five tools, error messages instead of crashes, unknown tools, shell timeout and approval, asking the user, a 22-turn task with a bounded prompt, error recovery, stuck detection, the turn limit, several tasks in a row, and the log file. It replays the same bridge calls that `agent/agent.metta` makes; the real MeTTa loop is exercised by the smoke tests and by running the agent.

Post-merge verification on 24 November 2026: **91 tests passed**.

Additional project utilities:

```bash
python scale_test.py 5000
python benchmark.py
python verify_state.py
```

- `scale_test.py` exercises memory storage, restoration, and retrieval at a selected scale.
- `benchmark.py` runs a set of repeatable agent tasks.
- `verify_state.py` checks the state produced by the integration test workflow.
- `fake_openai_server.py` provides a local fake OpenAI-compatible server for provider integration testing.

These utilities may create local SQLite databases or benchmark output files.

## 12. Known Limitations

- Memory retrieval is keyword-based, not embedding-based semantic search.
- The destructive-command check is pattern based (see Shell safety); it is not a sandbox.
- The agent performs one action per model turn, so tasks with many steps need many model calls (and free-tier rate limits make them slow).
- "Stuck" detection is a heuristic: it catches exact repetition and runs of errors, not every kind of unproductive loop.
- In `--task` mode with piped input there may be no user to answer `ask_user` or approvals; the agent then receives an error message and must state its assumption or stop.
- The web-fetch tool extracts basic page text and does not behave like a full browser.
- The provider interface exchanges plain text. Tool requests are represented in the agent's response format rather than using native LLM function calling.
- Recovery selects the most recently updated unfinished task; it does not automatically resume every unfinished task.
- A tool may execute again if the process stops after the tool runs but before its result is saved. Therefore, tool execution is not guaranteed to happen exactly once.
- Context compaction summarizes older transcript content, so some detail may be lost.
- Hosted LLM providers require network access and valid API credentials.

## 13. Project Structure

```text
.
├── agent/
│   ├── agent.metta
│   ├── checkpoint.py
│   ├── loop.py
│   ├── memory.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── mock.py
│   │   └── openai_compat.py
│   ├── tools/
│   │   ├── builtin.py
│   │   ├── parser.py
│   │   ├── registry.py
│   │   ├── shell_safety.py
│   │   └── userio.py
│   ├── test_requirements.py
│   └── test_*.py
├── logs/                (three required completed task logs committed; other logs ignored)
├── NOTES.md
├── smoke/
│   ├── 01_pycall.metta
│   ├── 02_memory.metta
│   ├── 03_loop_mock.metta
│   └── 04_debug.metta
├── agent_bridge.py
├── agent_main.py
├── benchmark.py
├── fake_openai_server.py
├── scale_test.py
├── verify_state.py
└── README.md
```

## 14. Design Summary

This project uses MeTTa for the agent's control flow and Python for practical integration. The combination supports multi-step task execution, tool use, persistent memory, and checkpoint recovery.

The current implementation provides a working foundation that can be extended with more advanced memory retrieval, stronger browser capabilities, and more robust recovery guarantees.
