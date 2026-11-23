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
- **Tool execution:** Supports calculations, date and time lookup, workspace file operations, waiting, and web fetching.
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

File operations are restricted to the agent workspace. The web-fetch tool is a basic HTTP text extractor, not a full browser.

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
python agent_main.py --file smoke/01_pycall.metta
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
| `AGENT_MAX_STEPS`         | `25`                    | Maximum steps allowed for a task                       |
| `AGENT_MAX_CONTEXT_CHARS` | `12000`                 | Transcript size threshold for context compaction       |

The agent workspace defaults to `agent_workspace/`.

Example:

```bash
export AGENT_TRACE=1
export AGENT_MAX_STEPS=30
```

SQLite databases and generated workspace files are local runtime data and are excluded from Git.

## 11. Testing

Run the automated test suite from the repository root:

```bash
python -m pytest agent/ -v
```

The latest recorded test run completed with **69 tests passed**.

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
│   │   └── registry.py
│   └── test_*.py
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
