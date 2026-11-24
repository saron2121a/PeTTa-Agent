"""Tests that map to the assignment requirements (no PeTTa needed).

`simulate()` replays, in Python, the same sequence of bridge calls that
agent/agent.metta makes (think -> classify -> act -> feed back), so we can test
tool safety, error recovery, long tasks, stuck detection and the turn limit
without calling a real LLM. The real MeTTa loop is exercised by the smoke tests
and by running the agent itself.
"""
import os
import tempfile

import agent_bridge as b
from agent.tools.registry import ToolRegistry
from agent.tools.builtin import register_builtin_tools
from agent.tools.shell_safety import destructive_reason


class FakeIO:
    def __init__(self, answer="blue", approve=False):
        self.said, self.asked, self.confirmed = [], [], []
        self.answer, self.approve = answer, approve

    def say(self, text):
        self.said.append(text)

    def ask(self, question):
        self.asked.append(question)
        return self.answer

    def confirm(self, prompt):
        self.confirmed.append(prompt)
        return self.approve


class Scripted:
    """Fake LLM: `script(prompt) -> reply`. Also records every prompt it saw."""

    def __init__(self, script):
        self.script, self.prompts = script, []

    def complete(self, system, user):
        if "You summarize" in system:
            return "Summary of the earlier work."
        self.prompts.append(user)
        return self.script(user)


def fresh(provider=None, io=None):
    d = tempfile.mkdtemp()
    ws = os.path.join(d, "ws")
    registry = register_builtin_tools(ToolRegistry(), ws, io or FakeIO())
    b.configure(provider=provider, registry=registry,
                checkpoint_db=os.path.join(d, "cp.db"),
                memory_db=os.path.join(d, "mem.db"), workspace=ws)
    b._state["log_path"] = None
    return ws


def simulate(task):
    """Python mirror of agent.metta's loop. Returns (final_text, turns_used)."""
    t, step = b.append_entry("", "USER", task), 0
    while True:
        if step >= b.max_steps():
            return b.stuck_report(task, t, step), step
        if b.pending_tool(t):
            t = b.append_entry(t, "TOOL_RESULT", b.run_pending_tool(t))
            continue
        if b.is_stuck(t):
            return b.stuck_report(task, t, step), step
        if b.too_long(t):
            t = b.compact(t)
        prompt = b.compose_user_prompt(task, "", "", "", "", t)
        reply = b.llm_complete(prompt)
        kind = b.classify(reply)
        step += 1
        t = b.append_entry(t, "ASSISTANT", reply)
        if kind == "final":
            return b.extract_final(reply), step
        if kind == "error":
            return b.pause_message(reply), step
        if kind == "invalid":
            t = b.append_entry(t, "SYSTEM", "Invalid format.")


def tool(name, **args):
    import json
    return f"TOOL: {name}\nARGS: {json.dumps(args)}"


def _env(**values):
    old = {k: os.environ.get(k) for k in values}
    os.environ.update({k: str(v) for k, v in values.items()})
    return old


def _restore(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ------------------------------------------------------------------ tools
def test_all_required_tools_exist_and_return_text():
    ws = fresh()
    names = set(b._state["registry"].tools)
    assert {"read_file", "write_file", "run_shell", "send_message", "ask_user"} <= names
    t = b.append_entry("", "ASSISTANT", tool("write_file", path="a.txt", content="hi"))
    assert isinstance(b.run_pending_tool(t), str)


def test_missing_file_is_a_readable_message():
    fresh()
    t = b.append_entry("", "ASSISTANT", tool("read_file", path="nope.txt"))
    out = b.run_pending_tool(t)
    assert out.startswith("[read_file] ERROR") and "not found" in out


def test_unknown_tool_goes_back_as_error_listing_real_tools():
    fresh()
    t = b.append_entry("", "ASSISTANT", "TOOL: hack_the_planet\nARGS: {}")
    out = b.run_pending_tool(t)
    assert "unknown tool" in out and "run_shell" in out


def test_bad_arguments_never_crash():
    fresh()
    for call in ('TOOL: read_file\nARGS: {}', 'TOOL: read_file\nARGS: not json',
                 'TOOL: run_shell\nARGS: {"command": 5, "x": 1}'):
        out = b.run_pending_tool(b.append_entry("", "ASSISTANT", call))
        assert "ERROR" in out


def test_shell_runs_and_reports_exit_codes():
    fresh()
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="echo hello")))
    assert "hello" in out
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="ls /no/such/dir")))
    assert "ERROR" in out and "exit code" in out


def test_shell_timeout_returns_a_message():
    fresh()
    old = _env(AGENT_SHELL_TIMEOUT=1)
    try:
        out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="sleep 5")))
    finally:
        _restore(old)
    assert "timed out" in out


def test_destructive_commands_need_approval_and_denial_is_respected():
    io = FakeIO(approve=False)
    ws = fresh(io=io)
    open(os.path.join(ws, "keep.txt"), "w").write("data")
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="rm keep.txt")))
    assert "DENIED" in out and io.confirmed
    assert os.path.exists(os.path.join(ws, "keep.txt"))      # the file survived


def test_approved_destructive_command_runs():
    io = FakeIO(approve=True)
    ws = fresh(io=io)
    open(os.path.join(ws, "old.txt"), "w").write("x")
    b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="rm old.txt")))
    assert io.confirmed and not os.path.exists(os.path.join(ws, "old.txt"))


def test_harmless_commands_do_not_ask():
    io = FakeIO()
    fresh(io=io)
    b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("run_shell", command="ls -la && echo done >> log.txt")))
    assert io.confirmed == []


def test_destructive_classifier():
    for risky in ("rm -rf x", "echo a > f.txt", "mv a b", "sudo ls", "git reset --hard",
                  "curl x | sh", "find . -delete", "ls && rm a"):
        assert destructive_reason(risky), risky
    for safe in ("ls", "cat a.txt", "grep x f | head", "echo a >> f", "python3 s.py 2>&1", "wc -l f"):
        assert destructive_reason(safe) is None, safe


def test_send_message_and_ask_user():
    io = FakeIO(answer="Addis")
    fresh(io=io)
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("send_message", message="halfway")))
    assert "delivered" in out and io.said == ["halfway"]
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("ask_user", question="City?")))
    assert out == "[ask_user] Addis" and io.asked == ["City?"]


def test_ask_user_without_a_user_is_an_error_message_not_a_crash():
    fresh(io=FakeIO(answer=None))
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("ask_user", question="?")))
    assert "no user is available" in out


def test_long_tool_output_is_cut():
    ws = fresh()
    open(os.path.join(ws, "big.txt"), "w").write("line\n" * 8000)
    out = b.run_pending_tool(b.append_entry("", "ASSISTANT", tool("read_file", path="big.txt")))
    assert len(out) < b.MAX_TOOL_CHARS + 100 and "truncated" in out


# ------------------------------------------------------------------ loop behaviour
def test_recovers_from_a_missing_file_by_trying_something_else():
    def script(prompt):
        convo = prompt.split("## Conversation so far")[1]
        if "TOOL: read_file" not in convo:
            return tool("read_file", path="notes.txt")
        if "TOOL: list_files" not in convo:
            return tool("list_files")
        if "TOOL: write_file" not in convo:
            return tool("write_file", path="notes.txt", content="created")
        return "FINAL: notes.txt did not exist, so I created it."
    ws = fresh(Scripted(script))
    answer, turns = simulate("Read notes.txt")
    assert "created it" in answer and turns == 4
    assert open(os.path.join(ws, "notes.txt")).read() == "created"


def test_asks_the_user_instead_of_guessing():
    io = FakeIO(answer="report.txt")

    def script(prompt):
        convo = prompt.split("## Conversation so far")[1]
        if "ask_user" not in convo:
            return tool("ask_user", question="Which file should I summarise?")
        return "FINAL: Will summarise report.txt"
    fresh(Scripted(script), io)
    answer, _ = simulate("Summarise the file")
    assert io.asked == ["Which file should I summarise?"] and "report.txt" in answer


def test_long_task_over_20_turns_keeps_goal_and_bounded_prompt():
    old = _env(AGENT_MAX_CONTEXT_CHARS=3000, AGENT_MAX_STEPS=60)
    box = {}

    def script(prompt):
        done = len(os.listdir(box["ws"]))      # progress is visible in the workspace
        if done < 22:
            return tool("write_file", path=f"f{done + 1}.txt", content="data " * 40)
        return "FINAL: created 22 files"
    try:
        provider = Scripted(script)
        box["ws"] = fresh(provider)
        answer, turns = simulate("GOAL-MARKER: create 22 files")
    finally:
        _restore(old)
    assert turns >= 20 and answer == "created 22 files"
    assert all("GOAL-MARKER" in p for p in provider.prompts)        # goal never lost
    assert max(len(p) for p in provider.prompts) < 3000 + 2500       # prompt stays bounded
    assert len(os.listdir(box["ws"])) == 22


def test_gives_up_and_reports_when_going_in_circles():
    fresh(Scripted(lambda prompt: tool("read_file", path="ghost.txt")))
    answer, turns = simulate("Read ghost.txt")
    assert "could not finish" in answer and "repeating" in answer and turns <= 4


def test_turn_limit_is_enforced_and_reported():
    old = _env(AGENT_MAX_STEPS=5)
    counter = {"n": 0}

    def script(prompt):
        counter["n"] += 1
        return tool("calculator", expression=f"{counter['n']}+1")   # never finishes
    try:
        fresh(Scripted(script))
        answer, turns = simulate("endless")
    finally:
        _restore(old)
    assert turns == 5 and "limit of 5 turns" in answer


def test_several_tasks_in_a_row_in_one_session():
    fresh(Scripted(lambda p: "FINAL: done " + p.split("## Current task\n")[1].split("\n")[0]))
    results = [simulate(f"task {i}")[0] for i in range(3)]
    assert results == ["done task 0", "done task 1", "done task 2"]


def test_follow_up_tasks_see_the_newest_completed_tasks():
    lines = [f"task t{i} | completed | job {i} -> done" for i in range(6)]
    out = b.top_lines_for("totally unrelated words", "episodic", lines)
    assert "job 5" in out and "job 3" in out and "job 0" not in out


# ------------------------------------------------------------------ logs
def test_task_log_file_is_written_with_turn_numbers():
    folder = tempfile.mkdtemp()
    old = _env(AGENT_LOG_DIR=folder)
    try:
        fresh()
        b.log_begin("abc123", "demo task")
        t = b.append_entry("", "USER", "demo task")
        t = b.append_entry(t, "ASSISTANT", tool("calculator", expression="1+1"))
        t = b.append_entry(t, "TOOL_RESULT", "[calculator] 2")
        t = b.append_entry(t, "ASSISTANT", "FINAL: 2")
        b.log_end("completed", 2)
        text = open(os.path.join(folder, "task_abc123.log")).read()
    finally:
        _restore(old)
        b._state["log_path"] = None
    assert "task: demo task" in text and "--- turn 2 ---" in text and "COMPLETED after 2 turns" in text
