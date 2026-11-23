"""Tests for the Python helpers used by the MeTTa loop (no PeTTa needed)."""
import tempfile
import os

import agent_bridge as b
from agent.providers.openai_compat import MockTextProvider, OpenAICompatibleProvider


def setup_module(_module=None):
    d = tempfile.mkdtemp()
    b.configure(provider=MockTextProvider(),
                checkpoint_db=os.path.join(d, "cp.db"),
                memory_db=os.path.join(d, "mem.db"),
                workspace=os.path.join(d, "ws"))


def test_classify_and_extract():
    assert b.classify("FINAL: 42") == "final"
    assert b.extract_final("Thinking...\nFINAL: 42") == "42"
    assert b.classify("just a plain answer") == "final"
    assert b.classify('TOOL: calculator\nARGS: {"expression": "1+1"}') == "tool"
    assert b.classify("TOOL:") == "invalid"
    assert b.classify("   ") == "invalid"
    assert b.classify("REMEMBER: semantic | a | b | c") == "remember"
    assert b.classify("<<LLM_ERROR>> boom") == "error"


def test_parse_tool_call_variants():
    assert b.parse_tool_call('TOOL: calculator\nARGS: {"expression": "2+2"}') == (
        "calculator", {"expression": "2+2"})
    assert b.parse_tool_call("TOOL: `current_datetime`")[0] == "current_datetime"
    assert b.parse_tool_call('TOOL: calculator\nARGS: ```json\n{"expression": "1"}\n```')[1] == {
        "expression": "1"}
    try:
        b.parse_tool_call("TOOL: x\nARGS: not json")
        assert False
    except ValueError:
        pass


def test_remember_fields_and_validation():
    r = "REMEMBER: Semantic | ፕሮጀክት | uses | PeTTa"
    assert [b.rem_field(r, i) for i in range(4)] == ["semantic", "ፕሮጀክት", "uses", "PeTTa"]
    assert b.rem_field("REMEMBER: nonsense | a | b | c", 0) == ""
    assert b.rem_field("REMEMBER: semantic | only | two", 1) == ""


def test_transcript_and_pending_tool():
    t = b.append_entry("", "USER", "task")
    assert not b.pending_tool(t)
    t = b.append_entry(t, "ASSISTANT", 'TOOL: calculator\nARGS: {"expression": "5+3"}')
    assert b.pending_tool(t)
    assert b.run_pending_tool(t) == "[calculator] 8"
    t = b.append_entry(t, "TOOL_RESULT", "[calculator] 8")
    assert not b.pending_tool(t)


def test_tool_errors_become_text_not_crashes():
    t = b.append_entry("", "ASSISTANT", "TOOL: nope\nARGS: {}")
    assert "ERROR" in b.run_pending_tool(t)
    t = b.append_entry("", "ASSISTANT", 'TOOL: read_file\nARGS: {"path": "../../etc/passwd"}')
    assert "outside the workspace" in b.run_pending_tool(t)


def test_top_lines_ranking_and_unicode():
    lines = ["ፕሮጀክት | uses | PeTTa", "weather | is | sunny"]
    assert b.top_lines("tell me about ፕሮጀክት", lines) == "- ፕሮጀክት | uses | PeTTa"
    assert b.top_lines("nothing relevant here", lines) == ""


def _with_context_limit(limit, fn):
    old = os.environ.get("AGENT_MAX_CONTEXT_CHARS")
    os.environ["AGENT_MAX_CONTEXT_CHARS"] = str(limit)
    try:
        return fn()
    finally:
        if old is None:
            os.environ.pop("AGENT_MAX_CONTEXT_CHARS")
        else:
            os.environ["AGENT_MAX_CONTEXT_CHARS"] = old


def test_compaction_keeps_recent_and_pending_state():
    t = ""
    for i in range(20):
        t = b.append_entry(t, "USER" if i % 2 == 0 else "ASSISTANT", f"message {i} " + "x" * 50)
    t = b.append_entry(t, "ASSISTANT", 'TOOL: calculator\nARGS: {"expression": "1+1"}')
    new = _with_context_limit(600, lambda: b.compact(t))
    assert new.startswith("<<SUMMARY>>") and b.has_summary(new)
    assert b.pending_tool(new)
    assert len(new) < len(t)


def test_compaction_with_few_but_large_entries_still_summarises():
    """The Test 5 bug: few entries, many characters used to give '(no summary)'."""
    t = ""
    for i in range(4):
        t = b.append_entry(t, "USER" if i % 2 == 0 else "ASSISTANT", f"entry {i} " + "y" * 2000)
    t = b.append_entry(t, "ASSISTANT", 'TOOL: write_file\nARGS: {"path": "a.txt", "content": "' + "z" * 3000 + '"}')
    assert b.too_long(t) is False  # default limit is large...
    new = _with_context_limit(1500, lambda: b.compact(t))
    assert b.has_summary(new)
    assert b.summary_of(new) != ""
    assert b.pending_tool(new)
    assert "z" * 3000 in new          # the last entry is never clipped
    assert len(new) < len(t)


def test_compaction_without_old_entries_makes_no_summary():
    t = b.append_entry("", "USER", "short task")
    new = _with_context_limit(1500, lambda: b.compact(t))
    assert not b.has_summary(new) and b.summary_of(new) == ""


def test_recall_stemming_and_past_intent():
    lines = ["task a1 | completed | calculated 12*7 -> 84",
             "task b2 | completed | wrote squares.txt -> 385",
             "project | uses | PeTTa"]
    out = b.top_lines_for("What tasks did we complete previously?", "episodic", lines)
    assert "calculated" in out and "squares" in out   # recent episodes included
    out = b.top_lines_for("which framework do projects use", "semantic", lines)
    assert "PeTTa" in out
    assert b.top_lines_for("unrelated words", "semantic", lines) == ""


def test_checkpoint_roundtrip_and_resume_lookup():
    b.save_checkpoint("t1", "do it", "<<USER>>\ndo it\n\n", 3, "running")
    assert b.find_unfinished() == "t1"
    assert b.cp_task("t1") == "do it"
    assert b.cp_step("t1") == 3
    b.save_checkpoint("t1", "do it", "<<USER>>\ndo it\n\n", 4, "completed")
    assert b.find_unfinished() == ""


def test_memory_persistence_is_idempotent():
    b.persist_memory("semantic", "a", "b", "c")
    b.persist_memory("semantic", "a", "b", "c")


class _Err(Exception):
    def __init__(self, status):
        self.status_code = status


class _FlakyClient:
    def __init__(self, failures):
        self.failures = failures
        self.calls = 0
        self.chat = self
        self.completions = self

    def create(self, model, messages):
        self.calls += 1
        if self.calls <= self.failures:
            raise _Err(429)

        class R:
            choices = [type("C", (), {"message": type("M", (), {"content": "hi"})})]
        return R


def test_provider_retries_rate_limits():
    client = _FlakyClient(2)
    p = OpenAICompatibleProvider(model="m", client=client, sleep=lambda s: None)
    assert p.complete("sys", "user") == "hi"
    assert client.calls == 3


def test_provider_gives_up_on_non_retryable():
    class Bad(_FlakyClient):
        def create(self, model, messages):
            raise _Err(400)
    p = OpenAICompatibleProvider(model="m", client=Bad(0), sleep=lambda s: None)
    try:
        p.complete("s", "u")
        assert False
    except _Err:
        pass


def test_same_and_format_memory_if():
    assert b.same("final", "final") and not b.same("final", "tool")
    assert b.format_memory_if("skill", "skill", "a", "b", "c") == "a | b | c"
    assert b.format_memory_if("skill", "semantic", "a", "b", "c") == ""
    assert b.top_lines("alpha", ["", "alpha | beta | gamma"]) == "- alpha | beta | gamma"


def test_wait_tool_is_registered_and_capped():
    t = b.append_entry("", "ASSISTANT", 'TOOL: wait\nARGS: {"seconds": 0}')
    assert b.run_pending_tool(t) == "[wait] Waited 0 seconds"


def test_abandon_unfinished():
    b.save_checkpoint("zz9", "stuck task", "<<USER>>\nstuck task\n\n", 1, "running")
    assert "Abandoned" in b.abandon_unfinished()
    assert b.find_unfinished() == ""
    assert b.abandon_unfinished() == "No unfinished task."


class _Resp:
    def __init__(self, headers):
        self.headers = headers


class _QuotaErr(Exception):
    def __init__(self, status, body="", headers=None):
        self.status_code = status
        self.body = body
        self.response = _Resp(headers or {})


class _ScriptedClient:
    """Raises the scripted errors in order (per model), then answers."""
    def __init__(self, errors_by_model):
        self.errors = {k: list(v) for k, v in errors_by_model.items()}
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, model, messages):
        self.calls.append(model)
        queue = self.errors.get(model, [])
        if queue:
            raise queue.pop(0)

        class R:
            choices = [type("C", (), {"message": type("M", (), {"content": "ok:" + model})})]
        return R


def test_provider_honours_retry_after_header():
    waits = []
    client = _ScriptedClient({"m": [_QuotaErr(429, headers={"retry-after": "7"})]})
    p = OpenAICompatibleProvider(model="m", client=client, sleep=waits.append)
    assert p.complete("s", "u") == "ok:m"
    assert waits == [7.0]


def test_provider_parses_google_retry_hint():
    waits = []
    body = "{'message': 'quota. Please retry in 17.5s.'}"
    client = _ScriptedClient({"m": [_QuotaErr(429, body=body)]})
    p = OpenAICompatibleProvider(model="m", client=client, sleep=waits.append)
    p.complete("s", "u")
    assert waits == [17.5]


def test_daily_quota_is_not_retried_and_falls_back():
    body = "GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    client = _ScriptedClient({"m": [_QuotaErr(429, body=body)] * 10})
    p = OpenAICompatibleProvider(model="m", client=client, sleep=lambda s: None,
                                 fallback_models=["backup"])
    assert p.complete("s", "u") == "ok:backup"
    assert client.calls == ["m", "backup"]      # no pointless retries on m


def test_daily_quota_without_fallback_raises_quickly():
    body = "RequestsPerDay"
    client = _ScriptedClient({"m": [_QuotaErr(429, body=body)] * 10})
    p = OpenAICompatibleProvider(model="m", client=client, sleep=lambda s: None)
    try:
        p.complete("s", "u")
        assert False
    except _QuotaErr:
        assert client.calls == ["m"]


def test_unknown_model_404_falls_back():
    client = _ScriptedClient({"old": [_QuotaErr(404, body="no longer available")]})
    p = OpenAICompatibleProvider(model="old", client=client, sleep=lambda s: None,
                                 fallback_models=["new"])
    assert p.complete("s", "u") == "ok:new"


def test_min_interval_throttles_calls():
    now = [0.0]
    waits = []
    client = _ScriptedClient({})
    p = OpenAICompatibleProvider(model="m", client=client, min_interval=5,
                                 clock=lambda: now[0],
                                 sleep=lambda s: (waits.append(s), now.__setitem__(0, now[0] + s)))
    p.complete("s", "u")
    now[0] += 1.0
    p.complete("s", "u")
    assert waits == [4.0]


def test_failed_model_is_skipped_on_later_calls():
    body = "RequestsPerDay"
    client = _ScriptedClient({"m": [_QuotaErr(429, body=body)] * 10})
    p = OpenAICompatibleProvider(model="m", client=client, sleep=lambda s: None,
                                 fallback_models=["backup"])
    p.complete("s", "u")
    p.complete("s", "u")
    assert client.calls == ["m", "backup", "backup"]   # m is not retried while cooling down


def test_past_question_lists_completed_tasks_first():
    lines = [f"task t{i} | completed | job {i} -> done" for i in range(8)]
    lines.insert(3, "task x | summary | something about tasks")
    out = b.top_lines_for("what tasks did we complete previously", "episodic", lines)
    assert out.count("completed") == 8


def test_summary_has_no_marker_artifacts():
    """Test 5 artifact: the summary memory used to start with '< <SUMMARY>>'."""
    seen = {}

    class Echo:
        def complete(self, system, user):
            seen["user"] = user
            return "<<SUMMARY>>\nDid things"

    old_provider = b._state["provider"]
    b._state["provider"] = Echo()
    try:
        t = ""
        for i in range(12):
            t = b.append_entry(t, "USER" if i % 2 == 0 else "ASSISTANT", f"message {i} " + "x" * 80)
        new = _with_context_limit(600, lambda: b.compact(t))
    finally:
        b._state["provider"] = old_provider
    assert "<<" not in seen["user"]          # the summariser never sees our markers
    assert b.summary_of(new) == "Did things"
    assert "< <" not in new
