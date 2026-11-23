"""Python helpers that the MeTTa agent (agent/agent.metta) calls via py-call.

Design rule: MeTTa owns the agent loop, the state and every decision.
This module only does what MeTTa cannot do by itself: HTTP calls to the LLM,
running tools, string parsing, and writing to SQLite.
It must stay importable as a top-level module ("agent_bridge").
"""
import json
import os
import re
import sqlite3
import time
import uuid

from agent.checkpoint import CheckpointManager
from agent.tools.registry import ToolRegistry
from agent.tools.builtin import register_builtin_tools

VALID_MEMORY_TYPES = {"semantic", "episodic", "skill", "procedural", "context"}
MAX_TOOL_CHARS = 4000
KEEP_RECENT = 6

ROLE_RE = re.compile(r"^<<([A-Z_]+)>>\n", re.M)
DIRECTIVE_RE = re.compile(r"^[ \t]*(FINAL|TOOL|REMEMBER)[ \t]*:", re.I | re.M)

_stats = {"llm_calls": 0, "llm_seconds": 0.0, "tool_calls": 0, "tool_seconds": 0.0}
PAST_RE = re.compile(r"\b(previous(ly)?|earlier|before|last time|yesterday|history|"
                     r"past|ago|did we|have we|completed)\b", re.I)
LIMITS = {"semantic": 8, "skill": 4, "procedural": 4, "episodic": 5}

_state = {"provider": None, "registry": None, "checkpoints": None,
          "memory_db": "agent_memory_store.db", "workspace": "agent_workspace"}


def _trace(message):
    """Print what the agent is doing when AGENT_TRACE=1 (for demos and debugging)."""
    if os.getenv("AGENT_TRACE", "").lower() in ("1", "true", "yes"):
        print(f"[trace] {message}", flush=True)


def stats_reset():
    for key in _stats:
        _stats[key] = 0 if key.endswith("calls") else 0.0


# ------------------------------------------------------------------ setup
def configure(provider=None, registry=None, checkpoint_db=None,
              memory_db=None, workspace=None):
    if workspace:
        _state["workspace"] = workspace
    if provider is not None:
        _state["provider"] = provider
    _state["registry"] = registry or register_builtin_tools(
        ToolRegistry(), _state["workspace"])
    _state["checkpoints"] = CheckpointManager(checkpoint_db or "agent_checkpoints.db")
    if memory_db:
        _state["memory_db"] = memory_db
    _ensure_memory_table()
    return True


def init_runtime():
    """Called once by agent_main.py (before the MeTTa loop starts)."""
    from agent.providers.openai_compat import create_provider
    from agent.memory import AgentMemory  # imports PeTTa

    configure(provider=create_provider(),
              checkpoint_db=os.getenv("AGENT_CHECKPOINT_DB", "agent_checkpoints.db"),
              memory_db=os.getenv("AGENT_MEMORY_DB", "agent_memory_store.db"))
    memory = AgentMemory(db_path=_state["memory_db"])
    return memory.restore_to_petta()  # SQLite -> AtomSpace


def max_steps():
    return int(os.getenv("AGENT_MAX_STEPS", "25"))


def _max_context():
    return int(os.getenv("AGENT_MAX_CONTEXT_CHARS", "12000"))


# ------------------------------------------------------- tiny smoke helpers
def ping():
    return "pong"


def is_true():
    return True


def echo_list(items):
    return " | ".join(str(x) for x in items) + f" (len={len(items)})"


def same(a, b):
    """String equality done in Python (MeTTa's == fails between Python-made and typed strings)."""
    return str(a) == str(b)


def show(text):
    print(text, flush=True)
    return True


def new_task_id():
    return uuid.uuid4().hex[:12]


def pause_message(reply):
    return ("Paused: the LLM call failed (" + str(reply).replace("<<LLM_ERROR>>", "").strip()
            + "). The task is saved and resumes automatically on the next start. "
            + pause_hint())


def task_label(task_id):
    return f"task {task_id}"


def short(task, answer):
    return f"{str(task)[:120]} -> {str(answer)[:200]}"


# ------------------------------------------------------------- transcript
def append_entry(transcript, role, content):
    content = re.sub(r"(?m)^<<", "< <", str(content).strip())
    return f"{transcript}<<{role}>>\n{content}\n\n"


def entries(transcript):
    found = list(ROLE_RE.finditer(transcript))
    result = []
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(transcript)
        result.append((m.group(1), transcript[m.end():end].strip()))
    return result


def _render(items):
    out = ""
    for role, content in items:
        out = append_entry(out, role, content)
    return out


# ------------------------------------------------------- reply classification
def classify(reply):
    if str(reply).startswith("<<LLM_ERROR>>"):
        return "error"
    if not reply or not str(reply).strip():
        return "invalid"
    m = DIRECTIVE_RE.search(reply)
    if m is None:
        return "final"  # plain text answer
    kind = m.group(1).upper()
    if kind == "TOOL":
        return "tool" if _tool_name(reply) else "invalid"
    return kind.lower()


def extract_final(reply):
    m = DIRECTIVE_RE.search(reply)
    if m and m.group(1).upper() == "FINAL":
        return reply[m.end():].strip()
    return reply.strip()


def _tool_name(reply):
    m = re.search(r"^[ \t]*TOOL[ \t]*:[ \t]*(.+)$", reply, re.I | re.M)
    if not m:
        return ""
    return m.group(1).strip().strip("`'\"").split("(")[0].strip()


def parse_tool_call(reply):
    name = _tool_name(reply)
    args = {}
    am = re.search(r"^[ \t]*ARGS[ \t]*:", reply, re.I | re.M)
    if am:
        rest = reply[am.end():]
        start = rest.find("{")
        if start == -1:
            if rest.strip():
                raise ValueError("ARGS must be a JSON object like {\"key\": \"value\"}")
        else:
            args, _ = json.JSONDecoder().raw_decode(rest[start:])
            if not isinstance(args, dict):
                raise ValueError("ARGS must be a JSON object")
    return name, args


def rem_field(reply, index):
    """Field `index` (0..3) of 'REMEMBER: type | subject | relation | object'."""
    m = re.search(r"^[ \t]*REMEMBER[ \t]*:[ \t]*(.+)$", reply, re.I | re.M)
    if not m:
        return ""
    parts = [p.strip().strip("\"'") for p in m.group(1).split("|", 3)]
    if len(parts) != 4 or not all(parts) or parts[0].lower() not in VALID_MEMORY_TYPES:
        return ""
    parts[0] = parts[0].lower()
    return parts[index]


# ------------------------------------------------------------------ tools
def pending_tool(transcript):
    items = entries(transcript)
    return bool(items) and items[-1][0] == "ASSISTANT" and classify(items[-1][1]) == "tool"


def _coerce(name, args):
    spec = _state["registry"].tools.get(name)
    if spec is None:
        return args
    fixed = dict(args)
    for key, value in args.items():
        wanted = spec["parameters"].get(key, {}).get("type")
        if wanted is float and type(value) is int:
            fixed[key] = float(value)
        elif wanted is str and isinstance(value, (int, float)) and not isinstance(value, bool):
            fixed[key] = str(value)
    return fixed


def run_pending_tool(transcript):
    name, args, started = "?", {}, time.time()
    try:
        name, args = parse_tool_call(entries(transcript)[-1][1])
        _trace(f"TOOL {name} {json.dumps(args, ensure_ascii=False)[:200]}")
        text = str(_state["registry"].execute(name, _coerce(name, args)))
    except Exception as error:
        text = f"ERROR: {type(error).__name__}: {error}"
    elapsed = time.time() - started
    _stats["tool_calls"] += 1
    _stats["tool_seconds"] += elapsed
    _trace(f"  result ({elapsed:.2f}s): {text[:150]!r}")
    return f"[{name}] {text}"[:MAX_TOOL_CHARS]


def describe_tools():
    lines = []
    for tool in _state["registry"].describe_tools():
        params = ", ".join(f"{n}: {p['type'].__name__}" for n, p in tool["parameters"].items())
        notes = "; ".join(f"{n} = {p['description']}" for n, p in tool["parameters"].items())
        lines.append(f"- {tool['name']}({params}) - {tool['description']}"
                     + (f" [{notes}]" if notes else ""))
    return "\n".join(lines)


# --------------------------------------------------------------- the prompt
def build_system_prompt():
    return f"""You are a general-purpose autonomous agent with long-term memory and tools.
Work on the user's task step by step. Reply with EXACTLY ONE action per message,
in one of these formats (the keyword must start the line):

FINAL: <your final answer to the user>

TOOL: <tool name>
ARGS: <JSON object with the arguments>

REMEMBER: <type> | <subject> | <relation> | <object>
  type is one of: semantic (facts), episodic (events), skill (a reusable way to
  do a kind of task: subject=skill name, relation=when to use it, object=the steps),
  procedural (how-to knowledge), context (current project state).

Available tools:
{describe_tools()}

Rules:
- After a TOOL action you will receive a TOOL_RESULT; then continue.
- Use REMEMBER for durable facts, preferences and skills worth keeping for later.
- Use the provided memories and skills when relevant; do not invent memories.
- If a tool returns an ERROR, read it and fix your next action.
- Give FINAL as soon as the task is done."""


def compose_user_prompt(task, facts, skills, procedures, episodes, transcript):
    parts = ["## Current task\n" + task]
    for title, block in (("Relevant knowledge (semantic memory)", facts),
                         ("Relevant skills (skill memory)", skills),
                         ("Relevant procedures", procedures),
                         ("Related past episodes", episodes)):
        if block:
            parts.append(f"## {title}\n{block}")
    parts.append("## Conversation so far\n" + transcript.strip())
    return "\n\n".join(parts)


def format_memory(mtype, subject, relation, obj):
    return f"{subject} | {relation} | {obj}"


def format_memory_if(wanted_type, mtype, subject, relation, obj):
    """Formatted line if this atom has the wanted type, else '' (ignored by top_lines)."""
    if str(wanted_type) != str(mtype):
        return ""
    return format_memory(mtype, subject, relation, obj)


def _stem(word):
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def _rank(task, lines, limit):
    words = {_stem(w) for w in re.findall(r"\w{3,}", str(task).lower())}
    scored = []
    for line in lines:
        score = sum(1 for w in words if w in line.lower())
        if score:
            scored.append((score, line))
    scored.sort(key=lambda pair: -pair[0])
    return [line for _s, line in scored[:limit]]


def top_lines(task, lines, limit=5):
    if isinstance(lines, str):
        lines = [lines]
    unique = list(dict.fromkeys(str(x) for x in lines if str(x).strip()))
    return "\n".join(f"- {line}" for line in _rank(task, unique, limit))


def top_lines_for(task, mtype, lines):
    """Pick the memories of one type that matter for the task.

    Keyword match (with simple stemming). For episodic memory, questions about
    the past ("earlier", "previous", "completed"...) also get the newest episodes.
    """
    if isinstance(lines, str):
        lines = [lines]
    unique = list(dict.fromkeys(str(x) for x in lines if str(x).strip()))
    limit = LIMITS.get(str(mtype), 5)
    if str(mtype) == "episodic" and PAST_RE.search(str(task)):
        # a question about the past: newest completed tasks first, then keyword matches
        limit = 12
        chosen = [l for l in reversed(unique) if "| completed |" in l][:limit]
        for line in _rank(task, unique, limit):
            if len(chosen) >= limit:
                break
            if line not in chosen:
                chosen.append(line)
    else:
        chosen = _rank(task, unique, limit)
    _trace(f"recall {mtype}: {len(chosen)} of {len(unique)} memories")
    return "\n".join(f"- {line}" for line in chosen)


# ---------------------------------------------------------------- LLM calls
def llm_complete(user_prompt):
    started = time.time()
    try:
        reply = _state["provider"].complete(build_system_prompt(), user_prompt)
    except Exception as error:
        reply = f"<<LLM_ERROR>> {type(error).__name__}: {str(error)[:300]}"
    elapsed = time.time() - started
    _stats["llm_calls"] += 1
    _stats["llm_seconds"] += elapsed
    _trace(f"LLM call {elapsed:.1f}s -> {reply[:120]!r}")
    return reply


def too_long(transcript):
    return len(transcript) > _max_context()


def _clip(text, cap):
    if len(text) <= cap:
        return text
    return text[:cap // 2] + " ...[truncated]... " + text[-(cap // 2):]


def compact(transcript):
    """Shrink the transcript under the context limit.

    Newest entries are kept (by size, not by count); older ones are summarised by
    the LLM. The last entry is never touched, so a pending tool call survives.
    """
    limit = _max_context()
    cap = max(300, limit // 3)
    items = entries(transcript)
    if not items:
        return transcript
    body = [(r, _clip(c, cap)) for r, c in items[:-1]] + [items[-1]]
    budget, kept, used = limit // 2, [], 0
    for role, content in reversed(body):
        size = len(content) + len(role) + 8
        if kept and used + size > budget:
            break
        kept.append((role, content))
        used += size
    kept.reverse()
    old = body[:len(body) - len(kept)]
    if not old or all(role == "SUMMARY" for role, _c in old):
        _trace("compaction: nothing new to summarise")
        return _render(old + kept)
    # plain text for the summariser: our <<ROLE>> markers must not leak into its answer
    old_text = "\n\n".join(f"{role}: {content}" for role, content in old)[-20000:]
    try:
        summary = _state["provider"].complete(
            "You summarize an agent's earlier work. Keep facts, decisions, "
            "tool results and unfinished goals. Be concise.", old_text)
    except Exception:
        summary = old_text[:cap]
    summary = re.sub(r"<<[A-Z_]+>>\s*", "", summary).strip()
    _trace(f"compaction: {len(transcript)} -> summary of {len(old)} entries + {len(kept)} recent")
    return append_entry("", "SUMMARY", _clip(summary, cap)) + _render(kept)


def has_summary(transcript):
    items = entries(transcript)
    return bool(items) and items[0][0] == "SUMMARY"


def summary_of(transcript):
    for role, content in entries(transcript):
        if role == "SUMMARY":
            return content[:300]
    return ""


# -------------------------------------------------- memory persistence (SQL)
def _ensure_memory_table():
    with sqlite3.connect(_state["memory_db"]) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, memory_type TEXT NOT NULL,
            subject TEXT NOT NULL, relation TEXT NOT NULL, object TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(memory_type, subject, relation, object))""")


def persist_memory(mtype, subject, relation, obj):
    """Disk copy of an atom that MeTTa has just added to AtomSpace."""
    with sqlite3.connect(_state["memory_db"]) as c:
        c.execute("INSERT OR IGNORE INTO memories "
                  "(memory_type, subject, relation, object) VALUES (?,?,?,?)",
                  (mtype, subject, relation, obj))
    _trace(f"memory stored [{mtype}] {str(subject)[:60]} | {str(relation)[:40]} | {str(obj)[:60]}")
    return True


# --------------------------------------------------------------- checkpoints
def save_checkpoint(task_id, task, transcript, step, status):
    _state["checkpoints"].save_checkpoint(
        task_id, task, [{"role": "transcript", "content": transcript}],
        int(step), status)
    return True


def _cp(task_id):
    return _state["checkpoints"].load_checkpoint(task_id)


def cp_task(task_id):
    return _cp(task_id)["task"]


def cp_transcript(task_id):
    return _cp(task_id)["messages"][0]["content"]


def cp_step(task_id):
    step = _cp(task_id)["current_step"]
    return 0 if step >= max_steps() else step  # fresh allowance after a stop


def find_unfinished():
    with _state["checkpoints"].connect() as c:
        row = c.execute("SELECT task_id FROM checkpoints WHERE status='running' "
                        "ORDER BY updated_at DESC, rowid DESC LIMIT 1").fetchone()
    return row[0] if row else ""


def abandon_unfinished():
    task_id = find_unfinished()
    if not task_id:
        return "No unfinished task."
    cp = _cp(task_id)
    save_checkpoint(task_id, cp["task"], cp["messages"][0]["content"],
                    cp["current_step"], "failed")
    return f"Abandoned unfinished task {task_id}."


def pause_hint():
    return "Type /resume to retry it later, or /abandon to drop it."


# -------------------------------------------------------------------- input
def next_instruction():
    while True:
        try:
            text = input("\nagent> ").strip()
        except EOFError:
            return ""
        if text.lower() in ("exit", "quit"):
            return ""
        if text:
            return text
