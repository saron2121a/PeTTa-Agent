"""Run a fixed set of tasks through the MeTTa agent and print a results table.

  python benchmark.py            uses your configured LLM (Gemini)
Writes benchmark_results.md (put it on your slides). Uses separate bench databases
so your real memories are untouched. Tip: export LLM_MIN_INTERVAL=5 on the free tier.
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("AGENT_MEMORY_DB", "bench_memory.db")
os.environ.setdefault("AGENT_CHECKPOINT_DB", "bench_checkpoints.db")

import agent_bridge  # noqa: E402
from petta import PeTTa  # noqa: E402

# (name, task, any-of groups: every group needs at least one of its words in the answer)
TASKS = [
    ("calculator", "What is 123*45? Use the calculator.", [["5535", "5,535"]]),
    ("multi-step files",
     "Create bench_squares.txt with the squares of 1 to 10, one per line, read it back and tell me the sum.",
     [["385"]]),
    ("memory write", "Remember that my benchmark codename is Falcon.", [["remember", "stored", "noted", "falcon"]]),
    ("memory recall", "What is my benchmark codename?", [["falcon"]]),
    ("tool error", "Read the file nonexistent_bench.txt and tell me what it says.",
     [["not", "no such", "doesn't", "does not", "missing", "couldn't", "cannot", "can't"]]),
    ("date tool", "What day of the week is it today? Use the date tool.",
     [["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]]),
]


def main():
    petta = PeTTa()
    agent_bridge.init_runtime()
    petta.load_metta_file(os.path.join(ROOT, "agent", "agent.metta"))
    rows, started = [], time.time()
    for name, task, groups in TASKS:
        agent_bridge.stats_reset()
        t0 = time.time()
        result = petta.process_metta_string('!(agent-run "%s")' % task.replace('"', '\\"'))
        answer = " ".join(str(x) for x in result) if isinstance(result, list) else str(result)
        seconds = time.time() - t0
        ok = all(any(w in answer.lower() for w in g) for g in groups)
        rows.append((name, ok, seconds, agent_bridge._stats["llm_calls"],
                     agent_bridge._stats["tool_calls"], answer.replace("\n", " ")[:70]))
        print(f"{'PASS' if ok else 'FAIL'}  {name:18} {seconds:6.1f}s  {answer[:80]!r}", flush=True)
    total = time.time() - started
    passed = sum(1 for r in rows if r[1])
    lines = ["| Task | Result | Seconds | LLM calls | Tool calls | Answer |", "|---|---|---|---|---|---|"]
    for name, ok, sec, llm, tools, ans in rows:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {sec:.1f} | {llm} | {tools} | {ans} |")
    lines.append(f"\n**{passed}/{len(rows)} passed, {total:.0f}s total**")
    open("benchmark_results.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
