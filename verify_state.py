"""Check the saved state after the final integration demo (Test 10).

  python verify_state.py
Reads AGENT_CHECKPOINT_DB / AGENT_MEMORY_DB (same variables the agent uses) and the
agent_workspace folder, and prints PASS/FAIL for each check.
"""
import os
import sqlite3
import sys

CP_DB = os.getenv("AGENT_CHECKPOINT_DB", "agent_checkpoints.db")
MEM_DB = os.getenv("AGENT_MEMORY_DB", "agent_memory_store.db")
WORKSPACE = os.getenv("AGENT_WORKSPACE", "agent_workspace")
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"   ({detail})" if detail else ""))


def read(path):
    try:
        with open(os.path.join(WORKSPACE, path), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


def main():
    if not os.path.exists(CP_DB) or not os.path.exists(MEM_DB):
        print(f"Missing database ({CP_DB} or {MEM_DB}). Did you set the same "
              "AGENT_* variables you used when running the agent?")
        sys.exit(2)
    with sqlite3.connect(CP_DB) as c:
        tasks = c.execute("SELECT task_id, task, status, current_step FROM checkpoints "
                          "ORDER BY updated_at, rowid").fetchall()
    print("Tasks in checkpoint database:")
    for tid, task, status, step in tasks:
        print(f"  {tid}  {status:9} steps={step}  {task[:70]}")
    print()
    check("at least 3 tasks completed", sum(1 for t in tasks if t[2] == "completed") >= 3)
    check("no task left unfinished ('running')", not any(t[2] == "running" for t in tasks))
    with sqlite3.connect(MEM_DB) as c:
        mems = c.execute("SELECT memory_type, subject, relation, object FROM memories").fetchall()
    check("memory database has episodic records",
          sum(1 for m in mems if m[0] == "episodic") >= 3,
          f"{len(mems)} memories total")
    check("codename 'Falcon' stored as a memory",
          any("falcon" in " ".join(m[1:]).lower() for m in mems))
    squares = read("demo_squares.txt")
    check("demo_squares.txt has the squares 1..5",
          squares is not None and [x.strip() for x in squares.split()] == ["1", "4", "9", "16", "25"])
    done = read("demo_done.txt")
    check("demo_done.txt written after the interrupted task",
          done is not None and "done" in done.lower())
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
