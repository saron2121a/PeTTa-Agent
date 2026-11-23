"""Measure memory recall as the number of stored memories grows.

  python scale_test.py 1000      (then try 5000)
Uses its own database (scale_test.db). Hides one 'needle' fact among N distractors,
loads everything into AtomSpace, and times recall through MeTTa. No LLM key needed.
"""
import os
import random
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ["LLM_PROVIDER"] = "mock"
os.environ["AGENT_MEMORY_DB"] = "scale_test.db"
os.environ["AGENT_CHECKPOINT_DB"] = "scale_test_cp.db"

import agent_bridge  # noqa: E402
from petta import PeTTa  # noqa: E402

COLORS = ["red", "blue", "green", "amber", "violet", "teal"]
THINGS = ["lamp", "chair", "river", "engine", "garden", "window", "bridge", "pencil"]


def main(n):
    for f in ("scale_test.db", "scale_test_cp.db"):
        if os.path.exists(f):
            os.remove(f)
    agent_bridge.configure(provider=None, memory_db="scale_test.db",
                           checkpoint_db="scale_test_cp.db")
    rows = [("semantic", f"{random.choice(THINGS)} number {i}", "has colour",
             random.choice(COLORS)) for i in range(n)]
    rows.append(("semantic", "Zanzibar launch code", "is", "BLUEFIN-42"))
    with sqlite3.connect("scale_test.db") as c:
        c.executemany("INSERT OR IGNORE INTO memories (memory_type, subject, relation, object) "
                      "VALUES (?,?,?,?)", rows)

    petta = PeTTa()
    t0 = time.time()
    restored = agent_bridge.init_runtime()
    restore_s = time.time() - t0
    petta.load_metta_file(os.path.join(ROOT, "agent", "agent.metta"))

    times, ok = [], 0
    for _ in range(3):
        t0 = time.time()
        out = petta.process_metta_string('!(recall-lines "what is the Zanzibar launch code" "semantic")')
        times.append(time.time() - t0)
        ok += "BLUEFIN-42" in str(out)
    print(f"memories in AtomSpace : {restored}")
    print(f"restore from SQLite   : {restore_s:.2f}s")
    print(f"recall (3 runs)       : {', '.join(f'{t:.3f}s' for t in times)}")
    print(f"needle found          : {ok}/3")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)
