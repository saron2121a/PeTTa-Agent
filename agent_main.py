"""Entry point. Starts PeTTa, then hands control to the MeTTa agent loop.

  python agent_main.py                 interactive agent (runs until you exit)
  python agent_main.py --task "..."    run one task and print the answer
  python agent_main.py --file X.metta  load a MeTTa file after the agent (smoke tests)
Environment: see README.md (LLM_PROVIDER, GEMINI_API_KEY, LLM_MODEL, ...).
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import agent_bridge  # noqa: E402  (must be imported so MeTTa's py-call finds it)
from petta import PeTTa  # noqa: E402


def main(argv):
    petta = PeTTa()
    restored = agent_bridge.init_runtime()
    print(f"[agent] memories restored into AtomSpace: {restored}", flush=True)
    petta.load_metta_file(os.path.join(ROOT, "agent", "agent.metta"))

    if "--file" in argv:
        petta.load_metta_file(argv[argv.index("--file") + 1])
    elif "--task" in argv:
        task = argv[argv.index("--task") + 1].replace('"', '\\"')
        result = petta.process_metta_string(f'!(agent-run "{task}")')
        print("\n".join(str(x) for x in result) if isinstance(result, list) else result)
    else:
        petta.process_metta_string("!(agent-serve)")


if __name__ == "__main__":
    main(sys.argv[1:])
