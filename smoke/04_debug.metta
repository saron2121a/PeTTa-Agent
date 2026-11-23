; Smoke test 4: the comparison that used to fail. Expected: 25, true, final, true, hi
!(println! (max-steps))
!(println! (< 0 (max-steps)))
!(println! (py-call (agent_bridge.classify "FINAL: hi")))
!(println! (py-call (agent_bridge.same (py-call (agent_bridge.classify "FINAL: hi")) "final")))
!(println! (agent-dispatch "final" "FINAL: hi" "t1" "task" "tr" 1))
