; Smoke test 3: the full MeTTa loop with the mock LLM (run with LLM_PROVIDER=mock).
; Expected: a final answer containing 8, then a confirmation that memory was stored.
!(println! (agent-run "Calculate 5 + 3"))
!(println! (agent-run "Remember that my project uses PeTTa"))
!(println! (recall-lines "project PeTTa" "semantic"))
