; Smoke test 1: can MeTTa call my Python module? (run: python agent_main.py --file smoke/01_pycall.metta)
; Expected output: pong / bool ok / "a | b (len=2)"
!(println! (py-call (agent_bridge.ping)))
!(if (py-call (agent_bridge.is_true)) (println! "bool ok") (println! "bool FAILED"))
!(println! (py-call (agent_bridge.echo_list (collapse (superpose ("a" "b"))))))
