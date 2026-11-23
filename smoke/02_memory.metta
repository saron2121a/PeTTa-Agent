; Smoke test 2: memory stored in AtomSpace and recalled through MeTTa match,
; including Amharic text. Expected: both lines print "- ... | uses | PeTTa" style results.
!(remember "semantic" "ፕሮጀክት" "uses" "PeTTa symbolic reasoning")
!(remember "skill" "summarize-file" "when asked to summarize a document" "read_file then write a short summary")
!(println! (recall-lines "tell me about ፕሮጀክት" "semantic"))
!(println! (recall-lines "please summarize a document" "skill"))
