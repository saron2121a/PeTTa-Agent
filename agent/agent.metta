; =====================================================================
;  MeTTa agent: the loop, the state and every decision live HERE.
;  Python (agent_bridge.py) is only called through py-call for things
;  MeTTa cannot do: the LLM HTTP call, running tools, text parsing, SQLite.
;
;  State is passed explicitly:  $tid (task id)  $task  $transcript  $step
;  Memory is AtomSpace atoms in &agent_memories:
;        (agent-memory <type> <subject> <relation> <object>)
;  types: semantic | episodic | skill | procedural | context
; =====================================================================

(= (max-steps) (py-call (agent_bridge.max_steps)))

; ---------------------------------------------------------------- memory
(= (remember $type $subject $relation $object)
   (let* (($_a (add-atom &agent_memories
                         (agent-memory $type $subject $relation $object)))
          ($_b (py-call (agent_bridge.persist_memory $type $subject $relation $object))))
     stored))

; Fetch memories FROM ATOMSPACE with MeTTa match. The type check and the
; ranking by task words are small Python helpers (strings made in Python and
; strings typed in .metta files are different values in PeTTa, so == fails).
(= (recall-lines $task $type)
   (py-call (agent_bridge.top_lines_for $task $type
      (collapse (match &agent_memories
                       (agent-memory $t $s $r $o)
                       (py-call (agent_bridge.format_memory_if $type $t $s $r $o)))))))

(= (build-prompt $task $transcript)
   (let* (($facts  (recall-lines $task "semantic"))
          ($skills (recall-lines $task "skill"))
          ($procs  (recall-lines $task "procedural"))
          ($past   (recall-lines $task "episodic")))
     (py-call (agent_bridge.compose_user_prompt
                 $task $facts $skills $procs $past $transcript))))

; --------------------------------------------------- context management
(= (maybe-compact $tid $transcript)
   (if (py-call (agent_bridge.too_long $transcript))
       (let $new (py-call (agent_bridge.compact $transcript))
         (if (py-call (agent_bridge.has_summary $new))
             (let $_m (remember "episodic"
                                (py-call (agent_bridge.task_label $tid))
                                "summary"
                                (py-call (agent_bridge.summary_of $new)))
               $new)
             $new))
       $transcript))

; --------------------------------------------------------- the main loop
(= (agent-run $task)
   (let* (($tid (py-call (agent_bridge.new_task_id)))
          ($t0  (py-call (agent_bridge.append_entry "" "USER" $task)))
          ($_c  (py-call (agent_bridge.save_checkpoint $tid $task $t0 0 "running"))))
     (agent-step $tid $task $t0 0)))

(= (agent-resume $tid)
   (let* (($task (py-call (agent_bridge.cp_task $tid)))
          ($tr   (py-call (agent_bridge.cp_transcript $tid)))
          ($step (py-call (agent_bridge.cp_step $tid))))
     (agent-step $tid $task $tr $step)))

; One turn: stop at the step limit; run a tool the model asked for (also
; after a crash/restart); otherwise ask the model what to do next.
(= (agent-step $tid $task $transcript $step)
   (if (< $step (max-steps))
       (if (py-call (agent_bridge.pending_tool $transcript))
           (agent-act $tid $task $transcript $step)
           (agent-think $tid $task $transcript $step))
       (agent-give-up $tid $task $transcript $step)))

(= (agent-think $tid $task $transcript $step)
   (let* (($tc     (maybe-compact $tid $transcript))
          ($prompt (build-prompt $task $tc))
          ($reply  (py-call (agent_bridge.llm_complete $prompt)))
          ($kind   (py-call (agent_bridge.classify $reply)))
          ($next   (+ $step 1)))
     (if (py-call (agent_bridge.same $kind "error"))
         (agent-pause $reply)
         (let* (($t2 (py-call (agent_bridge.append_entry $tc "ASSISTANT" $reply)))
                ($_c (py-call (agent_bridge.save_checkpoint $tid $task $t2 $next "running"))))
           (agent-dispatch $kind $reply $tid $task $t2 $next)))))

(= (agent-dispatch $kind $reply $tid $task $transcript $step)
   (if (py-call (agent_bridge.same $kind "final"))
       (agent-finish $tid $task $transcript $step $reply)
       (if (py-call (agent_bridge.same $kind "remember"))
           (agent-remember $tid $task $transcript $step $reply)
           (if (py-call (agent_bridge.same $kind "tool"))
               (agent-step $tid $task $transcript $step)
               (agent-continue $tid $task $transcript $step "SYSTEM"
                  "Invalid format. Reply with FINAL:, TOOL:/ARGS: or REMEMBER:.")))))

; Run the pending tool (Python), record the result, keep going.
(= (agent-act $tid $task $transcript $step)
   (let* (($result (py-call (agent_bridge.run_pending_tool $transcript))))
     (agent-continue $tid $task $transcript $step "TOOL_RESULT" $result)))

; Store a memory atom in AtomSpace (MeTTa), then keep going.
(= (agent-remember $tid $task $transcript $step $reply)
   (let* (($ty (py-call (agent_bridge.rem_field $reply 0)))
          ($su (py-call (agent_bridge.rem_field $reply 1)))
          ($re (py-call (agent_bridge.rem_field $reply 2)))
          ($ob (py-call (agent_bridge.rem_field $reply 3))))
     (if (py-call (agent_bridge.same $ty ""))
         (agent-continue $tid $task $transcript $step "SYSTEM"
            "REMEMBER format error. Use: REMEMBER: type | subject | relation | object")
         (let* (($_m (remember $ty $su $re $ob)))
           (agent-continue $tid $task $transcript $step "SYSTEM" "Memory stored.")))))

(= (agent-continue $tid $task $transcript $step $role $content)
   (let* (($t2 (py-call (agent_bridge.append_entry $transcript $role $content)))
          ($_c (py-call (agent_bridge.save_checkpoint $tid $task $t2 $step "running"))))
     (agent-step $tid $task $t2 $step)))

(= (agent-finish $tid $task $transcript $step $reply)
   (let* (($answer (py-call (agent_bridge.extract_final $reply)))
          ($_c (py-call (agent_bridge.save_checkpoint $tid $task $transcript $step "completed")))
          ($_m (remember "episodic"
                         (py-call (agent_bridge.task_label $tid))
                         "completed"
                         (py-call (agent_bridge.short $task $answer)))))
     $answer))

(= (agent-give-up $tid $task $transcript $step)
   (let* (($_c (py-call (agent_bridge.save_checkpoint $tid $task $transcript $step "failed"))))
     "Maximum agent steps reached."))

; LLM unreachable: leave the checkpoint 'running' so it resumes on next start.
(= (agent-pause $reply)
   (py-call (agent_bridge.pause_message $reply)))

; ----------------------------------------------- run indefinitely (serve)
(= (agent-serve)
   (let* (($_h (py-call (agent_bridge.show "PeTTa agent ready. Type an instruction ('exit' to quit).")))
          ($pending (py-call (agent_bridge.find_unfinished))))
     (if (py-call (agent_bridge.same $pending ""))
         (agent-loop)
         (let* (($_r   (py-call (agent_bridge.show "Resuming an unfinished task...")))
                ($ans  (agent-resume $pending))
                ($_p   (py-call (agent_bridge.show $ans))))
           (agent-loop)))))

(= (agent-loop)
   (let $instr (py-call (agent_bridge.next_instruction))
     (if (py-call (agent_bridge.same $instr ""))
         (py-call (agent_bridge.show "Goodbye."))
         (let* (($ans (agent-handle $instr))
                ($_p  (py-call (agent_bridge.show $ans))))
           (agent-loop)))))

; Commands: /resume retries a paused task, /abandon drops it; anything else is a task.
(= (agent-handle $instr)
   (if (py-call (agent_bridge.same $instr "/resume"))
       (agent-resume-pending)
       (if (py-call (agent_bridge.same $instr "/abandon"))
           (py-call (agent_bridge.abandon_unfinished))
           (agent-run $instr))))

(= (agent-resume-pending)
   (let $pending (py-call (agent_bridge.find_unfinished))
     (if (py-call (agent_bridge.same $pending ""))
         "No unfinished task."
         (agent-resume $pending))))
