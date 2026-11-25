# Live CLI Demonstration

**Date:** 24 November 2026
**Interface:** Interactive terminal CLI
**Provider/model:** Gemini / gemini-3.5-flash-lite

This document records a real interactive test of the PeTTa agent. It is intended as presentation evidence and does not replace the committed task execution logs.

## 1. Basic file task

**User request**

> Create a file called greeting.txt containing exactly: Hello from my PeTTa agent. Then read the file and tell me what it contains.

**Observed agent actions**

1. Called `write_file` with `greeting.txt` and the requested text.
2. Received: `Wrote 26 characters to greeting.txt`.
3. Called `read_file` for `greeting.txt`.
4. Received: `Hello from my PeTTa agent.`
5. Returned a final response confirming the file contents.

**What this demonstrates:** The agent can interpret a task, select tools, execute them in sequence, use tool results, and report completion.

## 2. Asking the user for missing information

**User request**

> Create a professional invitation for an event and save it as invitation.txt.

**Observed agent action**

The agent called `ask_user` and asked:

> What is the name, date, time, and location of the event you would like the invitation for?

The agent waited for the user's response rather than inventing those details.

**User response**

> The event is a university technology exhibition on December 10, 2026, from 9:00 AM to 1:00 PM at the university main hall.

**Observed agent actions after the answer**

1. Received the answer through `ask_user`.
2. Called `write_file` to create `invitation.txt`.
3. Received: `Wrote 375 characters to invitation.txt`.
4. Called `read_file` to read the invitation.
5. Received the file contents (the terminal trace displayed a shortened preview).
6. Returned a final response confirming completion.

**What this demonstrates:** The agent can pause a task, request missing information, receive the user's answer, and continue the same task.

**Performance note:** The initial model response that selected `ask_user` took approximately 121.9 seconds in this run. Hosted-model latency can therefore affect live demonstrations.

## 3. Multiple tasks in one session

**User request**

> List the files in my workspace and tell me which ones I created during this session.

**Observed agent actions**

1. Called `list_files` to inspect the workspace.
2. Called `run_shell` with `ls -lu` to inspect file information.
3. Used available episodic task history and file information to identify the two files associated with the earlier tasks.
4. Returned a final answer identifying `greeting.txt` and `invitation.txt` as created during this session.

The agent then accepted the `exit` command and printed:

`Goodbye.`

**What this demonstrates:** The CLI can accept multiple tasks in one running session, use previous task history, return to its input prompt, and exit cleanly.

## 4. Test summary

| Capability                           | Observed result |
| ------------------------------------ | --------------- |
| Interactive CLI input                | Passed          |
| Sequential file writing and reading  | Passed          |
| Asking the user for missing details  | Passed          |
| Continuing after receiving an answer | Passed          |
| Multiple tasks without restarting    | Passed          |
| Use of episodic task history         | Demonstrated    |
| Clean CLI exit                       | Passed          |

## 5. Presentation notes and limitations

- The transcript records actual tool calls and outputs from the test session.
- The invitation's full text was written and read by the agent, but the captured terminal trace showed only a shortened preview.
- File timestamps and episodic history were used in the final workspace-listing task. The agent's conclusion was not an independent audit of every file's creation history.
- Response times depend on the hosted model and network conditions. The `ask_user` selection took approximately two minutes in this run.
- This was a terminal-based demonstration, not a graphical chat interface.
