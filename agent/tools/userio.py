"""Talking to the human at the terminal (used by send_message, ask_user and approvals).

Kept in one small class so tests can swap in a fake and so the agent never
crashes when there is nobody to answer (stdin closed / piped input).
"""


class ConsoleIO:

    def say(self, text):
        print(f"\n[agent] {text}", flush=True)

    def ask(self, question):
        """Return the user's typed answer, or None if no user can answer."""
        try:
            return input(f"\n[agent asks] {question}\nyou> ")
        except EOFError:
            return None

    def confirm(self, prompt):
        """True only if the user explicitly types y / yes."""
        try:
            answer = input(f"\n[approval needed] {prompt}\nAllow? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")
