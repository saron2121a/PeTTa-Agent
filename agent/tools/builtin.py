"""Built-in tools for the agent. Add a tool = one register() call.

All file tools are sandboxed to a workspace folder.
"""
import ast
import datetime
import math
import operator
import os
import re
import signal
import subprocess
import time
import urllib.parse
import urllib.request

from agent.tools.shell_safety import destructive_reason
from agent.tools.userio import ConsoleIO

SHELL_OUTPUT_CHARS = 3500

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "abs": abs, "round": round, "sin": math.sin,
    "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCS and not node.keywords):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("Unsupported expression")


def calculate(expression):
    value = _eval(ast.parse(expression.strip(), mode="eval").body)
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _safe_path(workspace, path):
    full = os.path.abspath(os.path.join(workspace, path))
    root = os.path.abspath(workspace)
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("Path is outside the workspace")
    return full


def _shell_timeout():
    try:
        return max(1.0, float(os.getenv("AGENT_SHELL_TIMEOUT", "30")))
    except ValueError:
        return 30.0


def _clip_output(text, cap=SHELL_OUTPUT_CHARS):
    if len(text) <= cap:
        return text
    return (text[:cap // 2] + f"\n...[output truncated, {len(text)} chars total]...\n"
            + text[-(cap // 2):])


def register_builtin_tools(registry, workspace="agent_workspace", io=None):
    """Register every tool. `io` talks to the human (default: the terminal)."""
    io = io or ConsoleIO()
    os.makedirs(workspace, exist_ok=True)

    def current_datetime():
        return datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")

    def read_file(path):
        full = _safe_path(workspace, path)
        if not os.path.exists(full):
            return (f"ERROR: file not found: {path}. "
                    "Use list_files to see what exists, or check the spelling.")
        if os.path.isdir(full):
            return f"ERROR: {path} is a folder, not a file. Use list_files."
        with open(full, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(20000)
        return text if text else "(the file is empty)"

    def write_file(path, content):
        full = _safe_path(workspace, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} characters to {path}"

    def list_files():
        names = []
        for base, _dirs, files in os.walk(workspace):
            for name in files:
                names.append(os.path.relpath(os.path.join(base, name), workspace))
        return "\n".join(sorted(names)) or "(workspace is empty)"

    def web_fetch(url):
        if urllib.parse.urlparse(url).scheme not in ("http", "https"):
            raise ValueError("Only http/https URLs are allowed")
        request = urllib.request.Request(url, headers={"User-Agent": "petta-agent"})
        with urllib.request.urlopen(request, timeout=15) as response:
            raw = response.read(40000).decode("utf-8", errors="replace")
        text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", raw, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()[:6000]

    def run_shell(command):
        command = str(command).strip()
        if not command:
            return "ERROR: empty command"
        reason = destructive_reason(command)
        if reason:
            allowed = io.confirm(f"The agent wants to run: {command}\n  Why it needs approval: {reason}")
            if not allowed:
                return ("DENIED: the user did not approve this command (" + reason + "). "
                        "Do not retry it. Choose a safer way, or ask the user what to do.")
        timeout = _shell_timeout()
        try:
            proc = subprocess.Popen(
                command, shell=True, cwd=os.path.abspath(workspace),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors="replace",
                start_new_session=True)
        except Exception as error:
            return f"ERROR: could not start the command: {type(error).__name__}: {error}"
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            partial = ""
            try:
                partial = proc.communicate(timeout=5)[0] or ""
            except Exception:
                pass
            return (f"ERROR: the command timed out after {timeout:g} seconds and was stopped."
                    " Try a faster or smaller command.\n" + _clip_output(partial))
        output = _clip_output(output or "")
        if proc.returncode == 0:
            return output if output.strip() else "(command finished with no output)"
        return f"ERROR: the command failed with exit code {proc.returncode}.\n{output}"

    def send_message(message):
        io.say(str(message))
        return "Message delivered to the user."

    def ask_user(question):
        answer = io.ask(str(question))
        if answer is None:
            return ("ERROR: no user is available to answer right now. "
                    "State your assumption clearly in your final answer, or stop and explain what is missing.")
        answer = answer.strip()
        return answer if answer else "(the user gave an empty answer)"

    registry.register("calculator",
        "Evaluate a math expression (+ - * / // % **, sqrt, sin, log, pi...).",
        {"expression": {"description": "e.g. (5+3)*2", "type": str}}, calculate)
    registry.register("current_datetime",
        "Get the current local date and time.", {}, current_datetime)
    registry.register("read_file",
        "Read a text file from the agent workspace.",
        {"path": {"description": "relative path inside the workspace", "type": str}},
        read_file)
    registry.register("write_file",
        "Write (overwrite) a text file in the agent workspace.",
        {"path": {"description": "relative path inside the workspace", "type": str},
         "content": {"description": "full file content", "type": str}}, write_file)
    registry.register("list_files",
        "List the files in the agent workspace.", {}, list_files)
    def wait(seconds):
        seconds = max(0.0, min(float(seconds), 120.0))
        time.sleep(seconds)
        return f"Waited {seconds:g} seconds"

    registry.register("wait",
        "Pause for a number of seconds (max 120), e.g. to wait before a next step.",
        {"seconds": {"description": "how many seconds", "type": float}}, wait)
    registry.register("run_shell",
        "Run a shell command inside the agent workspace and return its output. "
        "Has a timeout. Destructive commands (rm, mv, overwriting with >, sudo...) "
        "need the user's approval first.",
        {"command": {"description": "the shell command line", "type": str}}, run_shell)
    registry.register("send_message",
        "Show a message to the user (a progress note or result) without ending the task.",
        {"message": {"description": "text to show the user", "type": str}}, send_message)
    registry.register("ask_user",
        "Ask the user a question and wait for the typed answer. Use it whenever "
        "required information is missing instead of guessing.",
        {"question": {"description": "the question for the user", "type": str}}, ask_user)
    registry.register("web_fetch",
        "Download a web page and return its text (first ~6000 characters).",
        {"url": {"description": "http(s) URL", "type": str}}, web_fetch)
    return registry
