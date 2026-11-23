"""Built-in tools for the agent. Add a tool = one register() call.

All file tools are sandboxed to a workspace folder.
"""
import ast
import datetime
import math
import operator
import os
import re
import time
import urllib.parse
import urllib.request

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


def register_builtin_tools(registry, workspace="agent_workspace"):
    os.makedirs(workspace, exist_ok=True)

    def current_datetime():
        return datetime.datetime.now().strftime("%A %Y-%m-%d %H:%M:%S")

    def read_file(path):
        with open(_safe_path(workspace, path), "r", encoding="utf-8") as f:
            return f.read(20000)

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
    registry.register("web_fetch",
        "Download a web page and return its text (first ~6000 characters).",
        {"url": {"description": "http(s) URL", "type": str}}, web_fetch)
    return registry
