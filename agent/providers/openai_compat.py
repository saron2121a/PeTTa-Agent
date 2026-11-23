"""Text-in / text-out provider for any OpenAI-compatible endpoint.

Works with Gemini (its OpenAI-compatible endpoint), OpenAI, OpenRouter,
Ollama, Groq, local servers, ... Only plain text is exchanged, so tools are
described in the prompt header and parsed from the reply.

Reliability features (v3):
  * retries on 429/5xx/timeouts, honouring the server's Retry-After / "retry in Xs"
  * a daily-quota 429 is not retried (waiting seconds cannot fix it)
  * optional fallback models (separate quotas): LLM_FALLBACK_MODELS
  * optional minimum gap between calls: LLM_MIN_INTERVAL (seconds)
"""
import os
import random
import re
import sys
import time

PRESETS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": "gemini-3.5-flash",
        "fallbacks": ["gemini-flash-latest"],
    },
    "openai": {"base_url": None, "key_env": "OPENAI_API_KEY", "model": "gpt-4o-mini"},
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "key_env": "OPENROUTER_API_KEY",
        "model": "openai/gpt-4o-mini",
    },
    "ollama": {"base_url": "http://localhost:11434/v1", "key_env": None, "model": "llama3.1"},
}
RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}
RETRY_NAMES = {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}
FALLBACK_STATUS = {404, 429}


def _log(message):
    print(f"[llm] {message}", file=sys.stderr, flush=True)


class OpenAICompatibleProvider:

    def __init__(self, base_url=None, api_key=None, model=None, client=None,
                 max_retries=6, sleep=time.sleep, fallback_models=None,
                 min_interval=0.0, clock=time.monotonic, max_wait=90.0):
        self.model = model
        self.fallback_models = list(fallback_models or [])
        self.max_retries = max_retries
        self.min_interval = float(min_interval or 0)
        self.max_wait = max_wait
        self._sleep = sleep
        self._clock = clock
        self._last_call = None
        self._cooldown = {}  # model -> time until which a failed model is skipped
        if client is None:
            from openai import OpenAI  # imported lazily so tests need no package
            # max_retries=0: this class does its own (visible, smarter) retrying
            client = OpenAI(api_key=api_key or "none", base_url=base_url,
                            max_retries=0, timeout=120)
        self.client = client

    # ---- error inspection -------------------------------------------------
    @staticmethod
    def _status(error):
        return getattr(error, "status_code", None)

    @staticmethod
    def _text(error):
        return f"{getattr(error, 'body', '')} {error}"

    def _retryable(self, error):
        return type(error).__name__ in RETRY_NAMES or self._status(error) in RETRY_STATUS

    def _is_daily_quota(self, error):
        text = self._text(error).lower()
        return "perday" in text or "per day" in text

    def _retry_after(self, error):
        headers = getattr(getattr(error, "response", None), "headers", None)
        if headers is not None:
            try:
                value = headers.get("retry-after")
                if value is not None:
                    return float(value)
            except (TypeError, ValueError):
                pass
        text = self._text(error)
        m = (re.search(r"retry in ([\d.]+)\s*s", text, re.I)
             or re.search(r"retryDelay['\"]?\s*:\s*['\"]([\d.]+)s", text, re.I))
        return float(m.group(1)) if m else None

    def _can_fallback(self, error):
        return (self._status(error) in FALLBACK_STATUS
                or type(error).__name__ in {"RateLimitError", "NotFoundError"})

    # ---- calling ----------------------------------------------------------
    def _throttle(self):
        if self.min_interval > 0 and self._last_call is not None:
            gap = self.min_interval - (self._clock() - self._last_call)
            if gap > 0:
                self._sleep(gap)
        self._last_call = self._clock()

    def _complete_one(self, model, messages):
        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages)
                return response.choices[0].message.content or ""
            except Exception as error:
                if (not self._retryable(error) or self._is_daily_quota(error)
                        or attempt >= self.max_retries):
                    raise
                wait = self._retry_after(error)
                if wait is None:
                    wait = min(2 ** attempt, 30) + random.random()
                wait = min(wait, self.max_wait)
                _log(f"rate limited/unavailable ({self._status(error)}), "
                     f"retry {attempt + 1}/{self.max_retries} in {wait:.0f}s")
                self._sleep(wait)

    def _models_to_try(self):
        models = [self.model] + [m for m in self.fallback_models if m != self.model]
        now = self._clock()
        usable = [m for m in models if self._cooldown.get(m, 0) <= now]
        return usable or models  # if every model is cooling down, try them anyway

    def _cool_down(self, model, error):
        if self._status(error) == 404:
            seconds = 86400
        elif self._is_daily_quota(error):
            seconds = 3600
        else:
            seconds = 120
        self._cooldown[model] = self._clock() + seconds

    def complete(self, system, user):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        models = self._models_to_try()
        for index, model in enumerate(models):
            started = self._clock()
            try:
                reply = self._complete_one(model, messages)
                if self._clock() - started > 5:
                    _log(f"{model} answered in {self._clock() - started:.1f}s")
                return reply
            except Exception as error:
                if not self._can_fallback(error):
                    raise
                self._cool_down(model, error)
                if index + 1 >= len(models):
                    raise
                _log(f"model {model} failed ({self._status(error)}) after "
                     f"{self._clock() - started:.1f}s; skipping it for a while, "
                     f"trying {models[index + 1]}")


class MockTextProvider:
    """Deterministic fake LLM that speaks the text protocol (for tests/demos)."""

    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        if "You summarize" in system:
            return "Summary of the earlier steps."
        m = re.search(r"## Current task\n(.*?)\n\n## ", user, re.S)
        task = (m.group(1) if m else user).strip()
        convo = user.split("## Conversation so far", 1)[-1]
        results = re.findall(r"<<TOOL_RESULT>>\n(.*?)\n\n", convo, re.S)
        if results:
            return f"FINAL: The tool returned {results[-1]}"
        if "Memory stored." in convo:
            return "FINAL: I have stored that in memory."
        if task.lower().startswith("remember"):
            return "REMEMBER: semantic | user project | uses | PeTTa"
        expr = re.search(r"(\d[\d\s+\-*/().]*\d)", task)
        if expr:
            return 'TOOL: calculator\nARGS: {"expression": "%s"}' % expr.group(1).strip()
        return f"FINAL: You said: {task}"


def create_provider(name=None):
    """Build a provider from environment variables.

    LLM_PROVIDER = gemini | openai | openrouter | ollama | mock | custom
    LLM_BASE_URL, LLM_API_KEY, LLM_MODEL override the preset values.
    LLM_FALLBACK_MODELS=a,b (or "none"), LLM_MIN_INTERVAL=seconds.
    """
    name = (name or os.getenv("LLM_PROVIDER") or "gemini").lower()
    if name == "mock":
        return MockTextProvider()
    preset = PRESETS.get(name, {"base_url": None, "key_env": None, "model": None})
    base_url = os.getenv("LLM_BASE_URL") or preset["base_url"]
    api_key = os.getenv("LLM_API_KEY") or (
        os.getenv(preset["key_env"]) if preset.get("key_env") else None)
    model = os.getenv("LLM_MODEL") or preset["model"]
    if not model:
        raise ValueError("Set LLM_MODEL for this provider")
    local = bool(base_url) and any(h in base_url for h in ("localhost", "127.0.0.1"))
    if not api_key and not local and name != "ollama":
        raise ValueError("No API key found: set LLM_API_KEY"
                         + (f" or {preset['key_env']}" if preset.get("key_env") else ""))
    raw = os.getenv("LLM_FALLBACK_MODELS")
    if raw is None:
        fallbacks = preset.get("fallbacks", [])
    else:
        fallbacks = [] if raw.strip().lower() == "none" else [
            m.strip() for m in raw.split(",") if m.strip()]
    return OpenAICompatibleProvider(
        base_url=base_url, api_key=api_key, model=model, fallback_models=fallbacks,
        min_interval=float(os.getenv("LLM_MIN_INTERVAL", "0") or 0))
