"""A tiny local server that speaks the OpenAI chat-completions protocol.

It lets you prove the agent works with a NON-Gemini, OpenAI-compatible endpoint
over real HTTP, without any account or key (it answers with the mock LLM).

  terminal 1:  python fake_openai_server.py            (listens on port 8099)
  terminal 2:  LLM_PROVIDER=custom LLM_BASE_URL=http://127.0.0.1:8099/v1 \
               LLM_MODEL=fake python agent_main.py --task "Calculate 5 + 3"
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

from agent.providers.openai_compat import MockTextProvider

LLM = MockTextProvider()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        messages = body.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        reply = {
            "id": "chatcmpl-fake", "object": "chat.completion",
            "created": int(time.time()), "model": body.get("model", "fake"),
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": LLM.complete(system, user)}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        print("[fake-server]", fmt % args, flush=True)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8099
    print(f"Fake OpenAI-compatible server on http://127.0.0.1:{port}/v1 (Ctrl+C to stop)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
