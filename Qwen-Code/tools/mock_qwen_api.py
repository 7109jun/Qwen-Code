#!/usr/bin/env python3
"""A scripted stand-in for the Qwen (DashScope) streaming API, for end-to-end tests of the `qwen` binary.

    python3 tools/mock_qwen_api.py scenario.json PORT

scenario.json maps a model name to the list of replies it gives, in order, e.g.
{"mock-thinking": ["{\"type\":\"instruction\",...}", ...], "mock-coder": [...]}.
Every request is appended to requests.log next to the scenario file. The last reply repeats.
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

scenario = json.loads(Path(sys.argv[1]).read_text())
port = int(sys.argv[2])
log = Path(sys.argv[1]).with_name("requests.log")
cursor = {k: 0 for k in scenario}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        model = body.get("model")
        msgs = body["input"]["messages"]
        with log.open("a") as f:
            f.write(json.dumps({"model": model, "auth": self.headers.get("Authorization"), "last_user": msgs[-1]["content"][:600], "n": len(msgs)}) + "\n")
        replies = scenario.get(model)
        if not replies or self.headers.get("Authorization") != "Bearer test-key":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            data = b'{"code":"InvalidApiKey","message":"bad key or model"}'
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        i = min(cursor[model], len(replies) - 1)
        cursor[model] += 1
        text = replies[i]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        chunks = [text[k:k + 40] for k in range(0, len(text), 40)] or [""]
        events = []
        for n, c in enumerate(chunks):
            ev = {"output": {"choices": [{"message": {"role": "assistant", "content": c, "reasoning_content": "thinking..." if n == 0 else ""}, "finish_reason": "null"}]}}
            events.append(ev)
        events.append({"output": {"choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}]}, "usage": {"input_tokens": 10, "output_tokens": len(text) // 4}})
        for ev in events:
            payload = ("id:1\nevent:result\n:HTTP_STATUS/200\ndata:" + json.dumps(ev) + "\n\n").encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(payload), payload))
        self.wfile.write(b"0\r\n\r\n")


ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
