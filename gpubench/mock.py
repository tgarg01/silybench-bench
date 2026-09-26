"""A GPU-free stand-in for vLLM, so CI can run the AGENTS.md flow end to end.

`python -m gpubench.mock serve` is a tiny OpenAI-compatible server whose latency grows with the
number of in-flight requests (so the capacity search has something to find), and
`python -m gpubench.mock bench <vllm bench serve args>` drives it and writes a result JSON in
vllm's format. Runs made this way are marked runtime=mock and can never be submitted.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

KV_CACHE_TOKENS = 100_000
BASE_TTFT_MS = 20.0
BASE_ITL_MS = 4.0
ITL_PER_ACTIVE_MS = 1.0  # => the 50 ms ITL target is crossed around 46 users


class _State:
    active = 0
    lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the log to startup lines
        pass

    def do_GET(self) -> None:
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        n = int(body.get("max_tokens") or 16)
        with _State.lock:
            _State.active += 1
        try:
            itl = (BASE_ITL_MS + ITL_PER_ACTIVE_MS * _State.active) / 1000
            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                time.sleep(BASE_TTFT_MS / 1000)
                for i in range(n):
                    chunk = {"choices": [{"index": 0, "text": f"t{i} "}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(itl)
                self.wfile.write(b"data: [DONE]\n\n")
                return
            time.sleep(BASE_TTFT_MS / 1000 + itl * n)
            tokens = [f"t{i}" for i in range(n)]
            top = [{t: -0.01, "x": -5.0} for t in tokens]
            text = "<tool_call>\n<function=run_command>\n<parameter=command>\nls\n</parameter>\n"\
                   "</function>\n</tool_call>" if n > 100 else "mock answer"
            out = {"choices": [{"index": 0, "text": text, "finish_reason": "length",
                                "logprobs": {"tokens": tokens, "top_logprobs": top}}]}
            data = json.dumps(out).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            with _State.lock:
                _State.active -= 1


def serve(port: int) -> None:
    # The same startup line vLLM prints, parsed by gpubench.server.parse_kv_cache_tokens.
    print(f"INFO mock vLLM: GPU KV cache size: {KV_CACHE_TOKENS:,} tokens", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def bench(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--dataset-name")
    ap.add_argument("--dataset-path")
    ap.add_argument("--num-prompts", type=int)
    ap.add_argument("--max-concurrency", type=int)
    ap.add_argument("--random-input-len", type=int, default=0)
    ap.add_argument("--random-output-len", type=int, default=0)
    ap.add_argument("--custom-output-len", type=int, default=0)
    ap.add_argument("--result-dir")
    ap.add_argument("--result-filename")
    args, _ = ap.parse_known_args(argv)
    out_len = args.random_output_len or args.custom_output_len or 16
    in_len = args.random_input_len
    if args.dataset_name == "custom":
        prompts = [json.loads(line)["prompt"] for line in
                   Path(args.dataset_path).read_text().splitlines() if line.strip()]
        in_len = max(1, len(prompts[0].split()))

    def one(_: int) -> dict:
        t0 = time.perf_counter()
        times = []
        with httpx.stream("POST", f"{args.base_url}/v1/completions", timeout=120,
                          json={"model": args.model, "prompt": "x", "max_tokens": out_len,
                                "stream": True}) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    times.append(time.perf_counter())
        ttft = (times[0] - t0) * 1000
        itls = [(b - a) * 1000 for a, b in zip(times, times[1:], strict=False)]
        e2e = (times[-1] - t0) * 1000
        return {"ttft": ttft, "itls": itls, "e2el": e2e,
                "tpot": (e2e - ttft) / max(1, len(times) - 1)}

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        res = list(pool.map(one, range(args.num_prompts)))
    duration = time.perf_counter() - start
    itls = [x for r in res for x in r["itls"]]
    out = {"completed": len(res), "failed": 0, "duration": duration,
           "request_throughput": len(res) / duration,
           "output_throughput": len(res) * out_len / duration,
           "total_token_throughput": len(res) * (out_len + in_len) / duration,
           "median_itl_ms": statistics.median(itls) if itls else 0.0}
    for metric, xs in [("ttft", [r["ttft"] for r in res]), ("tpot", [r["tpot"] for r in res]),
                       ("itl", itls), ("e2el", [r["e2el"] for r in res])]:
        out[f"p95_{metric}_ms"] = _pct(xs, 95)
        out[f"p99_{metric}_ms"] = _pct(xs, 99)
    Path(args.result_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.result_dir) / args.result_filename).write_text(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(int(sys.argv[sys.argv.index("--port") + 1]))
    elif sys.argv[1] == "bench":
        bench(sys.argv[2:])
