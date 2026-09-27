"""Multi-turn agent sessions: the load generator for `dataset: sessions` workloads.

Each simulated user runs one agent session at a time: it sends turn 1 (a ~85k-token context),
waits for the full answer, then sends turn 2, which resends the whole context plus the next tool
result, and so on. With prefix caching the server only has to prefill each turn's new tokens.
Without it, every turn is a full prefill. Sessions are handed out from a shared queue to `c`
concurrent users.

The prefix cache is reset before each measurement (`POST /reset_prefix_cache`, available when
the server runs with VLLM_SERVER_DEV_MODE=1), so levels and repeats never warm each other. The
hit rate comes from vLLM's Prometheus counters.

Writes the same JSON keys as `vllm bench serve --save-result`, so gpubench.perf builds a
PerfPoint from it unchanged, plus first-turn / later-turn TTFT and the cache hit rate.
"""

from __future__ import annotations

import json
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx


def load_sessions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prefix_cache_counters(base_url: str) -> tuple[float, float] | None:
    """(hits, queries) summed over engines, from /metrics; None if unavailable."""
    try:
        text = httpx.get(f"{base_url}/metrics", timeout=10).text
    except httpx.HTTPError:
        return None
    totals = {"hits": 0.0, "queries": 0.0}
    found = False
    for line in text.splitlines():
        m = re.match(r"vllm:prefix_cache_(hits|queries)(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
                     line)
        if m:
            totals[m.group(1)] += float(m.group(2))
            found = True
    return (totals["hits"], totals["queries"]) if found else None


def reset_prefix_cache(base_url: str) -> bool:
    try:
        r = httpx.post(f"{base_url}/reset_prefix_cache", timeout=60)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def _stream(base_url: str, model: str, prompt: str, output_len: int, timeout: float) -> dict:
    body = {"model": model, "prompt": prompt, "max_tokens": output_len, "temperature": 0,
            "ignore_eos": True, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    chunk_times: list[float] = []
    usage = None
    with httpx.stream("POST", f"{base_url}/v1/completions", json=body, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            if data.get("usage"):
                usage = data["usage"]
            if data.get("choices") and data["choices"][0].get("text"):
                chunk_times.append(time.perf_counter())
    end = time.perf_counter()
    if not chunk_times:
        raise RuntimeError("no tokens streamed")
    out_tokens = (usage or {}).get("completion_tokens") or len(chunk_times)
    ttft = (chunk_times[0] - t0) * 1000
    e2e = (end - t0) * 1000
    return {
        "ttft": ttft,
        "itls": [(b - a) * 1000 for a, b in zip(chunk_times, chunk_times[1:], strict=False)],
        "e2el": e2e,
        "tpot": (e2e - ttft) / max(1, out_tokens - 1),
        "output_tokens": out_tokens,
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
    }


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def run_sessions(base_url: str, model: str, dataset: Path, concurrency: int, num_sessions: int,
                 output_len: int, result_path: Path, seed: int = 0, reset_cache: bool = True,
                 timeout: float = 1800) -> dict:
    """Run `num_sessions` sessions with `concurrency` users; write and return the result."""
    sessions = load_sessions(dataset)
    start = seed % len(sessions)
    picked = [sessions[(start + i) % len(sessions)] for i in range(num_sessions)]
    if reset_cache:
        reset_prefix_cache(base_url)
    before = prefix_cache_counters(base_url)

    work: queue.Queue = queue.Queue()
    for s in picked:
        work.put(s)
    results: list[dict] = []
    failed = 0
    lock = threading.Lock()

    def user() -> None:
        nonlocal failed
        while True:
            try:
                s = work.get_nowait()
            except queue.Empty:
                return
            for turn, prompt in enumerate(s["turns"]):
                try:
                    r = _stream(base_url, model, prompt, output_len, timeout)
                except Exception:
                    with lock:
                        failed += 1
                    break  # an agent can't continue a session after a failed turn
                r["turn"] = turn
                with lock:
                    results.append(r)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for _ in range(concurrency):
            pool.submit(user)
    duration = time.perf_counter() - t0
    after = prefix_cache_counters(base_url)

    ttft = [r["ttft"] for r in results]
    itls = [x for r in results for x in r["itls"]]
    out_tok = sum(r["output_tokens"] for r in results)
    in_tok = sum(r["prompt_tokens"] or 0 for r in results)
    first = [r["ttft"] for r in results if r["turn"] == 0]
    later = [r["ttft"] for r in results if r["turn"] > 0]
    out = {
        "completed": len(results), "failed": failed, "duration": duration,
        "sessions": num_sessions, "turns_per_session": len(picked[0]["turns"]) if picked else 0,
        "request_throughput": len(results) / duration,
        "output_throughput": out_tok / duration,
        "total_token_throughput": (out_tok + in_tok) / duration,
        "median_itl_ms": _pct(itls, 50),
        "ttft_first_turn_p95_ms": _pct(first, 95) if first else None,
        "ttft_later_turns_p99_ms": _pct(later, 99) if later else None,
        "prefix_cache_hit_rate": None,
    }
    for metric, xs in [("ttft", ttft), ("tpot", [r["tpot"] for r in results]), ("itl", itls),
                       ("e2el", [r["e2el"] for r in results])]:
        out[f"p95_{metric}_ms"] = _pct(xs, 95)
        out[f"p99_{metric}_ms"] = _pct(xs, 99)
    if before and after and after[1] > before[1]:
        out["prefix_cache_hit_rate"] = (after[0] - before[0]) / (after[1] - before[1])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out))
    return out
