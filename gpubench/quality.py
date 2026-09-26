"""Quality checks for custom long-context scenarios (the perf runs force fixed-length outputs,
so they say nothing about correctness).

Built from exactly the same agent contexts as the perf prompts, rendered with thinking off:

- recall: after the 100k-token context, ask for the first output line of a command the agent
  ran deep inside it (20-80% depth). The answer is in the context verbatim, so scoring is
  exact. Catches long-context attention / KV-cache bugs, the typical failure of long-context
  kernel optimizations.
- drift: the context as-is, at the moment the agent decides its next tool call. Greedy
  decoding with top-20 logprobs. Not scored on its own; it is the baseline an optimized build is
  compared with (same tool call? how far do the token distributions move?).

Every run records responses in <run_dir>/quality/<workload>.jsonl. `gpubench quality compare`
checks a candidate (after an optimization) against the baseline (the published run).
"""

from __future__ import annotations

import json
import math
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel

from gpubench.prompts import fitted_contexts, sha256_file

# Output lines that aren't facts (SWE-agent's status footer, our trim marker, ...).
_BORING = ("(Open file", "(Current directory", "bash-$", "[...]", "[File:", "Your command ran",
           "File updated", "Found ", "No matches", "(", "[", "---", "```")
QUESTION = ("Question about this session. Answer from the context above; do not call any tool. "
            "What was the first line of the output of the command `{cmd}`? Reply with exactly "
            "that line and nothing else.")


class QualityResult(BaseModel):
    workload: str
    dataset_sha256: str
    responses: str  # file under the run dir, e.g. quality/toolcall-100k-512.jsonl
    recall_n: int = 0
    recall_accuracy: float | None = None  # expected line appears in the answer
    recall_exact: float | None = None  # answer's first line equals the expected line
    drift_n: int = 0
    drift_tool_call_rate: float | None = None  # share of answers containing a tool call


# --- building -----------------------------------------------------------------------------------

def _norm(s: str) -> str:
    return " ".join(s.split())


def recall_candidates(messages: list[dict], text: str) -> list[dict]:
    """(command, first output line, depth) pairs usable as exact-answer questions."""
    calls = [(i, m["tool_calls"][0]["function"]["arguments"]["command"])
             for i, m in enumerate(messages)
             if m["role"] == "assistant" and m.get("tool_calls")]
    counts: dict[str, int] = {}
    for _, cmd in calls:
        counts[cmd] = counts.get(cmd, 0) + 1
    out = []
    for i, cmd in calls:
        depth = i / len(messages)
        if not 0.2 <= depth <= 0.8 or counts[cmd] != 1 or len(cmd) < 4 or cmd == "submit":
            continue
        if "\n" in cmd or "`" in cmd or i + 1 >= len(messages) or messages[i + 1]["role"] != "tool":
            continue
        lines = [ln.strip() for ln in messages[i + 1]["content"].splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        if first.startswith(_BORING) or not 12 <= len(first) <= 200 or text.count(first) != 1:
            continue
        out.append({"cmd": cmd, "expected": first, "depth": round(depth, 3)})
    return out


def build_quality_dataset(model: str, tokens: int, count: int, out: Path, seed: int = 42,
                          shards: int = 4, contexts: int = 200,
                          echo: Callable[[str], None] = print) -> str:
    """`count` recall + `count` drift items from the first contexts of the perf dataset."""
    rng = random.Random(seed + 1)
    items: list[dict] = []
    recall = drift = 0
    for n, (renderer, tool, text, msgs) in enumerate(
            fitted_contexts(model, tokens, contexts, seed, shards, echo)):
        if drift < count:
            items.append({"id": f"drift-{n}", "kind": "drift", "context": n,
                          "prompt": renderer.render(msgs, [tool], enable_thinking=False)})
            drift += 1
        cands = recall_candidates(msgs, text)
        if recall < count and cands:
            c = rng.choice(cands)
            q = msgs + [{"role": "user", "content": QUESTION.format(cmd=c["cmd"])}]
            items.append({"id": f"recall-{n}", "kind": "recall", "context": n,
                          "prompt": renderer.render(q, [tool], enable_thinking=False),
                          "expected": c["expected"], "cmd": c["cmd"], "depth": c["depth"]})
            recall += 1
        if recall >= count and drift >= count:
            break
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    digest = sha256_file(out)
    echo(f"wrote {recall} recall + {drift} drift items to {out} (sha256 {digest})")
    return digest


# --- running -------------------------------------------------------------------------------------

def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _complete(base_url: str, model: str, item: dict, max_tokens: int, logprobs: int | None,
              timeout: float = 900) -> dict:
    body = {"model": model, "prompt": item["prompt"], "max_tokens": max_tokens,
            "temperature": 0, "seed": 0}
    if logprobs:
        body["logprobs"] = logprobs
    r = httpx.post(f"{base_url}/v1/completions", json=body, timeout=timeout)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    out = {"id": item["id"], "kind": item["kind"], "text": choice["text"],
           "finish_reason": choice.get("finish_reason")}
    lp = choice.get("logprobs") or {}
    if lp:
        out["tokens"] = lp.get("tokens")
        out["top_logprobs"] = lp.get("top_logprobs")
    if item["kind"] == "recall":
        out["expected"] = item["expected"]
    return out


def score_recall(expected: str, answer: str) -> tuple[bool, bool]:
    """(contains, exact first line), whitespace-normalised."""
    lines = [ln for ln in answer.strip().splitlines() if ln.strip()]
    first = _norm(lines[0]) if lines else ""
    return _norm(expected) in _norm(answer), first == _norm(expected)


_TOOL_CALL = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM = re.compile(r"<parameter=([^>]+)>\n?(.*?)\n?</parameter>", re.DOTALL)


def parse_tool_call(text: str) -> tuple[str, dict[str, str]] | None:
    m = _TOOL_CALL.search(text)
    if not m:
        return None
    return m.group(1).strip(), {k.strip(): v.strip() for k, v in _PARAM.findall(m.group(2))}


def run_quality(base_url: str, model: str, workload: str, dataset: Path, run_dir: Path,
                concurrency: int = 1, recall_max_tokens: int = 96,
                drift_max_tokens: int = 256, echo: Callable[[str], None] = print,
                limit: int | None = None) -> QualityResult:
    items = load_items(dataset)
    if limit:
        items = ([i for i in items if i["kind"] == "recall"][:limit]
                 + [i for i in items if i["kind"] == "drift"][:limit])
    out = run_dir / "quality" / f"{workload}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    def one(item: dict) -> dict:
        if item["kind"] == "recall":
            return _complete(base_url, model, item, recall_max_tokens, None)
        return _complete(base_url, model, item, drift_max_tokens, 20)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        responses = list(pool.map(one, items))
    with out.open("w") as f:
        for r in responses:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    recall = [score_recall(r["expected"], r["text"]) for r in responses if r["kind"] == "recall"]
    drift = [r for r in responses if r["kind"] == "drift"]
    result = QualityResult(
        workload=workload, dataset_sha256=sha256_file(dataset),
        responses=str(out.relative_to(run_dir)),
        recall_n=len(recall),
        recall_accuracy=sum(c for c, _ in recall) / len(recall) if recall else None,
        recall_exact=sum(e for _, e in recall) / len(recall) if recall else None,
        drift_n=len(drift),
        drift_tool_call_rate=(sum(parse_tool_call(r["text"]) is not None for r in drift)
                              / len(drift)) if drift else None,
    )
    echo(f"quality {workload}: recall {result.recall_accuracy} (exact {result.recall_exact}) "
         f"over {result.recall_n}; tool calls in {result.drift_tool_call_rate} of {result.drift_n}")
    return result


# --- comparing an optimized build with the baseline ----------------------------------------------

@dataclass
class Thresholds:
    min_same_tool_call: float = 0.97  # of prompts where the baseline made a tool call
    max_mean_kl: float = 0.02  # nats, over positions before the first divergence
    min_top1_agreement: float = 0.98
    max_recall_drop: float = 0.02  # absolute


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the tokens both top-k lists share, each renormalised."""
    shared = set(p) & set(q)
    if not shared:
        return float("inf")
    zp = math.log(sum(math.exp(p[t]) for t in shared))
    zq = math.log(sum(math.exp(q[t]) for t in shared))
    return sum(math.exp(p[t] - zp) * ((p[t] - zp) - (q[t] - zq)) for t in shared)


def compare_responses(base: list[dict], cand: list[dict], t: Thresholds | None = None) -> dict:
    t = t or Thresholds()
    by_id = {r["id"]: r for r in cand}
    same_call = base_calls = 0
    kls: list[float] = []
    agree = positions = 0
    divergence: list[int] = []
    for b in base:
        c = by_id.get(b["id"])
        if c is None or b["kind"] != "drift":
            continue
        bc = parse_tool_call(b["text"])
        if bc is not None:
            base_calls += 1
            same_call += parse_tool_call(c["text"]) == bc
        bt, ct = b.get("tokens") or [], c.get("tokens") or []
        n = 0
        while n < min(len(bt), len(ct)) and bt[n] == ct[n]:
            n += 1
        divergence.append(n)
        # Distributions are comparable only while both saw the same prefix: positions <= n.
        for i in range(min(n + 1, len(bt), len(ct))):
            positions += 1
            agree += bt[i] == ct[i]
            bl, cl = (b.get("top_logprobs") or [])[i:i + 1], (c.get("top_logprobs") or [])[i:i + 1]
            if bl and cl and bl[0] and cl[0]:
                k = _kl(bl[0], cl[0])
                if math.isfinite(k):
                    kls.append(k)

    def recall_acc(rs: list[dict]) -> float | None:
        rec = [score_recall(r["expected"], r["text"])[0] for r in rs if r["kind"] == "recall"]
        return sum(rec) / len(rec) if rec else None

    report = {
        "same_tool_call": same_call / base_calls if base_calls else None,
        "baseline_tool_calls": base_calls,
        "mean_kl": sum(kls) / len(kls) if kls else None,
        "top1_agreement": agree / positions if positions else None,
        "median_divergence_token": sorted(divergence)[len(divergence) // 2] if divergence else None,
        "recall_baseline": recall_acc(base),
        "recall_candidate": recall_acc(cand),
    }
    checks = {
        "same_tool_call": report["same_tool_call"] is not None
        and report["same_tool_call"] >= t.min_same_tool_call,
        "mean_kl": report["mean_kl"] is not None and report["mean_kl"] <= t.max_mean_kl,
        "top1_agreement": report["top1_agreement"] is not None
        and report["top1_agreement"] >= t.min_top1_agreement,
        "recall": report["recall_baseline"] is None or (
            report["recall_candidate"] is not None
            and report["recall_baseline"] - report["recall_candidate"] <= t.max_recall_drop),
    }
    report["checks"] = checks
    report["pass"] = all(checks.values())
    return report
