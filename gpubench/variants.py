"""Compare serving variants of one workload (e.g. FP8 optimizations) against a baseline.

`gpubench variants report <run dirs> --baseline base` prints a markdown report: capacity within
the SLO, the p99 first-token latency curve, prefill speed, requests/hour and $ per 1k requests at
given GPU prices, the quality comparison against the baseline, and the nsys breakdown if
profiled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gpubench.quality import compare_responses, load_items
from gpubench.schema import RunResult


@dataclass
class VariantRun:
    label: str
    run_dir: Path
    result: RunResult


def load(run_dirs: list[Path]) -> list[VariantRun]:
    runs = []
    for d in run_dirs:
        r = RunResult.model_validate_json((d / "result.json").read_text())
        runs.append(VariantRun(r.model.variant or r.model.precision, d, r))
    return runs


def _points(r: RunResult, workload: str) -> dict[int, object]:
    return {p.concurrency: p for p in r.perf if p.workload == workload}


def summarize(v: VariantRun, workload: str, prices: dict[str, float]) -> dict:
    pts = _points(v.result, workload)
    cap = next((c for c in v.result.capacity if c.workload == workload), None)
    users = cap.max_users_slo if cap else 0
    one = pts.get(1)
    at_cap = pts.get(users)
    req_h = at_cap.request_throughput * 3600 if at_cap else None
    return {
        "label": v.label,
        "users": users,
        "ttft_p99_s": {c: p.ttft_ms.p99 / 1000 for c, p in sorted(pts.items())},
        "itl_median_ms": {c: p.itl_ms.median for c, p in sorted(pts.items())},
        "prefill_tok_s": (one.input_len / (one.ttft_ms.p95 / 1000)) if one and one.input_len
        else None,
        "req_per_h": req_h,
        "usd_per_1k_req": {k: (usd / req_h * 1000 if req_h else None)
                           for k, usd in prices.items()},
        "slo_ttft_s": cap.slo.ttft_p99_ms / 1000 if cap else None,
        "failed": not pts,
    }


def quality_vs(base: VariantRun, other: VariantRun, workload: str) -> dict | None:
    fb = base.run_dir / "quality" / f"{workload}.jsonl"
    fo = other.run_dir / "quality" / f"{workload}.jsonl"
    if not (fb.exists() and fo.exists()):
        return None
    return compare_responses(load_items(fb), load_items(fo))


def report(run_dirs: list[Path], baseline: str, workload: str,
           prices: dict[str, float]) -> str:
    runs = load(run_dirs)
    base = next((v for v in runs if v.label == baseline), None)
    if base is None:
        raise ValueError(f"no run with variant {baseline!r} among {[v.label for v in runs]}")
    rows = [summarize(v, workload, prices) for v in runs]
    levels = sorted({c for r in rows for c in r["ttft_p99_s"]})
    b = next(r for r in rows if r["label"] == baseline)
    lines = [f"# Variants of `{workload}` vs `{baseline}`", ""]
    slo = b["slo_ttft_s"]
    lines.append(f"Target: p99 first token ≤ {slo:g} s and median inter-token ≤ 50 ms. "
                 "Capacity = most users meeting both.")
    lines.append("")
    price_cols = [f"$/1k req ({k})" for k in prices]
    head = ["variant", "users", "vs base", "prefill tok/s", "req/h", *price_cols,
            "same tool call", "KL", "recall", "quality"]
    lines += ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for v, r in zip(runs, rows, strict=True):
        q = quality_vs(base, v, workload) if v is not base else None
        fmt = lambda x, d=0: "–" if x is None else f"{x:,.{d}f}"  # noqa: E731
        quality = ("baseline" if v is base else "–" if q is None
                   else ("PASS" if q["pass"] else "FAIL"))
        lines.append("| " + " | ".join([
            v.label, str(r["users"]) if not r["failed"] else "failed",
            "–" if v is base else f"{r['users'] - b['users']:+d}",
            fmt(r["prefill_tok_s"]), fmt(r["req_per_h"]),
            *[fmt(x, 3) for x in r["usd_per_1k_req"].values()],
            "–" if not q or q["same_tool_call"] is None else f"{q['same_tool_call']:.0%}",
            "–" if not q or q["mean_kl"] is None else f"{q['mean_kl']:.4f}",
            "–" if not q or q["recall_candidate"] is None else f"{q['recall_candidate']:.0%}",
            quality,
        ]) + " |")
    lines += ["", "p99 time to first token (s) by number of users:", ""]
    lines += ["| variant | " + " | ".join(str(c) for c in levels) + " |",
              "|---|" + "---|" * len(levels)]
    for r in rows:
        cells = []
        for c in levels:
            t = r["ttft_p99_s"].get(c)
            cells.append("–" if t is None else (f"**{t:.1f}**" if slo and t <= slo else f"{t:.1f}"))
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    lines += ["", "median inter-token latency (ms):", ""]
    lines += ["| variant | " + " | ".join(str(c) for c in levels) + " |",
              "|---|" + "---|" * len(levels)]
    for r in rows:
        cells = ["–" if r["itl_median_ms"].get(c) is None else f"{r['itl_median_ms'][c]:.1f}"
                 for c in levels]
        lines.append(f"| {r['label']} | " + " | ".join(cells) + " |")
    profiled = [(v, p) for v in runs for p in v.result.profiles if p.workload == workload]
    if profiled:
        lines += ["", "Where GPU time goes (nsys, one request):", ""]
        for v, p in profiled:
            groups = ", ".join(f"{g['group']} {g['pct']:.0%}" for g in p.summary.get("groups", []))
            idle = p.summary.get("gpu_idle_pct")
            lines.append(f"- **{v.label}** (answer {p.summary.get('output_len')} tokens): {groups}"
                         + (f"; GPU idle {idle:.0%}" if idle is not None else ""))
    return "\n".join(lines) + "\n"
