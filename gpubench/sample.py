"""Synthetic placeholder results so the website can be built before real GPU runs.

Every run is marked `sample=True`; the site shows a banner for them and the
aggregator drops them as soon as any real run exists. The numbers come from a
toy queueing model and are NOT measurements.
"""

from __future__ import annotations

from datetime import UTC, datetime

from gpubench import __version__
from gpubench.capacity import kv_cache_max_users
from gpubench.config import SLO, Hardware, Parallelism, Workload, load_config
from gpubench.perf import slo_pass
from gpubench.schema import (
    ITL,
    AccuracyResult,
    CapacityResult,
    ModelInfo,
    Percentiles,
    PerfPoint,
    RunResult,
    SoftwareInfo,
)

# (model, precision) -> (base ITL ms, ITL growth per user ms, prefill ms per 1k tokens, KV tokens)
TOY: dict[tuple[str, str], tuple[float, float, float, int]] = {
    ("Qwen/Qwen3-8B", "bf16"): (6.5, 0.10, 45.0, 450_000),
    ("Qwen/Qwen3-8B", "fp8"): (4.8, 0.075, 30.0, 560_000),
    ("Qwen/Qwen3-14B", "bf16"): (10.5, 0.16, 80.0, 330_000),
    ("Qwen/Qwen3-14B", "fp8"): (7.2, 0.11, 52.0, 440_000),
}
TOY_ACCURACY = {
    "Qwen/Qwen3-8B": {"mmlu_pro": 0.56, "gpqa_diamond_cot_zeroshot": 0.39, "gsm8k": 0.89,
                      "minerva_math500": 0.62, "ifeval": 0.83, "arc_challenge_chat": 0.90},
    "Qwen/Qwen3-14B": {"mmlu_pro": 0.61, "gpqa_diamond_cot_zeroshot": 0.44, "gsm8k": 0.92,
                       "minerva_math500": 0.67, "ifeval": 0.85, "arc_challenge_chat": 0.93},
}
FP8_ACCURACY_DELTA = -0.006


def _toy_point(key, wl: Workload, c: int, slo: SLO, kv_tokens: int) -> PerfPoint:
    base_itl, growth, prefill_per_k, _ = TOY[key]
    active = min(c, kv_tokens // wl.tokens_per_request)
    itl_med = base_itl + growth * active * (1 + wl.input_len / 4096)
    prefill = prefill_per_k * wl.input_len / 1000 * (1 + active / 32)
    # Requests beyond KV capacity wait for a running request to finish first.
    queued = max(0, c - active)
    ttft_med = prefill + queued / active * (prefill + itl_med * wl.output_len)
    ttft = Percentiles(p95=ttft_med * 1.6, p99=ttft_med * 2.2)
    itl = ITL(median=itl_med, p95=itl_med * 1.35, p99=itl_med * 1.9)
    tpot = Percentiles(p95=itl_med * 1.1, p99=itl_med * 1.25)
    e2e = ttft_med + itl_med * wl.output_len
    e2el = Percentiles(p95=e2e * 1.15, p99=e2e * 1.3)
    req_s = active / (e2e / 1000)
    out_tps = req_s * wl.output_len
    power = 350 + 350 * min(1.0, active / 64)
    return PerfPoint(
        workload=wl.name, input_len=wl.input_len, output_len=wl.output_len,
        concurrency=c, num_prompts=max(50, 5 * c), repeats=3, completed=max(50, 5 * c),
        failed=0, duration_s=max(50, 5 * c) / req_s,
        ttft_ms=ttft, tpot_ms=tpot, itl_ms=itl, e2el_ms=e2el,
        request_throughput=req_s, output_throughput=out_tps,
        total_token_throughput=req_s * (wl.input_len + wl.output_len),
        output_throughput_per_gpu=out_tps, tokens_per_s_per_user=1000 / itl_med,
        avg_power_w=power, peak_memory_gb=74.0, output_tokens_per_joule=out_tps / power,
        usd_per_1m_output_tokens=3.0 / (out_tps * 3600) * 1e6,
        slo_pass=slo_pass(ttft.p99, itl.median, slo),
    )


def sample_runs() -> list[RunResult]:
    from gpubench.cli import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "qwen3-8b.yaml")
    slo = cfg.perf.slo
    runs = []
    for (hf_id, precision), (*_, kv_tokens) in TOY.items():
        key = (hf_id, precision)
        perf, capacity = [], []
        for wl in cfg.perf.workloads:
            fine = sorted(set(cfg.perf.concurrency) | set(range(1, 513)))
            points = {c: _toy_point(key, wl, c, slo, kv_tokens) for c in fine}
            passing = [c for c in fine if points[c].slo_pass]
            max_users = max(passing, default=0)
            # Like a real run: the sweep levels plus the level the capacity search lands on.
            measured = sorted(set(cfg.perf.concurrency) | ({max_users} if max_users else set()))
            perf += [points[c] for c in measured]
            capacity.append(CapacityResult(
                workload=wl.name, slo=slo, max_users_slo=max_users,
                max_users_kv_cache=kv_cache_max_users(kv_tokens, wl),
                output_throughput_at_max_users=points[max_users].output_throughput
                if max_users else None,
                probed={c: points[c].slo_pass for c in cfg.perf.concurrency},
            ))
        delta = FP8_ACCURACY_DELTA if precision == "fp8" else 0.0
        accuracy = [
            AccuracyResult(task=t, metric="exact_match", value=v + delta, stderr=0.01)
            for t, v in TOY_ACCURACY[hf_id].items()
        ]
        slug = hf_id.split("/")[-1].lower()
        runs.append(RunResult(
            run_id=f"sample_h100-80gbx1_{slug}_{precision}",
            campaign=slug, created_at=datetime(2026, 9, 24, tzinfo=UTC),
            git_commit=None, config_hash="sample", sample=True,
            hardware=Hardware(gpu_type="H100-80GB", provider="sample", machine_type="a3-highgpu-1g",
                              zone="us-central1-a", price_per_hour_usd=3.0),
            parallelism=Parallelism(),
            software=SoftwareInfo(engine_version="0.30.0",
                                  engine_image="vllm/vllm-openai:v0.30.0",
                                  gpubench_version=__version__),
            model=ModelInfo(hf_id=hf_id, revision="main", precision=precision,
                            max_model_len=16384, thinking=False),
            serving_args=[], kv_cache_tokens=kv_tokens,
            perf=perf, capacity=capacity, accuracy=accuracy,
        ))
    return runs

