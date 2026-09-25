"""Rough time and cost estimate for a campaign, shown before anyone pays for GPU hours.

A deliberately simple decode-bound model: per-token latency = weight bytes / effective memory
bandwidth, growing with the number of active sequences and their context. Calibrated on the
Qwen3-8B H100 runs: predicts 7.8 h of perf vs 7.9 h measured for BF16, 5.6 h vs 6.7 h for FP8.
Treat it as +/-50%: good enough to say "about 4 hours, about $10", not for billing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from gpubench.config import BenchConfig, ServingSession, Workload

# Memory bandwidth in TB/s by GPU family (prefix of the normalised gpu_type).
BANDWIDTH_TBS = {
    "B300": 8.0, "B200": 8.0, "GB200": 8.0, "H200": 4.8, "GH200": 4.0, "H100-NVL": 3.9,
    "H100-PCIe": 2.0, "H100": 3.35, "H800": 3.35, "A100-PCIe": 1.9, "A100": 2.0,
    "L40S": 0.86, "L40": 0.86, "L4": 0.3, "A10G": 0.6, "A10": 0.6, "A6000": 0.77,
    "RTX6000-Ada": 0.96, "RTXPRO6000": 1.8, "RTX5090": 1.8, "RTX4090": 1.0, "RTX3090": 0.94,
}
# Effective bytes read per parameter per decode step. FP8 is 1.35, not 1.0: activations,
# attention and the KV cache stay 16-bit (measured: 4.6 ms vs 6.8 ms ITL for Qwen3-8B).
BYTES_PER_PARAM = {"bf16": 2.0, "fp16": 2.0, "fp8": 1.35}
# Measured KV-cache tokens per GB of free memory is ~model-specific; this is Qwen3-8B BF16's.
KV_TOKENS_PER_GB = 6_500
EFFICIENCY = 0.70  # fraction of peak bandwidth a decode step achieves
CONTEXT_SCALE = 140_000  # active*context tokens at which per-token latency doubles
OVERHEAD_S_PER_REPEAT = 15  # client start, tokenizer load
STARTUP_S = 300  # model download (cached after the first session) + vLLM load + CUDA graphs
# Wall-clock minutes per accuracy task for an ~8B model on one H100 (scaled by speed below).
ACCURACY_MINUTES = {
    "mmlu_pro": 55, "gpqa_diamond_cot_zeroshot": 8, "gsm8k": 6, "minerva_math500": 12,
    "ifeval": 6, "arc_challenge_chat": 5,
}
DEFAULT_ACCURACY_MINUTES = 15
PROBES_PER_WORKLOAD = 3


def model_params_b(hf_id: str) -> float:
    """Parameter count in billions from the model name ("Qwen3-8B" -> 8, "A3B" MoE -> active)."""
    name = hf_id.split("/")[-1]
    active = re.search(r"A(\d+(?:\.\d+)?)B", name)
    if active:
        return float(active.group(1))
    total = re.findall(r"(\d+(?:\.\d+)?)B", name, re.IGNORECASE)
    return float(total[-1]) if total else 8.0


def bandwidth_tbs(gpu_type: str) -> float:
    for prefix in sorted(BANDWIDTH_TBS, key=len, reverse=True):
        if gpu_type.startswith(prefix):
            return BANDWIDTH_TBS[prefix]
    return 2.0


@dataclass
class Estimate:
    sessions: int
    perf_points: int
    perf_hours: float
    accuracy_hours: float
    startup_hours: float

    @property
    def hours(self) -> float:
        return self.perf_hours + self.accuracy_hours + self.startup_hours

    def cost(self, price_per_hour: float | None) -> float | None:
        return self.hours * price_per_hour if price_per_hour else None


def _point_seconds(session: ServingSession, wl: Workload, c: int, base_itl_s: float,
                   kv_users: int) -> float:
    perf = session.config.perf
    in_len, out_len = wl.input_len or 512, wl.output_len or 256
    active = max(1, min(c, kv_users))
    context = in_len + out_len / 2
    itl = base_itl_s * (1 + active * context / CONTEXT_SCALE) * (1 + active / 400)
    prompts = perf.num_prompts(c)
    decode = prompts * out_len * itl / active
    return perf.repeats * (decode + OVERHEAD_S_PER_REPEAT)


def estimate(cfg: BenchConfig) -> Estimate:
    hw = cfg.hardware
    gpu_type = hw.gpu_type if hw else "H100-80GB"
    gpus = (hw.gpu_count * hw.node_count) if hw else 1
    mem_gb = float(re.search(r"(\d+)GB", gpu_type).group(1)) if re.search(r"(\d+)GB", gpu_type) \
        else 80.0
    bw = bandwidth_tbs(gpu_type) * 1e12 * EFFICIENCY * gpus

    perf_s = acc_s = 0.0
    points = 0
    sessions = cfg.sessions()
    for s in sessions:
        params = model_params_b(s.model.hf_id) * 1e9
        weight_bytes = params * BYTES_PER_PARAM[s.precision]
        base_itl = weight_bytes / bw
        free_gb = max(1.0, mem_gb * gpus * s.model.gpu_memory_utilization - weight_bytes / 1e9)
        # FP8 weights leave more room, and 8B-class models' KV per token scales with size.
        kv_tokens = free_gb * KV_TOKENS_PER_GB * (8e9 / params) ** 0.5
        speed = 6.8e-3 / base_itl  # relative to Qwen3-8B BF16 on one H100
        for wl in cfg.perf.workloads:
            per_req = (wl.input_len or 512) + (wl.output_len or 256)
            kv_users = max(1, int(kv_tokens // per_req))
            levels = list(cfg.perf.concurrency)
            if cfg.perf.capacity.enabled:
                # Probes land near the capacity limit; approximate them at the KV limit.
                levels += [min(kv_users, max(levels))] * PROBES_PER_WORKLOAD
            for c in levels:
                perf_s += _point_seconds(s, wl, c, base_itl, kv_users)
                points += 1
        if cfg.accuracy.enabled:
            for t in cfg.accuracy.tasks:
                minutes = ACCURACY_MINUTES.get(t.name, DEFAULT_ACCURACY_MINUTES)
                if t.limit:
                    minutes = max(2.0, minutes * min(1.0, t.limit / 1000))
                acc_s += minutes * 60 / max(0.25, speed)
    return Estimate(
        sessions=len(sessions),
        perf_points=points,
        perf_hours=perf_s / 3600,
        accuracy_hours=acc_s / 3600,
        startup_hours=len(sessions) * STARTUP_S / 3600,
    )


def format_estimate(est: Estimate, price_per_hour: float | None) -> str:
    lines = [
        f"sessions: {est.sessions}  perf points (incl. capacity probes): {est.perf_points}",
        f"perf ~{est.perf_hours:.1f} h, accuracy ~{est.accuracy_hours:.1f} h, "
        f"startup ~{est.startup_hours:.1f} h",
        f"TOTAL ~{est.hours:.1f} h (rough, +/-50%; "
        f"range {math.floor(est.hours * 0.6 * 10) / 10}-{math.ceil(est.hours * 1.5 * 10) / 10} h)",
    ]
    cost = est.cost(price_per_hour)
    if cost is not None:
        lines.append(f"GPU cost ~${cost:.0f} at ${price_per_hour}/h "
                     f"(range ${cost * 0.6:.0f}-${cost * 1.5:.0f})")
    else:
        lines.append("GPU cost: pass --price-per-hour to estimate")
    return "\n".join(lines)
