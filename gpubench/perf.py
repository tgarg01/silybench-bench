"""Performance points via `vllm bench serve`, plus parsing into PerfPoint."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from gpubench.config import SLO, PerfConfig, ServingSession, Workload
from gpubench.prompts import stage_dataset
from gpubench.schema import ITL, Percentiles, PerfPoint
from gpubench.telemetry import GpuSampler, TelemetrySummary

LATENCY_METRICS = ("ttft", "tpot", "itl", "e2el")


def bench_serve_args(
    session: ServingSession,
    workload: Workload,
    concurrency: int,
    num_prompts: int,
    seed: int,
    result_filename: str,
    num_warmups: int,
    raw_dir: str = "/work/raw",
    datasets_dir: str = "/work/datasets",
) -> list[str]:
    """Arguments for `vllm bench serve`. Paths are as the load generator sees them
    (inside the container for the docker runtime, host paths for native)."""
    engine = session.config.engine
    args = [
        "vllm", "bench", "serve",
        "--backend", "vllm",
        "--model", session.served_model,
        "--base-url", f"http://localhost:{engine.port}",
        "--dataset-name", workload.dataset,
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency),
        "--request-rate", "inf",
        "--num-warmups", str(num_warmups),
        "--seed", str(seed),
        "--percentile-metrics", ",".join(LATENCY_METRICS),
        "--metric-percentiles", "95,99",
        "--save-result",
        "--result-dir", raw_dir,
        "--result-filename", result_filename,
        "--disable-tqdm",
    ]
    if workload.dataset == "random":
        args += [
            "--random-input-len", str(workload.input_len),
            "--random-output-len", str(workload.output_len),
            "--random-range-ratio", "0",
            # Force exactly output_len tokens so every run does identical work.
            "--ignore-eos",
        ]
    elif workload.dataset == "sharegpt":
        args += ["--dataset-path", workload.dataset_path or f"{datasets_dir}/sharegpt.json"]
    elif workload.dataset == "custom":
        args += [
            # Prompts are pre-rendered with the chat template and exactly input_len tokens long.
            "--dataset-path", f"{datasets_dir}/{Path(workload.dataset_path).name}",
            "--custom-output-len", str(workload.output_len),
            "--skip-chat-template",
            "--no-oversample",  # never resend a prompt (it would hit the prefix cache)
            "--ignore-eos",
        ]
    return args


def slo_pass(ttft_p99_ms: float, itl_median_ms: float, slo: SLO) -> bool:
    return ttft_p99_ms <= slo.ttft_p99_ms and itl_median_ms <= slo.itl_median_ms


def _median(raws: list[dict], key: str) -> float:
    return float(statistics.median(float(r[key]) for r in raws))


def build_perf_point(
    raws: list[dict],
    workload: Workload,
    concurrency: int,
    num_prompts: int,
    gpu_count: int,
    slo: SLO,
    telemetry: TelemetrySummary | None,
    price_per_hour_usd: float | None,
) -> PerfPoint:
    """Collapse repeated `vllm bench serve` result dicts into one PerfPoint (per-field median)."""

    def pct(metric: str) -> Percentiles:
        return Percentiles(
            p95=_median(raws, f"p95_{metric}_ms"), p99=_median(raws, f"p99_{metric}_ms")
        )

    itl = ITL(
        median=_median(raws, "median_itl_ms"),
        p95=_median(raws, "p95_itl_ms"),
        p99=_median(raws, "p99_itl_ms"),
    )
    ttft = pct("ttft")
    output_tps = _median(raws, "output_throughput")

    avg_power = telemetry.avg_power_w if telemetry else None
    tokens_per_joule = output_tps / avg_power if avg_power else None
    usd_per_1m = (
        price_per_hour_usd / (output_tps * 3600) * 1e6
        if price_per_hour_usd and output_tps > 0
        else None
    )

    def optional_median(key: str) -> float | None:
        vals = [float(r[key]) for r in raws if r.get(key) is not None]
        return float(statistics.median(vals)) if vals else None

    return PerfPoint(
        prefix_cache_hit_rate=optional_median("prefix_cache_hit_rate"),
        ttft_first_turn_p95_ms=optional_median("ttft_first_turn_p95_ms"),
        ttft_later_turns_p99_ms=optional_median("ttft_later_turns_p99_ms"),
        workload=workload.name,
        input_len=workload.input_len,
        output_len=workload.output_len,
        concurrency=concurrency,
        num_prompts=num_prompts,
        repeats=len(raws),
        completed=int(statistics.median(r["completed"] for r in raws)),
        failed=int(max(r.get("failed", 0) for r in raws)),
        duration_s=_median(raws, "duration"),
        ttft_ms=ttft,
        tpot_ms=pct("tpot"),
        itl_ms=itl,
        e2el_ms=pct("e2el"),
        request_throughput=_median(raws, "request_throughput"),
        output_throughput=output_tps,
        total_token_throughput=_median(raws, "total_token_throughput"),
        output_throughput_per_gpu=output_tps / gpu_count,
        tokens_per_s_per_user=1000.0 / itl.median if itl.median > 0 else 0.0,
        avg_power_w=avg_power,
        peak_memory_gb=telemetry.peak_memory_gb if telemetry else None,
        max_gpu_temp_c=telemetry.max_temp_c if telemetry else None,
        thermal_throttle_fraction=telemetry.thermal_throttle_fraction if telemetry else None,
        output_tokens_per_joule=tokens_per_joule,
        usd_per_1m_output_tokens=usd_per_1m,
        slo_pass=slo_pass(ttft.p99, itl.median, slo),
    )


def make_point_runner(
    session: ServingSession,
    exec_fn: Callable[[list[str]], None],
    work_dir: Path,
    raw_dir: str = "/work/raw",
    datasets_dir: str = "/work/datasets",
) -> Callable[[Workload, int], PerfPoint]:
    """Return measure(workload, concurrency) -> PerfPoint bound to a running server.

    `exec_fn` runs a command next to the server (VllmServer.exec); `raw_dir` is where that
    command writes results, which is always `work_dir / "raw"` on the host.
    """
    perf: PerfConfig = session.config.perf
    hw = session.hardware

    def measure(workload: Workload, concurrency: int) -> PerfPoint:
        if workload.dataset == "custom":
            stage_dataset(workload, work_dir / "datasets")
        num_prompts = perf.num_prompts(concurrency, workload)
        raws: list[dict] = []
        windows = []
        stem = f"{workload.name}_c{concurrency}"
        with GpuSampler(work_dir / "telemetry" / f"{stem}.csv") as sampler:
            for rep in range(perf.repeats_for(workload)):
                fname = f"{stem}_r{rep}.json"
                if workload.dataset == "sessions":
                    # Multi-turn agent sessions: a host-side client (vllm bench serve can't).
                    from gpubench.multiturn import run_sessions
                    from gpubench.prompts import ensure_dataset

                    run_sessions(
                        f"http://localhost:{session.config.engine.port}", session.served_model,
                        ensure_dataset(workload), concurrency, num_prompts,
                        workload.output_len, work_dir / "raw" / fname, seed=perf.seed + rep,
                        reset_cache=bool(session.extra_args().get("enable-prefix-caching")),
                    )
                else:
                    exec_fn(bench_serve_args(
                        session, workload, concurrency, num_prompts,
                        seed=perf.seed + rep, result_filename=fname,
                        # Warm up only before the first repeat.
                        num_warmups=perf.warmups_for(workload) if rep == 0 else 0,
                        raw_dir=raw_dir, datasets_dir=datasets_dir,
                    ))
                ended = datetime.now()  # same local clock as nvidia-smi timestamps
                raws.append(json.loads((work_dir / "raw" / fname).read_text()))
                # The measured interval is the last `duration` seconds before the client exits
                # (after tokenizer load and warmup), so average power over just that.
                windows.append((ended - timedelta(seconds=raws[-1]["duration"]), ended))
        return build_perf_point(
            raws, workload, concurrency, num_prompts,
            gpu_count=hw.gpu_count * hw.node_count,
            slo=perf.slo_for(workload),
            telemetry=sampler.summary(windows),
            price_per_hour_usd=hw.price_per_hour_usd,
        )

    return measure
