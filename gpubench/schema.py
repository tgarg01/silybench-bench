"""Result schema: one RunResult JSON per serving session.

This is the contract between the GPU runner, the aggregator and the website.
Hardware and parallelism fields are here from day one so multi-GPU / multi-node
results drop in without a schema change. Bump SCHEMA_VERSION on breaking changes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from gpubench.config import SLO, Hardware, Parallelism, Precision
from gpubench.quality import QualityResult

# v2: hardware.provider (was `cloud`), software.runtime, RunResult.complete.
# v3: experiment, fingerprint, raw_assets, profiles, model.checkpoint, capacity.skipped_levels.
# Older files still load (every new field has a default).
SCHEMA_VERSION = 3


class Percentiles(BaseModel):
    """Latency percentiles in ms. We report p95 and p99 only."""

    p95: float
    p99: float


class ITL(Percentiles):
    """Inter-token latency. The median is reported too (the headline 'IDT' metric)."""

    median: float


class PerfPoint(BaseModel):
    """One (workload, concurrency) measurement: the median across repeats."""

    workload: str
    input_len: int | None
    output_len: int | None
    concurrency: int
    num_prompts: int
    repeats: int
    completed: int
    failed: int
    duration_s: float

    ttft_ms: Percentiles
    tpot_ms: Percentiles
    itl_ms: ITL
    e2el_ms: Percentiles

    request_throughput: float  # req/s
    output_throughput: float  # output tok/s
    total_token_throughput: float  # (input+output) tok/s
    output_throughput_per_gpu: float
    tokens_per_s_per_user: float  # 1000 / median ITL

    avg_power_w: float | None = None
    peak_memory_gb: float | None = None
    output_tokens_per_joule: float | None = None
    # Multi-turn session workloads (gpubench.multiturn) only.
    prefix_cache_hit_rate: float | None = None
    ttft_first_turn_p95_ms: float | None = None
    ttft_later_turns_p99_ms: float | None = None
    max_gpu_temp_c: float | None = None
    thermal_throttle_fraction: float | None = None  # 0 = never slowed down for heat
    usd_per_1m_output_tokens: float | None = None

    slo_pass: bool


class CapacityResult(BaseModel):
    """How many users one deployment can serve simultaneously for a workload."""

    workload: str
    slo: SLO
    # Highest concurrency meeting every SLO limit (0 = even 1 user fails the SLO).
    max_users_slo: int
    # How many requests of this workload's size fit in the KV cache at once.
    max_users_kv_cache: int | None
    output_throughput_at_max_users: float | None
    # Concurrency levels actually measured during the search, for transparency.
    probed: dict[int, bool]
    # Sweep levels not run because they exceed the KV-cache limit several times over
    # (requests would only queue); see PerfConfig.max_kv_multiple.
    skipped_levels: list[int] = []


class AccuracyResult(BaseModel):
    task: str
    metric: str
    value: float
    stderr: float | None = None
    num_fewshot: int | None = None
    limit: int | None = None
    num_samples: int | None = None


class ModelInfo(BaseModel):
    hf_id: str  # base model
    checkpoint: str | None = None  # repo actually served when it differs (e.g. an FP8 checkpoint)
    variant: str | None = None  # named serving configuration (config ModelSpec.variants)
    revision: str  # resolved commit sha of the served repo when available
    precision: Precision
    max_model_len: int
    thinking: bool


class SoftwareInfo(BaseModel):
    engine: Literal["vllm"] = "vllm"
    engine_version: str | None = None
    engine_image: str
    engine_image_digest: str | None = None
    # How vLLM ran: "docker" (engine_image) or "native" (pip vllm==engine_version in a venv).
    runtime: Literal["docker", "native", "mock"] | None = None
    # Packages added to the load generator (not the server), e.g. pandas for custom datasets.
    bench_extras: list[str] = []
    lm_eval_version: str | None = None
    gpubench_version: str
    nvidia_driver: str | None = None
    cuda_version: str | None = None


class RawAsset(BaseModel):
    """A published file with everything captured by a run (GitHub release asset)."""

    name: str
    url: str
    sha256: str
    bytes: int
    contents: str  # e.g. "full run directory incl. per-question accuracy samples"


class Profile(BaseModel):
    """An Nsight profile of one load point. `summary` is tool-specific, small JSON."""

    tool: Literal["nsys", "ncu"]
    workload: str
    concurrency: int
    summary: dict = {}
    asset: str | None = None  # name of the RawAsset holding the .nsys-rep / .ncu-rep


class RunResult(BaseModel):
    schema_version: int = SCHEMA_VERSION
    run_id: str
    campaign: str  # config name
    created_at: datetime
    git_commit: str | None  # silybench-bench commit that produced this run
    config_hash: str
    sample: bool = False  # True = synthetic placeholder data, never real measurements
    experiment: str | None = None  # published experiment id (silybench-data experiments/)
    # False while the run is in progress (or if it died); `--resume` continues such runs.
    complete: bool = True

    hardware: Hardware
    parallelism: Parallelism
    software: SoftwareInfo
    model: ModelInfo
    serving_args: list[str]
    kv_cache_tokens: int | None = None
    # Hardware fingerprint of the machine (gpubench.fingerprint); the reference an
    # experiment's reproductions are checked against.
    fingerprint: dict | None = None
    raw_assets: list[RawAsset] = []
    profiles: list[Profile] = []
    # Correctness of custom long-context scenarios (gpubench.quality); the baseline that
    # optimized builds are compared against.
    quality: list[QualityResult] = []

    # Set by the aggregator when separate runs of the same setup (e.g. a perf run plus an
    # accuracy-only re-run) are combined into this one entry.
    merged_from: list[str] = []

    perf: list[PerfPoint] = []
    capacity: list[CapacityResult] = []
    accuracy: list[AccuracyResult] = []
