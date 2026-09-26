"""Benchmark campaign config (YAML) and its expansion into serving sessions.

A campaign config says *what* to measure (models, workloads, SLO, accuracy tasks). It is
hardware-agnostic: the machine it runs on is detected at run time (`gpubench.hardware`) and
the provider and price come from CLI flags. A config may still pin a `hardware` block (the GCP
Terraform path does). Each (model, precision) pair becomes a ServingSession: one vLLM server
lifetime in which we run the perf sweep, the capacity search and the accuracy suite.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

Precision = Literal["bf16", "fp16", "fp8"]


class Hardware(BaseModel):
    gpu_type: str  # normalised, e.g. "H100-80GB" (SXM), "H100-PCIe-80GB", "A100-80GB"
    gpu_count: int = 1
    node_count: int = 1
    # Where the machine was rented: "runpod", "lambda", "vast", "gcp", "aws", ...
    provider: str = "unknown"
    machine_type: str | None = None  # provider's instance name, e.g. "a3-highgpu-1g"
    zone: str | None = None  # zone / region / datacenter when known
    provisioning: Literal["spot", "on-demand", "reserved", "flex-start"] = "on-demand"
    # Hourly price for the whole machine; used for $/1M tokens. None = don't compute cost.
    price_per_hour_usd: float | None = None
    # Deprecated alias of `provider` (schema v1 results); kept so old result.json files load.
    cloud: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _provider_from_cloud(cls, data: object) -> object:
        if isinstance(data, dict) and "provider" not in data and data.get("cloud"):
            data = {**data, "provider": data["cloud"]}
        return data


class Parallelism(BaseModel):
    tp: int = 1  # tensor parallel
    pp: int = 1  # pipeline parallel
    dp: int = 1  # data parallel
    ep: int = 1  # expert parallel (MoE)

    def vllm_args(self) -> list[str]:
        args = ["--tensor-parallel-size", str(self.tp), "--pipeline-parallel-size", str(self.pp)]
        if self.dp > 1:
            args += ["--data-parallel-size", str(self.dp)]
        if self.ep > 1:
            args += ["--enable-expert-parallel"]
        return args


class Engine(BaseModel):
    name: Literal["vllm"] = "vllm"
    # Docker runtime: this image. Native runtime: `pip install vllm==<version from the tag>`,
    # so both runtimes serve with the same vLLM release.
    image: str = "vllm/vllm-openai:v0.30.0"
    port: int = 8000
    startup_timeout_s: int = 1800

    @property
    def version(self) -> str:
        """vLLM version pinned by the image tag ("vllm/vllm-openai:v0.30.0" -> "0.30.0")."""
        tag = self.image.rpartition(":")[2]
        if not re.fullmatch(r"v?\d+\.\d+(\.\d+)?(\.post\d+)?", tag):
            raise ValueError(f"engine.image {self.image!r} must be tagged with a vLLM version")
        return tag.removeprefix("v")


class ModelSpec(BaseModel):
    hf_id: str  # the base model; results, API prices and comparisons are keyed by it
    revision: str = "main"
    precisions: list[Precision] = ["bf16"]
    # Pre-quantized checkpoints served instead of quantizing hf_id on load,
    # e.g. {fp8: Qwen/Qwen3.8-27B-FP8} (how API providers typically serve FP8).
    checkpoints: dict[Precision, str] = Field(default_factory=dict)
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    # Extra vLLM flags, e.g. {"reasoning-parser": "qwen3"}. Values may be str/int/dict.
    serving_args: dict[str, object] = Field(default_factory=dict)
    # Recorded in results so thinking vs. non-thinking runs are never mixed up.
    thinking: bool = False

    @property
    def slug(self) -> str:
        return self.hf_id.split("/")[-1].lower()

    def checkpoint(self, precision: str) -> str:
        """The HF repo actually served for this precision."""
        return self.checkpoints.get(precision, self.hf_id)  # type: ignore[call-overload]


class SLO(BaseModel):
    """A concurrency level 'passes' when every limit holds. Drives max concurrent users."""

    ttft_p99_ms: float = 2000.0
    itl_median_ms: float = 50.0  # 50 ms => >= 20 tok/s per user


class QualitySuite(BaseModel):
    """Correctness checks run after a workload's perf (gpubench.quality): long-context recall
    questions + a drift baseline, from the same contexts as the perf prompts."""

    dataset_path: str
    dataset_url: str | None = None
    sha256: str
    recall_max_tokens: int = 96
    drift_max_tokens: int = 256
    limit: int | None = None  # items of each kind (smoke tests); None = all


class Workload(BaseModel):
    name: str
    # random: synthetic tokens of exactly input_len. custom: a JSONL of pre-rendered prompts
    # ({"prompt": ...}, chat template already applied) of input_len tokens each.
    dataset: Literal["random", "sharegpt", "custom"] = "random"
    input_len: int | None = None
    output_len: int | None = None
    dataset_path: str | None = None  # sharegpt / custom: local path (relative to the repo)
    dataset_url: str | None = None  # custom: downloaded to dataset_path if missing
    sha256: str | None = None  # custom: the file must hash to this (byte-identical prompts)
    # Per-workload overrides of the campaign defaults (e.g. a 100k-token scenario can't meet
    # a 2 s first-token target and can't run 512 users).
    slo: SLO | None = None
    concurrency: list[int] | None = None
    min_prompts: int | None = None
    repeats: int | None = None
    num_warmups: int | None = None
    quality: QualitySuite | None = None

    @model_validator(mode="after")
    def _check(self) -> Workload:
        if self.dataset in ("random", "custom") and (
            self.input_len is None or self.output_len is None
        ):
            raise ValueError(f"workload {self.name}: {self.dataset} needs input_len and output_len")
        if self.dataset == "custom" and not (self.dataset_path and self.sha256):
            raise ValueError(f"workload {self.name}: custom dataset needs dataset_path and sha256")
        return self

    @property
    def tokens_per_request(self) -> int | None:
        """ISL+OSL, used to turn KV-cache token capacity into a max-concurrency figure."""
        if self.input_len is None or self.output_len is None:
            return None
        return self.input_len + self.output_len


class CapacitySearch(BaseModel):
    enabled: bool = True
    # Stop bisecting when the pass/fail bracket is this narrow (in users).
    resolution: int = 4
    max_iterations: int = 6
    # Upper bound for probing past the sweep when its top level still passes
    # (used when the KV-cache limit is unknown; otherwise 2x that limit).
    max_users: int = 4096


class PerfConfig(BaseModel):
    workloads: list[Workload]
    enabled: bool = True  # False = only the quality suites (e.g. re-checking an optimization)
    concurrency: list[int] = [1, 4, 16, 32, 64, 128, 256, 512]
    # num_prompts = max(min_prompts, prompts_per_user * concurrency)
    prompts_per_user: int = 5
    min_prompts: int = 50
    num_warmups: int = 10
    repeats: int = 3
    seed: int = 42
    slo: SLO = SLO()
    capacity: CapacitySearch = CapacitySearch()

    # Skip sweep levels above this multiple of the KV-cache limit: past it requests only
    # queue, which takes hours and says nothing new. None = run every level.
    max_kv_multiple: float | None = 2.0

    def num_prompts(self, concurrency: int, workload: Workload | None = None) -> int:
        floor = workload.min_prompts if workload and workload.min_prompts else self.min_prompts
        return max(floor, self.prompts_per_user * concurrency)

    def slo_for(self, workload: Workload) -> SLO:
        return workload.slo or self.slo

    def concurrency_for(self, workload: Workload) -> list[int]:
        return workload.concurrency or self.concurrency

    def repeats_for(self, workload: Workload) -> int:
        return workload.repeats or self.repeats

    def warmups_for(self, workload: Workload) -> int:
        return self.num_warmups if workload.num_warmups is None else workload.num_warmups

    def levels(self, workload: Workload, kv_users: int | None) -> tuple[list[int], list[int]]:
        """(levels to measure, levels skipped) given how many requests fit in the KV cache."""
        levels = sorted(self.concurrency_for(workload))
        if not kv_users or self.max_kv_multiple is None:
            return levels, []
        limit = max(levels[0], int(self.max_kv_multiple * kv_users))
        return [c for c in levels if c <= limit], [c for c in levels if c > limit]


class AccuracyTask(BaseModel):
    name: str  # lm-eval task name
    num_fewshot: int | None = None
    limit: int | None = None  # cap examples (smoke tests)


class AccuracyConfig(BaseModel):
    enabled: bool = True
    tasks: list[AccuracyTask] = []
    num_concurrent: int = 64
    max_gen_toks: int = 4096


class BenchConfig(BaseModel):
    name: str
    # Optional: normally filled at run time by `with_hardware` (detected GPUs + CLI flags).
    hardware: Hardware | None = None
    parallelism: Parallelism = Parallelism()
    engine: Engine = Engine()
    models: list[ModelSpec]
    perf: PerfConfig
    accuracy: AccuracyConfig = AccuracyConfig()

    def config_hash(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def with_hardware(self, hardware: Hardware) -> BenchConfig:
        return self.model_copy(update={"hardware": hardware}, deep=True)

    def sessions(self) -> list[ServingSession]:
        return [
            ServingSession(config=self, model=m, precision=p)
            for m in self.models
            for p in m.precisions
        ]


class ServingSession(BaseModel):
    config: BenchConfig
    model: ModelSpec
    precision: Precision

    @property
    def hardware(self) -> Hardware:
        if self.config.hardware is None:
            raise ValueError("hardware unknown: run on a GPU machine or pass --gpu-type")
        return self.config.hardware

    @property
    def session_id(self) -> str:
        hw = self.config.hardware
        gpu = f"{hw.gpu_type.lower()}x{hw.gpu_count}" if hw else "gpu"
        return f"{gpu}_{self.model.slug}_{self.precision}"

    @property
    def served_model(self) -> str:
        """HF repo passed to `vllm serve` (a pre-quantized checkpoint, or the base model)."""
        return self.model.checkpoint(self.precision)

    @property
    def prequantized(self) -> bool:
        return self.served_model != self.model.hf_id

    def vllm_args(self) -> list[str]:
        """Full `vllm serve` argument list (after the model id)."""
        m = self.model
        args = [
            "--revision", m.revision,
            "--max-model-len", str(m.max_model_len),
            "--gpu-memory-utilization", str(m.gpu_memory_utilization),
            "--port", str(self.config.engine.port),
            "--seed", str(self.config.perf.seed),
        ]
        if self.prequantized:
            pass  # vLLM reads the quantization scheme from the checkpoint's config
        elif self.precision == "fp8":
            args += ["--quantization", "fp8"]
        else:
            args += ["--dtype", {"bf16": "bfloat16", "fp16": "float16"}[self.precision]]
        args += self.config.parallelism.vllm_args()
        for key, value in m.serving_args.items():
            flag = f"--{key}"
            if value is True:
                args.append(flag)
            elif isinstance(value, dict):
                args += [flag, json.dumps(value)]
            else:
                args += [flag, str(value)]
        return args


def load_config(path: str | Path) -> BenchConfig:
    with open(path) as f:
        return BenchConfig.model_validate(yaml.safe_load(f))
