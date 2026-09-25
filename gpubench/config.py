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
    hf_id: str
    revision: str = "main"
    precisions: list[Precision] = ["bf16"]
    max_model_len: int = 16384
    gpu_memory_utilization: float = 0.90
    # Extra vLLM flags, e.g. {"reasoning-parser": "qwen3"}. Values may be str/int/dict.
    serving_args: dict[str, object] = Field(default_factory=dict)
    # Recorded in results so thinking vs. non-thinking runs are never mixed up.
    thinking: bool = False

    @property
    def slug(self) -> str:
        return self.hf_id.split("/")[-1].lower()


class Workload(BaseModel):
    name: str
    dataset: Literal["random", "sharegpt"] = "random"
    input_len: int | None = None
    output_len: int | None = None
    dataset_path: str | None = None  # for sharegpt

    @model_validator(mode="after")
    def _check(self) -> Workload:
        if self.dataset == "random" and (self.input_len is None or self.output_len is None):
            raise ValueError(f"workload {self.name}: random dataset needs input_len and output_len")
        return self

    @property
    def tokens_per_request(self) -> int | None:
        """ISL+OSL, used to turn KV-cache token capacity into a max-concurrency figure."""
        if self.input_len is None or self.output_len is None:
            return None
        return self.input_len + self.output_len


class SLO(BaseModel):
    """A concurrency level 'passes' when every limit holds. Drives max concurrent users."""

    ttft_p99_ms: float = 2000.0
    itl_median_ms: float = 50.0  # 50 ms => >= 20 tok/s per user


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
    concurrency: list[int] = [1, 4, 16, 32, 64, 128, 256, 512]
    # num_prompts = max(min_prompts, prompts_per_user * concurrency)
    prompts_per_user: int = 5
    min_prompts: int = 50
    num_warmups: int = 10
    repeats: int = 3
    seed: int = 42
    slo: SLO = SLO()
    capacity: CapacitySearch = CapacitySearch()

    def num_prompts(self, concurrency: int) -> int:
        return max(self.min_prompts, self.prompts_per_user * concurrency)


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
        if self.precision == "fp8":
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
