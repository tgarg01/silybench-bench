"""Published experiments: what was run where, and whether this machine can reproduce it.

An experiment lives in silybench-data at experiments/<id>/:
  experiment.yaml   the exact environment (provider, machine type, zone, image, runtime),
                    the silybench-bench tag + commit, the campaign config and the run ids
  fingerprint.json  the reference hardware fingerprint captured during the experiment

`verify-host` compares this machine's fingerprint with the reference: identity fields must
match exactly, measured speeds within a tolerance, and conditions only produce warnings.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml
from pydantic import BaseModel

from gpubench.schema import RunResult

DEFAULT_DATA_REPO = "tgarg01/silybench-data"
MEASURED_TOLERANCE = 0.05  # +/-5% on bandwidth and matmul throughput
DOWNLOAD_WARN_FRACTION = 0.5  # warn if HF downloads are < half the reference speed


class Environment(BaseModel):
    provider: str
    machine_type: str | None = None
    zone: str | None = None
    provisioning: str | None = None
    image: str | None = None  # VM image / pod template
    runtime: str | None = None  # docker | native
    terraform_env: str | None = None  # infra/terraform/envs/<name> for GCP


class BenchRef(BaseModel):
    repo: str = "https://github.com/tgarg01/silybench-bench"
    tag: str
    commit: str | None = None
    config: str


class Experiment(BaseModel):
    id: str
    title: str
    status: str  # planned | published | pipeline
    description: str = ""
    bench: BenchRef
    environments: list[Environment]  # where it was run; reproductions must use one of these
    runs: list[str] = []
    release: str | None = None  # silybench-data release holding the raw archives
    published_at: str | None = None
    # Generated from the config by `gpubench experiment manifest` (shown before results exist).
    models: list[dict] = []
    scenarios: list[dict] = []
    # Run order: each entry is a set of `gpubench run` filters (watch.sh --phase), in order.
    phases: list[str] = []


@dataclass
class Check:
    field: str
    status: str  # PASS | FAIL | WARN
    expected: object
    actual: object
    note: str = ""


# --- loading -------------------------------------------------------------------------------

def _get(data_root: Path | None, repo: str, rel: str) -> str | None:
    if data_root:
        path = data_root / rel
        return path.read_text() if path.exists() else None
    r = httpx.get(f"https://raw.githubusercontent.com/{repo}/main/{rel}", timeout=30)
    return r.text if r.status_code == 200 else None


def load_experiment(exp_id: str, data_root: Path | None = None,
                    repo: str = DEFAULT_DATA_REPO) -> tuple[Experiment, dict | None]:
    text = _get(data_root, repo, f"experiments/{exp_id}/experiment.yaml")
    if text is None:
        raise FileNotFoundError(f"no experiment {exp_id!r} in {data_root or repo}")
    exp = Experiment.model_validate(yaml.safe_load(text))
    fp = _get(data_root, repo, f"experiments/{exp_id}/fingerprint.json")
    return exp, json.loads(fp) if fp else None


def list_experiments(data_root: Path | None = None, repo: str = DEFAULT_DATA_REPO) -> list[dict]:
    text = _get(data_root, repo, "derived/experiments.json")
    return json.loads(text) if text else []


def load_runs(exp: Experiment, data_root: Path | None = None,
              repo: str = DEFAULT_DATA_REPO) -> list[RunResult]:
    runs = []
    for run_id in exp.runs:
        text = _get(data_root, repo, f"runs/{run_id}/result.json")
        if text:
            runs.append(RunResult.model_validate_json(text))
    return runs


# --- verify-host -----------------------------------------------------------------------------

def _gpu0(fp: dict) -> dict:
    gpus = fp.get("gpu", {}).get("gpus") or [{}]
    return gpus[0]


def check_environment(exp: Experiment, cloud: dict) -> list[Check]:
    """Is this machine at one of the providers/machine types the experiment was run on?"""
    provider = cloud.get("provider")
    matches = [e for e in exp.environments if e.provider == provider]
    if not matches:
        allowed = ", ".join(sorted({e.provider for e in exp.environments}))
        return [Check("provider", "FAIL", allowed, provider,
                      f"this experiment was only run on {allowed}; reproducing it on "
                      f"{provider} is not supported")]
    env = matches[0]
    checks = [Check("provider", "PASS", env.provider, provider)]
    if env.machine_type:
        ok = cloud.get("machine_type") == env.machine_type
        checks.append(Check("machine_type", "PASS" if ok else "FAIL", env.machine_type,
                            cloud.get("machine_type")))
    if env.zone:
        ok = cloud.get("zone") == env.zone
        checks.append(Check("zone", "PASS" if ok else "WARN", env.zone, cloud.get("zone"),
                            "" if ok else "same machine type in another zone is usually the "
                            "same hardware; verify-host's measured checks decide"))
    return checks


def compare_fingerprints(ref: dict, cur: dict) -> list[Check]:
    checks: list[Check] = []
    rg, cg = _gpu0(ref), _gpu0(cur)

    def exact(field: str, expected, actual, note: str = "") -> None:
        if expected is None:
            return
        checks.append(Check(field, "PASS" if expected == actual else "FAIL", expected, actual,
                            note))

    exact("gpu.count", ref["gpu"].get("count"), cur["gpu"].get("count"))
    exact("gpu.name", rg.get("name"), cg.get("name"))
    exact("gpu.pci_device_id", rg.get("pci_device_id"), cg.get("pci_device_id"),
          "identifies the exact variant (e.g. H100 SXM vs PCIe vs NVL)")
    exact("gpu.memory_total_mib", rg.get("memory_total_mib"), cg.get("memory_total_mib"))
    exact("gpu.default_power_limit_w", rg.get("default_power_limit_w"),
          cg.get("default_power_limit_w"))
    exact("gpu.mig_mode", rg.get("mig_mode"), cg.get("mig_mode"))
    exact("gpu.ecc_mode", rg.get("ecc_mode"), cg.get("ecc_mode"))
    exact("driver", ref["gpu"].get("driver"), cur["gpu"].get("driver"))
    exact("cuda", ref["gpu"].get("cuda"), cur["gpu"].get("cuda"))
    exact("host.vcpus", ref["host"].get("vcpus"), cur["host"].get("vcpus"))
    exact("host.cpu_model", ref["host"].get("cpu_model"), cur["host"].get("cpu_model"))
    if rg.get("vbios") and rg.get("vbios") != cg.get("vbios"):
        checks.append(Check("gpu.vbios", "WARN", rg.get("vbios"), cg.get("vbios"),
                            "different firmware on the same SKU; measured checks decide"))
    if ref["host"].get("ram_gib") and cur["host"].get("ram_gib"):
        ok = abs(cur["host"]["ram_gib"] / ref["host"]["ram_gib"] - 1) <= 0.02
        checks.append(Check("host.ram_gib", "PASS" if ok else "FAIL", ref["host"]["ram_gib"],
                            cur["host"]["ram_gib"]))

    rm, cm = ref.get("measured", {}), cur.get("measured", {})
    for key in ("hbm_copy_gbs", "bf16_tflops", "fp8_tflops", "h2d_gbs", "d2h_gbs"):
        if rm.get(key) is None:
            continue
        if cm.get(key) is None:
            checks.append(Check(f"measured.{key}", "FAIL", rm[key], None, "not measured"))
            continue
        ratio = cm[key] / rm[key]
        ok = abs(ratio - 1) <= MEASURED_TOLERANCE
        checks.append(Check(f"measured.{key}", "PASS" if ok else "FAIL", round(rm[key], 1),
                            round(cm[key], 1), f"{(ratio - 1) * 100:+.1f}% "
                            f"(tolerance ±{MEASURED_TOLERANCE * 100:.0f}%)"))
    if rm.get("hf_download_mbps") and cm.get("hf_download_mbps"):
        slow = cm["hf_download_mbps"] < DOWNLOAD_WARN_FRACTION * rm["hf_download_mbps"]
        checks.append(Check("measured.hf_download_mbps", "WARN" if slow else "PASS",
                            round(rm["hf_download_mbps"]), round(cm["hf_download_mbps"]),
                            "slow downloads only lengthen setup; results are unaffected"
                            if slow else ""))

    limits = cur.get("conditions", {}).get("after_load_clock_limits") or []
    thermal = [r for gpu in limits for r in gpu if "thermal" in r or "hw_slowdown" in r]
    if thermal:
        checks.append(Check("conditions.clock_limits", "WARN", [], thermal,
                            "the GPU throttled during the microbenchmark (cooling problem?)"))
    temps = cur.get("conditions", {}).get("after_load_temperature_c") or []
    ref_temps = ref.get("conditions", {}).get("after_load_temperature_c") or []
    if temps and ref_temps and max(t or 0 for t in temps) > max(t or 0 for t in ref_temps) + 10:
        checks.append(Check("conditions.temperature_c", "WARN", ref_temps, temps,
                            "runs >10 °C hotter than the reference machine"))
    return checks


def verdict(checks: list[Check]) -> bool:
    return not any(c.status == "FAIL" for c in checks)


def format_checks(checks: list[Check]) -> str:
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}
    return "\n".join(
        f"{icon[c.status]} {c.field}: expected {c.expected}, got {c.actual}"
        + (f"  ({c.note})" if c.note else "") for c in checks
    )


# --- plan from recorded durations ----------------------------------------------------------

OVERHEAD_S_PER_REPEAT = 15
STARTUP_S = 300
FINGERPRINT_S = 120


def recorded_hours(runs: list[RunResult], include_accuracy: bool = True) -> dict[str, float]:
    """Wall-clock the published runs actually spent, per run id (hours)."""
    out = {}
    for r in runs:
        perf = sum(p.duration_s * p.repeats + OVERHEAD_S_PER_REPEAT * p.repeats for p in r.perf)
        out[r.run_id] = (perf + STARTUP_S + FINGERPRINT_S) / 3600
    return out


# --- compare a reproduction with the published numbers -----------------------------------------

COMPARE_TOLERANCE = 0.10


def compare_runs(ref: RunResult, rep: RunResult) -> list[Check]:
    checks: list[Check] = []
    ref_cap = {c.workload: c for c in ref.capacity}
    for c in rep.capacity:
        if c.workload not in ref_cap:
            continue
        e, a = ref_cap[c.workload].max_users_slo, c.max_users_slo
        ok = e == a or (e and abs(a / e - 1) <= COMPARE_TOLERANCE)
        checks.append(Check(f"{c.workload}: max users (SLO)", "PASS" if ok else "WARN", e, a))
    ref_pts = {(p.workload, p.concurrency): p for p in ref.perf}
    for p in rep.perf:
        q = ref_pts.get((p.workload, p.concurrency))
        if q is None or p.concurrency not in (1, 16):
            continue
        for name, e, a in [("output tok/s", q.output_throughput, p.output_throughput),
                           ("median ITL ms", q.itl_ms.median, p.itl_ms.median),
                           ("p99 TTFT ms", q.ttft_ms.p99, p.ttft_ms.p99)]:
            ok = e and abs(a / e - 1) <= COMPARE_TOLERANCE
            checks.append(Check(f"{p.workload} @{p.concurrency}: {name}",
                                "PASS" if ok else "WARN", round(e, 1), round(a, 1),
                                f"{(a / e - 1) * 100:+.1f}%" if e else ""))
    return checks


def data_root_from_env() -> Path | None:
    root = os.environ.get("SILYBENCH_DATA_DIR")
    return Path(root) if root else None


def echo_checks(checks: list[Check], echo: Callable[[str], None]) -> None:
    echo(format_checks(checks))


def manifest_from_config(cfg) -> dict:
    """models + scenarios blocks for experiment.yaml, derived from a campaign config."""
    perf = cfg.perf
    return {
        "models": [
            {"hf_id": m.hf_id, "precision": p, "checkpoint": m.checkpoint(p),
             "max_model_len": m.max_model_len}
            for m in cfg.models for p in m.precisions
        ] + [
            {"hf_id": m.hf_id, "precision": v.precision, "variant": name,
             "checkpoint": m.checkpoint(v.precision), "max_model_len": m.max_model_len,
             "serving_args": v.serving_args}
            for m in cfg.models for name, v in m.variants.items()
        ],
        "scenarios": [
            {"name": w.name, "input_len": w.input_len, "output_len": w.output_len,
             "dataset": w.dataset, "users": perf.concurrency_for(w),
             "repeats": perf.repeats_for(w), "slo": perf.slo_for(w).model_dump(),
             "quality": w.quality is not None}
            for w in perf.workloads
        ],
    }
