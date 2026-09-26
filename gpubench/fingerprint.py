"""Hardware fingerprint: what exactly a benchmark ran on, and how fast that machine really is.

An experiment's fingerprint is the reference that reproductions are checked against
(`gpubench verify-host`). It has three parts:

- identity: GPU model and PCI device id (tells SXM from PCIe/NVL), memory, power limit, ECC/MIG,
  GPU count, driver/CUDA, provider/machine type, CPU and RAM. Must match exactly.
- measured: a ~1 min microbenchmark in the vLLM environment: HBM copy bandwidth, BF16 and FP8
  matmul throughput, host<->device copy bandwidth, and download speed from Hugging Face.
  Must match within a tolerance (two healthy H100 SXM boxes differ by a few %).
- conditions: temperatures, clocks and throttle reasons. Recorded and warned about, never matched.

GPU serial numbers and UUIDs are never collected.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx

FINGERPRINT_VERSION = 1

# Runs with the vLLM environment's python (torch + CUDA). Prints one JSON line.
MICROBENCH = r"""
import json, time, torch
torch.backends.cuda.matmul.allow_tf32 = False
dev = torch.device("cuda:0")
out = {"torch": torch.__version__, "cuda_runtime": torch.version.cuda}

def timeit(fn, iters):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters

# HBM: device-to-device copy of 2 GiB (read + write = 2x bytes moved).
n = 2 * 1024**3
a = torch.empty(n, dtype=torch.uint8, device=dev); b = torch.empty_like(a)
t = timeit(lambda: b.copy_(a), 20)
out["hbm_copy_gbs"] = 2 * n / t / 1e9
del a, b

# Dense matmul throughput, 8192^3.
m = 8192
x = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
y = torch.randn(m, m, device=dev, dtype=torch.bfloat16)
t = timeit(lambda: x @ y, 50)
out["bf16_tflops"] = 2 * m**3 / t / 1e12
try:
    xf = x.to(torch.float8_e4m3fn); yf = y.t().contiguous().to(torch.float8_e4m3fn).t()
    one = torch.tensor(1.0, device=dev)
    mm = lambda: torch._scaled_mm(xf, yf, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
    t = timeit(mm, 50)
    out["fp8_tflops"] = 2 * m**3 / t / 1e12
except Exception as e:
    out["fp8_tflops"] = None
    out["fp8_error"] = str(e)[:200]

# Host <-> device over PCIe with pinned memory, 1 GiB.
n = 1024**3
h = torch.empty(n, dtype=torch.uint8, pin_memory=True)
d = torch.empty(n, dtype=torch.uint8, device=dev)
out["h2d_gbs"] = n / timeit(lambda: d.copy_(h, non_blocking=True), 10) / 1e9
out["d2h_gbs"] = n / timeit(lambda: h.copy_(d, non_blocking=True), 10) / 1e9
print(json.dumps(out))
"""


def _text(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    el = node.find(path)
    if el is None or el.text is None:
        return None
    value = el.text.strip()
    return None if value in ("", "N/A", "[N/A]", "Not Supported") else value


def _num(value: str | None) -> float | None:
    """'700.00 W' -> 700.0, '81559 MiB' -> 81559.0."""
    if value is None:
        return None
    try:
        return float(value.split()[0])
    except ValueError:
        return None


def parse_nvidia_smi_xml(xml_text: str) -> dict:
    """GPU identity + conditions from `nvidia-smi -q -x` (serials and UUIDs dropped)."""
    root = ET.fromstring(xml_text)
    gpus = []
    for g in root.findall("gpu"):
        power = g.find("gpu_power_readings")
        if power is None:
            power = g.find("power_readings")
        reasons = g.find("clocks_event_reasons")
        if reasons is None:
            reasons = g.find("clocks_throttle_reasons")
        active = []
        if reasons is not None:
            active = [c.tag for c in reasons
                      if (c.text or "").strip() == "Active" and "gpu_idle" not in c.tag]
        gpus.append({
            "name": _text(g, "product_name"),
            "architecture": _text(g, "product_architecture"),
            "pci_device_id": _text(g, "pci/pci_device_id"),
            "pci_sub_system_id": _text(g, "pci/pci_sub_system_id"),
            "vbios": _text(g, "vbios_version"),
            "memory_total_mib": _num(_text(g, "fb_memory_usage/total")),
            "default_power_limit_w": _num(_text(power, "default_power_limit")),
            "enforced_power_limit_w": _num(_text(power, "enforced_power_limit")),
            "max_sm_clock_mhz": _num(_text(g, "max_clocks/sm_clock")),
            "max_mem_clock_mhz": _num(_text(g, "max_clocks/mem_clock")),
            "ecc_mode": _text(g, "ecc_mode/current_ecc"),
            "mig_mode": _text(g, "mig_mode/current_mig"),
            "pcie_max_gen": _text(g, "pci/pci_gpu_link_info/pcie_gen/max_link_gen"),
            "pcie_max_width": _text(g, "pci/pci_gpu_link_info/link_widths/max_link_width"),
            "persistence_mode": _text(g, "persistence_mode"),
            "temperature_c": _num(_text(g, "temperature/gpu_temp")),
            "memory_temperature_c": _num(_text(g, "temperature/memory_temp")),
            "active_clock_limits": active,
        })
    return {
        "driver": _text(root, "driver_version"),
        "cuda": _text(root, "cuda_version"),
        "count": len(gpus),
        "gpus": gpus,
    }


def _read(path: str) -> str:
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def host_info() -> dict:
    cpu_model = next((line.split(":", 1)[1].strip() for line in _read("/proc/cpuinfo").splitlines()
                      if line.startswith("model name")), platform.processor() or None)
    mem_kb = next((int(line.split()[1]) for line in _read("/proc/meminfo").splitlines()
                   if line.startswith("MemTotal:")), None)
    return {
        "cpu_model": cpu_model,
        "vcpus": os.cpu_count(),
        "ram_gib": round(mem_kb / 1024**2, 1) if mem_kb else None,
        "os": platform.platform(),
        "kernel": platform.release(),
        "in_container": Path("/.dockerenv").exists() or "kubepods" in _read("/proc/1/cgroup")
        or bool(os.environ.get("RUNPOD_POD_ID") or os.environ.get("VAST_CONTAINERLABEL")),
    }


GCP_METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/"


def detect_provider() -> dict:
    """Where this machine is rented, from signals the providers themselves set."""
    try:
        headers = {"Metadata-Flavor": "Google"}
        mt = httpx.get(GCP_METADATA + "machine-type", headers=headers, timeout=1.5)
        if mt.status_code == 200:
            zone = httpx.get(GCP_METADATA + "zone", headers=headers, timeout=1.5).text
            sched = httpx.get(GCP_METADATA + "scheduling/provisioning-model", headers=headers,
                              timeout=1.5)
            # Set by infra/scripts/up.sh: the exact (pinned) boot image.
            image = httpx.get(GCP_METADATA + "attributes/gpubench-image", headers=headers,
                              timeout=1.5)
            return {
                "provider": "gcp",
                "machine_type": mt.text.rsplit("/", 1)[-1],
                "zone": zone.rsplit("/", 1)[-1],
                "provisioning": sched.text.lower() if sched.status_code == 200 else None,
                "image": image.text.rsplit("/", 1)[-1] if image.status_code == 200 else None,
            }
    except httpx.HTTPError:
        pass
    env = os.environ
    if env.get("RUNPOD_POD_ID"):
        return {"provider": "runpod", "machine_type": env.get("RUNPOD_GPU_TYPE_ID"),
                "zone": env.get("RUNPOD_DC_ID")}
    if env.get("VAST_CONTAINERLABEL") or env.get("CONTAINER_ID", "").startswith("C."):
        return {"provider": "vast", "machine_type": None, "zone": None}
    if Path("/etc/lambda-release").exists() or "lambda" in platform.node().lower():
        return {"provider": "lambda", "machine_type": None, "zone": None}
    return {"provider": "unknown", "machine_type": None, "zone": None}


def download_speed_mbps(url: str, nbytes: int = 256 * 1024**2) -> float | None:
    """MB/s for a ranged GET of `nbytes` (e.g. the first model shard on Hugging Face)."""
    headers = {"Range": f"bytes=0-{nbytes - 1}"}
    if token := os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    try:
        start, got = time.perf_counter(), 0
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_bytes(1 << 20):
                got += len(chunk)
        return got / (time.perf_counter() - start) / 1e6
    except httpx.HTTPError:
        return None


def hf_shard_url(repo: str, revision: str = "main") -> str | None:
    """URL of a model's first weight shard, for measuring download speed."""
    try:
        r = httpx.get(f"https://huggingface.co/{repo}/resolve/{revision}/"
                      "model.safetensors.index.json", follow_redirects=True, timeout=30)
        if r.status_code == 200:
            first = sorted(set(r.json()["weight_map"].values()))[0]
            return f"https://huggingface.co/{repo}/resolve/{revision}/{first}"
        return f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
    except (httpx.HTTPError, ValueError, KeyError):
        return None


def run_microbench(python: list[str]) -> dict:
    """Run MICROBENCH with the given python command (host venv or `docker run ... python3`)."""
    proc = subprocess.run([*python, "-c", MICROBENCH], capture_output=True, text=True,
                          timeout=600)
    if proc.returncode != 0:
        return {"error": (proc.stderr or proc.stdout)[-500:]}
    return json.loads(proc.stdout.strip().splitlines()[-1])


def nvidia_smi_xml() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, text=True,
                              timeout=60, check=True).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def collect(microbench_python: list[str] | None, speed_url: str | None,
            echo: Callable[[str], None] = print) -> dict:
    """Full fingerprint of this machine. Run it before vLLM starts (the GPU must be idle)."""
    xml = nvidia_smi_xml()
    gpu_idle = parse_nvidia_smi_xml(xml) if xml else {"count": 0, "gpus": []}
    fp = {
        "version": FINGERPRINT_VERSION,
        "collected_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "cloud": detect_provider(),
        "host": host_info(),
        "gpu": gpu_idle,
        "measured": {},
        "conditions": {
            "idle_temperature_c": [g["temperature_c"] for g in gpu_idle["gpus"]],
        },
    }
    if microbench_python:
        echo("fingerprint: GPU microbenchmark (~1 min)")
        fp["measured"] = run_microbench(microbench_python)
        xml = nvidia_smi_xml()  # right after the load: hot temperatures and clock limits
        if xml:
            hot = parse_nvidia_smi_xml(xml)["gpus"]
            fp["conditions"]["after_load_temperature_c"] = [g["temperature_c"] for g in hot]
            fp["conditions"]["after_load_clock_limits"] = [g["active_clock_limits"] for g in hot]
    if speed_url:
        fp["measured"]["hf_download_mbps"] = download_speed_mbps(speed_url)
    return fp


def summary(fp: dict) -> str:
    g = fp["gpu"]["gpus"][0] if fp["gpu"]["gpus"] else {}
    m = fp.get("measured", {})
    return (f"{fp['gpu']['count']}x {g.get('name')} ({g.get('pci_device_id')}, "
            f"{g.get('memory_total_mib')} MiB, {g.get('default_power_limit_w')} W) on "
            f"{fp['cloud'].get('provider')} {fp['cloud'].get('machine_type') or ''}; "
            f"HBM {m.get('hbm_copy_gbs') or 0:.0f} GB/s, BF16 {m.get('bf16_tflops') or 0:.0f} "
            f"TFLOPS, FP8 {m.get('fp8_tflops') or 0:.0f} TFLOPS, "
            f"H2D {m.get('h2d_gbs') or 0:.1f} GB/s")
