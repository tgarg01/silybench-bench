"""Detect the GPUs of the machine we're on and name them the way results are keyed.

Rented boxes (RunPod, Vast, Lambda, ...) report GPUs as e.g. "NVIDIA H100 80GB HBM3". Results
use a short normalised name so runs from different providers of the same GPU line up, and the
data repo can reprice one benchmark across every provider renting that GPU.
"""

from __future__ import annotations

import re
import subprocess

from gpubench.config import Hardware

# Marketed memory sizes; nvidia-smi reports slightly less usable memory (e.g. 81559 MiB).
STANDARD_GB = (8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 94, 96, 141, 180, 192, 288)

# (regex on the nvidia-smi name, family). First match wins, so specific names go first.
FAMILIES: list[tuple[str, str]] = [
    (r"GB200", "GB200"),
    (r"B300", "B300"),
    (r"B200", "B200"),
    (r"GH200", "GH200"),
    (r"H200 NVL", "H200-NVL"),
    (r"H200", "H200"),
    (r"H100 NVL", "H100-NVL"),
    (r"H100 PCIe", "H100-PCIe"),
    (r"H100", "H100"),  # SXM (HBM3), the default H100
    (r"H800", "H800"),
    (r"A100.*PCIE", "A100-PCIe"),
    (r"A100", "A100"),
    (r"L40S", "L40S"),
    (r"L40", "L40"),
    (r"L4\b", "L4"),
    (r"A10G", "A10G"),
    (r"A10\b", "A10"),
    (r"A6000", "A6000"),
    (r"RTX 6000 Ada", "RTX6000-Ada"),
    (r"RTX PRO 6000", "RTXPRO6000"),
    (r"RTX 5090", "RTX5090"),
    (r"RTX 4090", "RTX4090"),
    (r"RTX 3090", "RTX3090"),
]


def round_memory_gb(mib: float) -> int:
    """Closest marketed size a little above the reported (usable) memory."""
    gb = mib / 1024
    candidates = [size for size in STANDARD_GB if size * 0.9 <= gb <= size * 1.02]
    return min(candidates, key=lambda size: abs(size - gb)) if candidates else round(gb)


def normalise_gpu_name(name: str, memory_mib: float) -> str:
    """"NVIDIA H100 80GB HBM3", 81559 -> "H100-80GB"."""
    family = next(
        (fam for pattern, fam in FAMILIES if re.search(pattern, name, re.IGNORECASE)), None
    )
    if family is None:
        family = re.sub(r"[^A-Za-z0-9]+", "-", name.replace("NVIDIA", "")).strip("-")
    return f"{family}-{round_memory_gb(memory_mib)}GB"


def parse_gpu_query(csv_text: str) -> list[tuple[str, float]]:
    """Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`."""
    gpus = []
    for line in csv_text.splitlines():
        name, _, mem = line.rpartition(",")
        if name.strip() and mem.strip().replace(".", "").isdigit():
            gpus.append((name.strip(), float(mem)))
    return gpus


def query_gpus() -> list[tuple[str, float]]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    return parse_gpu_query(out)


def detect_hardware(
    provider: str,
    price_per_hour_usd: float | None = None,
    provisioning: str = "on-demand",
    gpu_type: str | None = None,
    gpu_count: int | None = None,
    machine_type: str | None = None,
    zone: str | None = None,
) -> Hardware:
    """Hardware block for this machine. Explicit arguments override what is detected."""
    gpus = query_gpus()
    if gpus:
        names = {normalise_gpu_name(n, m) for n, m in gpus}
        if len(names) > 1 and not gpu_type:
            raise ValueError(f"mixed GPUs on this machine: {sorted(names)}; pass --gpu-type")
        detected = names.pop()
    elif gpu_type is None:
        raise ValueError("no GPU detected (nvidia-smi failed); pass --gpu-type to override")
    else:
        detected = gpu_type
    return Hardware(
        gpu_type=gpu_type or detected,
        gpu_count=gpu_count or len(gpus) or 1,
        provider=provider.lower(),
        machine_type=machine_type,
        zone=zone,
        provisioning=provisioning,
        price_per_hour_usd=price_per_hour_usd,
    )
