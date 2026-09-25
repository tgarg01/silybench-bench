"""GPU telemetry via an nvidia-smi sampler running alongside each benchmark point."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FIELDS = "timestamp,index,power.draw,memory.used,utilization.gpu,clocks.sm"
TS_FORMAT = "%Y/%m/%d %H:%M:%S.%f"  # nvidia-smi timestamp, local time

Window = tuple[datetime, datetime]


@dataclass
class TelemetrySummary:
    avg_power_w: float  # summed across GPUs, averaged over time
    peak_memory_gb: float  # max per-GPU memory used
    avg_util_pct: float


def parse_samples(csv_text: str, windows: list[Window] | None = None) -> TelemetrySummary | None:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output for FIELDS.

    Power and utilization are averaged only over `windows` (the measured benchmark
    intervals) when given; otherwise the load generator's startup idle time would
    dilute them. Peak memory uses every sample.
    """
    power_by_ts: dict[str, float] = {}
    peak_mem_mib = 0.0
    utils: list[float] = []
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        ts, _idx, power, mem, util, _clk = parts
        try:
            peak_mem_mib = max(peak_mem_mib, float(mem))
            if windows is not None:
                t = datetime.strptime(ts, TS_FORMAT)
                if not any(start <= t <= end for start, end in windows):
                    continue
            power_by_ts[ts] = power_by_ts.get(ts, 0.0) + float(power)
            utils.append(float(util))
        except ValueError:
            continue  # "[N/A]" etc.
    if not power_by_ts:
        return None  # includes windows too short to catch a sample
    return TelemetrySummary(
        avg_power_w=sum(power_by_ts.values()) / len(power_by_ts),
        peak_memory_gb=peak_mem_mib / 1024,
        avg_util_pct=sum(utils) / len(utils),
    )


class GpuSampler:
    """Context manager: samples every `interval_ms` into a CSV file."""

    def __init__(self, out_path: Path, interval_ms: int = 500):
        self.out_path = out_path
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> GpuSampler:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [
                "nvidia-smi", f"--query-gpu={FIELDS}",
                "--format=csv,noheader,nounits", f"-lms={self.interval_ms}",
            ],
            stdout=self.out_path.open("w"),
            stderr=subprocess.DEVNULL,
        )
        return self

    def __exit__(self, *exc) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=10)

    def summary(self, windows: list[Window] | None = None) -> TelemetrySummary | None:
        return parse_samples(self.out_path.read_text(), windows)


def driver_info() -> tuple[str | None, str | None]:
    """(driver_version, cuda_version) from nvidia-smi, or (None, None) off-GPU."""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None, None
    driver = cuda = None
    for token in out.split("|"):
        if "Driver Version:" in token:
            driver = token.split("Driver Version:")[1].split()[0]
        if "CUDA Version:" in token:
            cuda = token.split("CUDA Version:")[1].split()[0]
    return driver, cuda
