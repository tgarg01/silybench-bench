"""GPU telemetry via an nvidia-smi sampler running alongside each benchmark point."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

BASE_FIELDS = "timestamp,index,power.draw,memory.used,utilization.gpu,clocks.sm"
# Newer drivers call throttle reasons "clocks_event_reasons"; older ones "clocks_throttle_reasons".
EXTRA_FIELDS = ("temperature.gpu,clocks.mem,clocks_event_reasons.active",
                "temperature.gpu,clocks.mem,clocks_throttle_reasons.active")
FIELDS = BASE_FIELDS  # kept for callers/tests that only need the base columns
# Bits of the active-reasons mask that mean the GPU slowed down for heat (not the normal
# power-cap limiting under load): HW slowdown, SW thermal, HW thermal.
THERMAL_MASK = 0x08 | 0x20 | 0x40
TS_FORMAT = "%Y/%m/%d %H:%M:%S.%f"  # nvidia-smi timestamp, local time

Window = tuple[datetime, datetime]


@dataclass
class TelemetrySummary:
    avg_power_w: float  # summed across GPUs, averaged over time
    peak_memory_gb: float  # max per-GPU memory used
    avg_util_pct: float
    max_temp_c: float | None = None
    thermal_throttle_fraction: float | None = None  # share of samples with a thermal limit


def parse_samples(csv_text: str, windows: list[Window] | None = None) -> TelemetrySummary | None:
    """Parse `nvidia-smi --format=csv,noheader,nounits` output for FIELDS.

    Power and utilization are averaged only over `windows` (the measured benchmark
    intervals) when given; otherwise the load generator's startup idle time would
    dilute them. Peak memory uses every sample.
    """
    power_by_ts: dict[str, float] = {}
    peak_mem_mib = 0.0
    utils: list[float] = []
    temps: list[float] = []
    throttled = samples = 0
    for line in csv_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) not in (6, 9):
            continue
        ts, _idx, power, mem, util, _clk = parts[:6]
        if len(parts) == 9:
            try:
                temps.append(float(parts[6]))
            except ValueError:
                pass
            try:
                samples += 1
                throttled += bool(int(parts[8], 16) & THERMAL_MASK)
            except ValueError:
                samples -= 1
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
        max_temp_c=max(temps) if temps else None,
        thermal_throttle_fraction=throttled / samples if samples else None,
    )


class GpuSampler:
    """Context manager: samples every `interval_ms` into a CSV file."""

    def __init__(self, out_path: Path, interval_ms: int = 500):
        self.out_path = out_path
        self.interval_ms = interval_ms
        self._proc: subprocess.Popen | None = None

    def __enter__(self) -> GpuSampler:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        if shutil.which("nvidia-smi") is None:  # CI mock runs: no GPU, no telemetry
            self.out_path.write_text("")
            return self
        self._proc = subprocess.Popen(
            [
                "nvidia-smi", f"--query-gpu={query_fields()}",
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


_FIELDS_CACHE: str | None = None


def query_fields() -> str:
    """The richest field list this driver's nvidia-smi accepts."""
    global _FIELDS_CACHE
    if _FIELDS_CACHE is None:
        _FIELDS_CACHE = BASE_FIELDS
        for extra in EXTRA_FIELDS:
            fields = f"{BASE_FIELDS},{extra}"
            try:
                ok = subprocess.run(["nvidia-smi", f"--query-gpu={fields}",
                                     "--format=csv,noheader,nounits"],
                                    capture_output=True, timeout=30).returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                break
            if ok:
                _FIELDS_CACHE = fields
                break
    return _FIELDS_CACHE


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
