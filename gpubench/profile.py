"""Nsight Systems profile of one load point: where GPU time goes, by kernel family.

Never mixed with the benchmark: a separate server start of the same pinned image and flags,
wrapped in `nsys profile`, capturing exactly the requests of one `vllm bench serve --profile`
call (vLLM's `--profiler-config {"profiler": "cuda"}` makes /start_profile call
cudaProfilerStart, and nsys only records between start and stop). For a 100k-token prompt with
a 1-token answer, that is exactly one prefill.

nsys comes from the host (Deep Learning VM images ship it with the CUDA toolkit; otherwise it
is installed from NVIDIA's apt repo) and is bind-mounted into the vLLM container.
"""

from __future__ import annotations

import csv
import io
import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from gpubench.config import ServingSession, Workload
from gpubench.schema import Profile
from gpubench.server import CONTAINER_NAME, LOCAL_ONLY, NOFILE, DockerServer

# Kernel families, first match wins (case-insensitive on the kernel name).
GROUPS: list[tuple[str, str]] = [
    # Specific names first: e.g. "reshape_and_cache_flash_kernel" writes the KV cache, it is
    # not FlashAttention.
    ("memory / KV cache", r"reshape_and_cache|memcpy|memset|copy_|_copy|fill_|cat_|gather"
                          r"|scatter"),
    ("full attention", r"flash_fwd|flash::|flashattn|fmha|attention|attn_fwd|paged"),
    ("linear attention (GDN)", r"gdn|gated_delta|delta_rule|chunk_|fla_|recurrent|causal_conv"
                               r"|conv1d|mamba|ssm|l2norm"),
    ("FP8/BF16 matmul", r"gemm|deep_gemm|cutlass|xmma|matmul|sm90|cublas|wgmma"),
    ("quantize / norm / activation", r"quant|rms|norm|silu|gelu|act_and_mul|rotary|rope"),
    ("sampling / other", r"."),
]


def _version_key(path: Path) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", next(
        (p.name for p in path.parents if re.match(r"\d{4}\.", p.name)), "0")))


def find_nsys() -> Path | None:
    """The newest nsys on the host: /opt/nvidia/nsight-systems*/<version>/, else PATH/CUDA."""
    versioned = sorted(Path("/opt/nvidia").glob("nsight-systems*/*/target-linux-x64/nsys"),
                       key=_version_key)
    if versioned:
        return versioned[-1].resolve()
    for c in (shutil.which("nsys"), "/usr/local/cuda/bin/nsys"):
        if c and Path(c).exists():
            return Path(c).resolve()
    return None


def ensure_nsys(echo: Callable[[str], None] = print) -> Path:
    """Install the latest nsight-systems-cli (a CUDA-toolkit nsys can be too old to trace the
    vLLM image's CUDA runtime), then return the newest nsys."""
    echo("installing the latest nsight-systems-cli from NVIDIA's apt repository")
    subprocess.run(["bash", "-c", (
        "set -e; . /etc/os-release; repo=ubuntu${VERSION_ID//./}; "
        "arch=$(dpkg --print-architecture); "
        "curl -fsSL https://developer.download.nvidia.com/devtools/repos/$repo/$arch/nvidia.pub "
        "| gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg; "
        "echo \"deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] "
        "https://developer.download.nvidia.com/devtools/repos/$repo/$arch/ /\" "
        "> /etc/apt/sources.list.d/nvidia-devtools.list; "
        "apt-get update -qq && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nsight-systems-cli")],
        check=False)
    found = find_nsys()
    if not found:
        raise RuntimeError("nsys not found (and nsight-systems-cli could not be installed)")
    echo(f"using {found}")
    return found


def nsys_root(nsys: Path) -> Path:
    """The Nsight Systems install directory to mount (the parent of bin/ or target-*/)."""
    for parent in nsys.parents:
        if (parent / "target-linux-x64").exists() or parent.name.startswith("nsight-systems"):
            return parent
    return nsys.parent.parent


class ProfiledDockerServer(DockerServer):
    """DockerServer whose vLLM runs under `nsys profile` (host nsys bind-mounted)."""

    def __init__(self, *args, nsys: Path, report: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.nsys = nsys
        self.report = report  # path inside the container, without extension

    def docker_cmd(self) -> list[str]:
        root = nsys_root(self.nsys)
        inner = str(Path("/opt/nsys") / self.nsys.relative_to(root))
        engine = self.session.config.engine
        return [
            "docker", "run", "-d",
            "--name", CONTAINER_NAME,
            "--gpus", "all", "--ipc", "host", "--network", "host",
            "--ulimit", f"nofile={NOFILE}:{NOFILE}",
            "--cap-add", "SYS_ADMIN",
            "-v", f"{self.hf_cache}:/root/.cache/huggingface",
            "-v", f"{self.work_dir.resolve()}:/work",
            "-v", f"{root}:/opt/nsys:ro",
            "-e", "HF_TOKEN",
            "--entrypoint", inner,
            engine.image,
            "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
            "--trace-fork-before-exec=true", "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop-shutdown", "--force-overwrite=true",
            "--output", self.report,
            "vllm", "serve", self.session.served_model, *self.session.vllm_args(), *LOCAL_ONLY,
            "--profiler-config", json.dumps({"profiler": "cuda"}),
        ]

    def wait_exit(self, timeout_s: int = 900) -> None:
        """nsys shuts the server down after the capture and writes the report; wait for it."""
        subprocess.run(["timeout", str(timeout_s), "docker", "wait", CONTAINER_NAME],
                       capture_output=True)

    def stop(self) -> None:
        super().stop()


# --- summarising ----------------------------------------------------------------------------

def group_of(name: str) -> str:
    for group, pattern in GROUPS:
        if re.search(pattern, name, re.IGNORECASE):
            return group
    return "sampling / other"


def summarise_kernels(kern_sum_csv: str, gpu_trace_csv: str | None = None,
                      top: int = 15) -> dict:
    """nsys `cuda_gpu_kern_sum` (+ optional `cuda_gpu_trace`) CSV -> summary dict."""
    rows = list(csv.DictReader(io.StringIO(kern_sum_csv)))
    kernels = []
    for r in rows:
        total_ns = float(r.get("Total Time (ns)") or 0)
        kernels.append({"name": r.get("Name", ""), "ms": total_ns / 1e6,
                        "instances": int(float(r.get("Instances") or 0))})
    busy = sum(k["ms"] for k in kernels)
    groups: dict[str, float] = {}
    for k in kernels:
        g = group_of(k["name"])
        groups[g] = groups.get(g, 0.0) + k["ms"]
    out = {
        "gpu_busy_ms": busy,
        "groups": [{"group": g, "ms": ms, "pct": ms / busy if busy else 0}
                   for g, ms in sorted(groups.items(), key=lambda x: -x[1])],
        "top_kernels": [{**k, "group": group_of(k["name"]), "pct": k["ms"] / busy if busy else 0}
                        for k in sorted(kernels, key=lambda k: -k["ms"])[:top]],
    }
    if gpu_trace_csv:
        starts, ends = [], []
        for r in csv.DictReader(io.StringIO(gpu_trace_csv)):
            try:
                s = float(r["Start (ns)"])
                starts.append(s)
                ends.append(s + float(r["Duration (ns)"]))
            except (KeyError, ValueError):
                continue
        if starts:
            span = (max(ends) - min(starts)) / 1e6
            out["capture_span_ms"] = span
            out["gpu_idle_pct"] = max(0.0, 1 - busy / span) if span else None
    return out


def nsys_stats(nsys: Path, report: Path) -> tuple[str, str | None]:
    def run(kind: str) -> str | None:
        proc = subprocess.run([str(nsys), "stats", "--report", kind, "--format", "csv",
                               "--force-export=true", str(report)],
                              capture_output=True, text=True)
        text = proc.stdout
        # nsys prints a banner before the CSV header; keep from the header on.
        idx = text.find('"Time')
        if idx < 0:
            idx = text.find("Time")
        return text[idx:] if idx >= 0 else None

    kern = run("cuda_gpu_kern_sum")
    if kern is None:
        raise RuntimeError(f"nsys stats produced no kernel summary for {report}")
    return kern, run("cuda_gpu_trace")


# --- running ----------------------------------------------------------------------------------

def profile_point(session: ServingSession, workload: Workload, concurrency: int,
                  work_dir: Path, hf_cache: Path, output_len: int = 1,
                  echo: Callable[[str], None] = print) -> Profile:
    """Profile `concurrency` requests of `workload` (answer length `output_len`) with nsys."""
    from gpubench.perf import bench_serve_args
    from gpubench.prompts import stage_dataset

    nsys = ensure_nsys(echo)
    prof_dir = work_dir / "profile"
    prof_dir.mkdir(parents=True, exist_ok=True)
    name = f"nsys_{workload.name}_c{concurrency}_out{output_len}"
    server = ProfiledDockerServer(session, prof_dir / f"{name}.vllm.log", hf_cache, work_dir,
                                  nsys=nsys, report=f"/work/profile/{name}")
    if workload.dataset == "custom":
        stage_dataset(workload, work_dir / "datasets")
    short = workload.model_copy(update={"output_len": output_len})
    server.start()
    try:
        args = bench_serve_args(session, short, concurrency, concurrency, 42,
                                f"{name}.bench.json", num_warmups=1,
                                raw_dir="/work/profile", datasets_dir="/work/datasets")
        args.insert(3, "--profile")
        echo(f"profiling {session.session_id} {workload.name} x{concurrency} (nsys)")
        server.exec(args, check=True)
        server.wait_exit()
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    report = prof_dir / f"{name}.nsys-rep"
    for _ in range(60):  # the report is finalised just after the container exits
        if report.exists():
            break
        time.sleep(5)
    kern, trace = nsys_stats(nsys, report)
    summary = summarise_kernels(kern, trace)
    (prof_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    echo("  " + ", ".join(f"{g['group']} {g['pct']:.0%}" for g in summary["groups"][:5]))
    return Profile(tool="nsys", workload=workload.name, concurrency=concurrency,
                   summary={"output_len": output_len, **summary}, asset=report.name)
