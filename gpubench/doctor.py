"""`gpubench doctor`: everything that has broken a run before, checked before paying for one."""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import httpx

from gpubench.config import BenchConfig
from gpubench.hardware import normalise_gpu_name, query_gpus
from gpubench.plan import BYTES_PER_PARAM, model_params_b
from gpubench.server import NOFILE, docker_has_nvidia, native_venv, pick_runtime
from gpubench.telemetry import driver_info

OK, WARN, FAIL = "✓", "!", "✗"


def _free_gb(path: Path) -> float:
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def run_checks(cfg: BenchConfig | None, out: Path, hf_cache: Path, runtime: str,
               echo: Callable[[str], None] = print) -> bool:
    failed = False

    def report(status: str, msg: str) -> None:
        nonlocal failed
        failed |= status == FAIL
        echo(f"{status} {msg}")

    gpus = query_gpus()
    if gpus:
        names = sorted({normalise_gpu_name(n, m) for n, m in gpus})
        report(OK, f"{len(gpus)} GPU(s): {', '.join(names)} ({gpus[0][0]})")
    else:
        report(FAIL, "no NVIDIA GPU visible (nvidia-smi failed)")
    driver, cuda = driver_info()
    if driver:
        report(OK, f"driver {driver}, CUDA {cuda}")

    rt = pick_runtime(runtime)
    if rt == "docker":
        if docker_has_nvidia():
            report(OK, "runtime: docker with the NVIDIA runtime")
        else:
            report(FAIL, "runtime docker requested but docker has no NVIDIA runtime")
    else:
        uv = shutil.which("uv") or (Path.home() / ".local/bin/uv").exists()
        report(OK if uv else FAIL, "runtime: native (pip vLLM in a uv venv; no Docker needed)"
               + ("" if uv else " - uv missing, run ./setup.sh"))
        if cfg:
            venv = native_venv(cfg.engine.version)
            report(OK if (venv / "bin/vllm").exists() else WARN,
                   f"vLLM {cfg.engine.version} "
                   + ("installed" if (venv / "bin/vllm").exists() else
                      "not installed yet (installs on first run, ~5 min; or "
                      "`uv run gpubench install-engine <config>`)"))

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or hard >= NOFILE:
        shown = "unlimited" if hard == resource.RLIM_INFINITY else hard
        report(OK, f"open-files limit can be raised to {NOFILE} (hard limit {shown})")
    else:
        report(WARN, f"open-files hard limit is {hard} (< {NOFILE}): capacity probes above "
                     f"~{hard // 2} users may fail; results are still valid lower bounds")

    need_gb = 30.0  # vLLM wheels/image, CUDA graphs, logs
    if cfg:
        for m in cfg.models:
            # FP8 quantizes BF16 weights on load, so one BF16 copy per model is downloaded.
            need_gb += model_params_b(m.hf_id) * BYTES_PER_PARAM["bf16"] * 1.05
    free_cache, free_out = _free_gb(hf_cache), _free_gb(out)
    status = OK if free_cache >= need_gb else FAIL
    report(status, f"disk: {free_cache:.0f} GB free for the model cache ({hf_cache}); "
                   f"need ~{need_gb:.0f} GB" + ("" if cfg else " (pass --config for the "
                                                  "exact figure)"))
    if free_out < 2:
        report(FAIL, f"only {free_out:.1f} GB free for results in {out}")

    try:
        shm = shutil.disk_usage("/dev/shm").total / 1e9
        tp = cfg.parallelism.tp if cfg else 1
        report(OK if shm >= 8 or tp == 1 else WARN,
               f"/dev/shm {shm:.1f} GB" + ("" if shm >= 8 or tp == 1 else
                                            " (small for tensor parallel; NCCL may fail)"))
    except OSError:
        pass

    try:
        httpx.get("https://huggingface.co/api/models/Qwen/Qwen3-8B", timeout=10)
        report(OK, "huggingface.co reachable")
    except httpx.HTTPError:
        report(FAIL, "cannot reach huggingface.co (model downloads will fail)")

    gated = cfg and cfg.accuracy.enabled and any(
        "gpqa" in t.name for t in cfg.accuracy.tasks)
    if os.environ.get("HF_TOKEN"):
        report(OK, "HF_TOKEN set")
    else:
        report(WARN if gated or not cfg else OK,
               "HF_TOKEN not set" + (": GPQA (gated) will be skipped - accept its terms on "
                                     "huggingface.co and export HF_TOKEN" if gated or not cfg
                                     else " (not needed for this campaign)"))

    if cfg and cfg.accuracy.enabled:
        has_lm_eval = subprocess.run([sys.executable, "-c", "import lm_eval"],
                                     capture_output=True).returncode == 0
        report(OK if has_lm_eval else FAIL,
               "lm-eval installed" if has_lm_eval else
               "lm-eval missing: run `uv sync --extra eval` (./setup.sh does this)")

    if shutil.which("gh"):
        authed = subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0
        report(OK if authed else WARN, "gh logged in" if authed else
               "gh not logged in (needed only for `gpubench submit`): gh auth login")
    else:
        report(WARN, "gh (GitHub CLI) missing: needed for `gpubench submit`; run ./setup.sh")

    report(OK if shutil.which("tmux") else WARN,
           "tmux available" if shutil.which("tmux") else
           "tmux missing (optional; `gpubench run --detach` works without it)")
    echo("" if not failed else "\nfix the ✗ items before starting a run")
    return not failed
