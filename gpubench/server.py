"""vLLM server lifecycle: start, wait for health, capture logs, run `vllm bench serve`.

Two runtimes serve the same vLLM release:
- docker: the pinned `engine.image` (GCP VMs, Lambda, bare metal: any host with the NVIDIA
  container runtime).
- native: `pip install vllm==<engine.version>` in a uv venv, for hosts that are themselves
  containers and have no Docker (RunPod, Vast.ai, most Kubernetes GPU pods).
"""

from __future__ import annotations

import os
import re
import resource
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import httpx

from gpubench.config import ServingSession
from gpubench.paths import engines_dir, uv_cache

Runtime = Literal["docker", "native", "mock"]
CONTAINER_NAME = "gpubench-vllm"
KV_CACHE_RE = re.compile(r"KV cache size: ([\d,]+) tokens")
# One socket per simulated user; the default 1024 fd limit broke a 1024-user probe.
NOFILE = 65535
# vllm/vllm-openai images lack the `vllm[bench]` extra pandas that `vllm bench serve` needs to
# read custom JSONL datasets (random prompts don't). Installed into the load generator only,
# pinned, and recorded in the result.
BENCH_EXTRAS = ["pandas==2.2.3"]
# Rented boxes often have public IPs: never expose the benchmark server beyond localhost.
LOCAL_ONLY = ["--host", "127.0.0.1"]


def parse_kv_cache_tokens(log_text: str) -> int | None:
    """Parse vLLM's startup line 'GPU KV cache size: 1,234,567 tokens, Maximum concurrency ...'."""
    match = KV_CACHE_RE.search(log_text)
    return int(match.group(1).replace(",", "")) if match else None


def docker_has_nvidia() -> bool:
    if not shutil.which("docker"):
        return False
    out = subprocess.run(["docker", "info"], capture_output=True, text=True)
    return out.returncode == 0 and "nvidia" in out.stdout.lower()


def pick_runtime(requested: str = "auto") -> Runtime:
    if requested in ("docker", "native", "mock"):
        return requested  # type: ignore[return-value]
    return "docker" if docker_has_nvidia() else "native"


def raise_nofile_limit() -> int:
    """Raise this process's open-files soft limit (inherited by vLLM and the load generator)."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = NOFILE if hard == resource.RLIM_INFINITY else min(NOFILE, hard)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def native_venv(version: str) -> Path:
    return engines_dir() / f"vllm-{version}"


def ensure_native_vllm(version: str) -> Path:
    """Install vllm==version (with its benchmark extras) into its own venv once; return the venv.

    vLLM gets its own venv so its pinned torch/CUDA wheels never clash with gpubench and
    lm-eval. `--torch-backend=auto` lets uv pick the torch build matching the host's driver.
    """
    venv = native_venv(version)
    vllm_bin = venv / "bin" / "vllm"
    if vllm_bin.exists():
        return venv
    uv = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    venv.parent.mkdir(parents=True, exist_ok=True)
    # Keep uv's multi-GB wheel cache (torch, vLLM) on the big disk too.
    env = {**os.environ, "UV_CACHE_DIR": str(uv_cache())}
    subprocess.run([uv, "venv", "--python", "3.12", "--seed", str(venv)], check=True, env=env)
    subprocess.run(
        [uv, "pip", "install", "--python", str(venv / "bin" / "python"),
         f"vllm[bench]=={version}", "--torch-backend=auto"],
        check=True, env=env,
    )
    return venv


class VllmServer:
    """Common lifecycle; subclasses say how to launch vLLM and run commands next to it."""

    runtime: Runtime

    def __init__(self, session: ServingSession, log_path: Path, hf_cache: Path, work_dir: Path):
        self.session = session
        self.log_path = log_path
        self.hf_cache = hf_cache
        self.work_dir = work_dir
        self.port = session.config.engine.port
        self._log_proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    # Where `vllm bench serve` should write results / find datasets, as *it* sees the paths.
    @property
    def raw_dir(self) -> str:
        raise NotImplementedError

    @property
    def datasets_dir(self) -> str:
        raise NotImplementedError

    def start(self) -> None:
        raise NotImplementedError

    def is_running(self) -> bool:
        raise NotImplementedError

    def exec(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        """Run a command (`vllm bench serve ...`) alongside the server."""
        raise NotImplementedError

    def version(self) -> str | None:
        raise NotImplementedError

    def image_digest(self) -> str | None:
        return None

    def ensure_bench_extras(self) -> list[str]:
        """Extra packages the load generator needs for custom datasets (none by default)."""
        return []

    def python_cmd(self) -> list[str]:
        """A python with the serving stack's torch/CUDA, for the fingerprint microbenchmark
        (run while the server is down)."""
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def _wait_healthy(self, timeout_s: int) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self.is_running():
                raise RuntimeError(f"vLLM exited during startup; see {self.log_path}")
            try:
                if httpx.get(f"{self.base_url}/health", timeout=5).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(5)
        raise TimeoutError(f"vLLM not healthy after {timeout_s}s; see {self.log_path}")

    def kv_cache_tokens(self) -> int | None:
        return parse_kv_cache_tokens(self.log_path.read_text(errors="replace"))

    def __enter__(self) -> VllmServer:
        if not self.is_running():
            self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class DockerServer(VllmServer):
    runtime: Runtime = "docker"

    # The run dir is mounted at /work so results written in-container land on the host.
    raw_dir = "/work/raw"
    datasets_dir = "/work/datasets"

    def docker_cmd(self) -> list[str]:
        engine = self.session.config.engine
        return [
            "docker", "run", "-d", "--rm",
            "--name", CONTAINER_NAME,
            "--gpus", "all",
            "--ipc", "host",
            "--ulimit", f"nofile={NOFILE}:{NOFILE}",
            "--network", "host",
            "-v", f"{self.hf_cache}:/root/.cache/huggingface",
            "-v", f"{self.work_dir.resolve()}:/work",
            "-e", "HF_TOKEN",
            *[x for k, v in self.session.env.items() for x in ("-e", f"{k}={v}")],
            engine.image,
            self.session.served_model,
            *self.session.vllm_args(),
            *LOCAL_ONLY,
        ]

    def start(self) -> None:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        subprocess.run(self.docker_cmd(), check=True, capture_output=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_proc = subprocess.Popen(
            ["docker", "logs", "-f", CONTAINER_NAME],
            stdout=self.log_path.open("w"),
            stderr=subprocess.STDOUT,
        )
        self._wait_healthy(self.session.config.engine.startup_timeout_s)

    def is_running(self) -> bool:
        return subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            capture_output=True, text=True,
        ).stdout.strip() == "true"

    def exec(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "exec", CONTAINER_NAME, *args], **kwargs)

    def ensure_bench_extras(self) -> list[str]:
        """pip-install BENCH_EXTRAS into the running container (load generator deps)."""
        self.exec(["python3", "-m", "pip", "install", "--quiet", "--no-cache-dir",
                   *BENCH_EXTRAS], check=True, capture_output=True)
        return BENCH_EXTRAS

    def version(self) -> str | None:
        out = self.exec(
            ["python3", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or None

    def python_cmd(self) -> list[str]:
        return ["docker", "run", "--rm", "--gpus", "all", "--ipc", "host",
                "--entrypoint", "python3", self.session.config.engine.image]

    def image_digest(self) -> str | None:
        out = subprocess.run(
            ["docker", "inspect", "-f", "{{index .RepoDigests 0}}",
             self.session.config.engine.image],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or None

    def stop(self) -> None:
        subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
        if self._log_proc:
            self._log_proc.wait(timeout=30)


class NativeServer(VllmServer):
    runtime: Runtime = "native"

    def __init__(self, *args, venv: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.venv = venv or native_venv(self.session.config.engine.version)
        self._proc: subprocess.Popen | None = None

    @property
    def raw_dir(self) -> str:
        return str((self.work_dir / "raw").resolve())

    @property
    def datasets_dir(self) -> str:
        return str((self.work_dir / "datasets").resolve())

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["HF_HOME"] = str(self.hf_cache)
        env["PATH"] = f"{self.venv / 'bin'}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(self.venv)
        env.update(self.session.env)
        return env

    def serve_cmd(self) -> list[str]:
        return [
            str(self.venv / "bin" / "vllm"), "serve",
            self.session.served_model, *self.session.vllm_args(), *LOCAL_ONLY,
        ]

    def start(self) -> None:
        ensure_native_vllm(self.session.config.engine.version)
        raise_nofile_limit()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        (self.work_dir / "raw").mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            self.serve_cmd(),
            stdout=self.log_path.open("w"),
            stderr=subprocess.STDOUT,
            env=self._env(),
            start_new_session=True,  # so stop() can kill vLLM's worker processes too
        )
        self._wait_healthy(self.session.config.engine.startup_timeout_s)

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def exec(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        if args and args[0] in ("vllm", "python3", "python"):
            args = [str(self.venv / "bin" / args[0]), *args[1:]]
        return subprocess.run(args, env=self._env(), **kwargs)

    def python_cmd(self) -> list[str]:
        ensure_native_vllm(self.session.config.engine.version)
        return [str(self.venv / "bin" / "python")]

    def version(self) -> str | None:
        out = self.exec(
            ["python", "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True,
        )
        return out.stdout.strip() or None

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            try:
                os.killpg(self._proc.pid, 15)
                self._proc.wait(timeout=60)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self._proc.pid, 9)
                except ProcessLookupError:
                    pass


class MockServer(VllmServer):
    """CI only: gpubench.mock instead of vLLM (no GPU). Its runs can never be submitted."""

    runtime: Runtime = "mock"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._proc: subprocess.Popen | None = None

    @property
    def raw_dir(self) -> str:
        return str((self.work_dir / "raw").resolve())

    @property
    def datasets_dir(self) -> str:
        return str((self.work_dir / "datasets").resolve())

    def start(self) -> None:
        import sys

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "gpubench.mock", "serve", "--port", str(self.port)],
            stdout=self.log_path.open("w"), stderr=subprocess.STDOUT,
        )
        self._wait_healthy(60)
        time.sleep(0.5)  # let the startup line reach the log

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def exec(self, args: list[str], **kwargs) -> subprocess.CompletedProcess:
        import sys

        if args[:3] == ["vllm", "bench", "serve"]:
            args = [sys.executable, "-m", "gpubench.mock", "bench", *args[3:]]
        return subprocess.run(args, **kwargs)

    def version(self) -> str | None:
        return "mock"

    def python_cmd(self) -> list[str]:
        raise RuntimeError("the mock runtime has no GPU to fingerprint; use --no-fingerprint")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=30)


def make_server(
    runtime: Runtime, session: ServingSession, log_path: Path, hf_cache: Path, work_dir: Path
) -> VllmServer:
    if runtime == "mock":
        return MockServer(session, log_path, hf_cache, work_dir)
    cls = DockerServer if runtime == "docker" else NativeServer
    return cls(session, log_path, hf_cache, work_dir)
