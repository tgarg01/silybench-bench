"""Where big files go (model weights, the native vLLM venv, uv's wheel cache).

Rented pods often have a small root disk and a large volume elsewhere (RunPod: /workspace;
Lambda: /home or /lambda; others: /data, /mnt, /ephemeral). Model downloads filling the root
disk is the most common way a run dies, so on first use we pick the writable location with the
most free space and remember it in ~/.config/silybench/cache_root. Override with
SILYBENCH_CACHE (and HF_HOME for the model cache).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

CANDIDATES = ["/workspace", "/data", "/mnt", "/ephemeral", "/scratch", "/lambda"]
STATE = Path.home() / ".config" / "silybench" / "cache_root"


def _free(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free
    except OSError:
        return -1


def _writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".silybench-write-test"
        probe.write_text("")
        probe.unlink()
        return True
    except OSError:
        return False


def cache_root() -> Path:
    if env := os.environ.get("SILYBENCH_CACHE"):
        return Path(env)
    if STATE.exists():
        remembered = Path(STATE.read_text().strip())
        if remembered.exists():
            return remembered
    options = [Path.home() / ".cache" / "silybench"]
    options += [Path(c) / "silybench-cache" for c in CANDIDATES if Path(c).is_dir()]
    best = max(options, key=lambda p: _free(p if p.exists() else p.parent))
    if not _writable(best):
        best = options[0]
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(str(best))
    except OSError:
        pass
    return best


def hf_cache() -> Path:
    if env := os.environ.get("HF_HOME"):
        return Path(env)
    return cache_root() / "huggingface"


def engines_dir() -> Path:
    return cache_root() / "engines"


def uv_cache() -> Path:
    return Path(os.environ.get("UV_CACHE_DIR") or cache_root() / "uv")
