"""Multi-turn agent sessions and prefix-caching variants."""

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from gpubench.config import Hardware, load_config

ROOT = Path(__file__).resolve().parents[1]
MINI = ROOT / "tests" / "fixtures" / "ci" / "sessions-mini.jsonl"
H100 = Hardware(gpu_type="H100-80GB", provider="gcp", price_per_hour_usd=6.571)


@pytest.fixture
def exp():
    return load_config(ROOT / "configs" / "qwen3.8-27b-h100.yaml").with_hardware(H100)


def test_session_filter_selects_by_tag(exp):
    from gpubench.cli import filter_config

    a = filter_config(exp, None, ["toolcall-100k-512"], sessions=["fp8", "fp8-mtp"])
    assert sorted(s.tag for s in a.sessions()) == ["fp8", "fp8-mtp"]
    b = filter_config(exp, None, ["agent-sessions-100k"],
                      sessions=["fp8", "fp8-pc", "fp8-mtp-pc"])
    assert sorted(s.tag for s in b.sessions()) == ["fp8", "fp8-mtp-pc", "fp8-pc"]
    import typer

    with pytest.raises(typer.BadParameter):
        filter_config(exp, None, None, sessions=["fp16"])


def test_prefix_caching_variant_flips_the_base_flag(exp):
    by = {s.tag: s for s in exp.sessions()}
    base, pc = by["fp8"].vllm_args(), by["fp8-pc"].vllm_args()
    assert "--no-enable-prefix-caching" in base and "--enable-prefix-caching" not in base
    assert "--no-enable-prefix-caching" not in pc and "--enable-prefix-caching" in pc
    assert pc[pc.index("--mamba-cache-mode") + 1] == "align"
    assert by["fp8-pc"].env == {"VLLM_SERVER_DEV_MODE": "1"} and by["fp8"].env == {}


def test_variant_env_reaches_docker(exp, tmp_path):
    from gpubench.server import DockerServer

    pc = next(s for s in exp.sessions() if s.tag == "fp8-pc")
    cmd = DockerServer(pc, tmp_path / "l", tmp_path, tmp_path).docker_cmd()
    assert cmd[cmd.index("VLLM_SERVER_DEV_MODE=1") - 1] == "-e"


def test_session_turns_are_prefixes():
    from gpubench.prompts import session_turns

    msgs = [{"role": "user", "content": "task"}]
    for i in range(5):
        msgs += [{"role": "assistant", "content": f"call {i}"},
                 {"role": "tool", "content": f"output {i} " * 50}]
    render = lambda ms: "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in ms)  # noqa: E731
    turns = session_turns(render, msgs, 3)
    assert len(turns) == 3 and turns[-1] == render(msgs)
    assert all(b.startswith(a) for a, b in zip(turns, turns[1:], strict=False))
    bad = lambda ms: str(len(ms)) + render(ms)  # noqa: E731  (changes the start every turn)
    with pytest.raises(ValueError):
        session_turns(bad, msgs, 3)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_multiturn_client_against_mock(tmp_path):
    from gpubench.multiturn import run_sessions

    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "gpubench.mock", "serve", "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                if httpx.get(f"{url}/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.1)
        out = run_sessions(url, "m", MINI, concurrency=2, num_sessions=3, output_len=4,
                           result_path=tmp_path / "r.json", reset_cache=True)
    finally:
        proc.terminate()
    assert out["completed"] == 9 and out["failed"] == 0  # 3 sessions x 3 turns
    assert out["ttft_first_turn_p95_ms"] > 0 and out["ttft_later_turns_p99_ms"] > 0
    assert out["prefix_cache_hit_rate"] is None  # the mock has no /metrics
    assert json.loads((tmp_path / "r.json").read_text())["completed"] == 9


def test_prefix_cache_counters_parse(monkeypatch):
    from gpubench import multiturn

    text = ('# HELP x\nvllm:prefix_cache_queries_total{engine="0",model_name="m"} 1000.0\n'
            'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 900.0\n')
    monkeypatch.setattr(multiturn.httpx, "get", lambda *a, **k: type("R", (), {"text": text}))
    assert multiturn.prefix_cache_counters("http://x") == (900.0, 1000.0)
