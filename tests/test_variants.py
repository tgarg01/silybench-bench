"""Serving variants, finer capacity search, nsys profile summary and the variant report."""

from pathlib import Path

import pytest

from gpubench.capacity import find_max_users
from gpubench.config import SLO, CapacitySearch, Hardware, Workload, load_config
from gpubench.perf import build_perf_point

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
H100 = Hardware(gpu_type="H100-80GB", provider="gcp", price_per_hour_usd=6.571)


@pytest.fixture
def tune():
    return load_config(ROOT / "configs" / "tune-qwen3.8-27b-fp8-100k.yaml").with_hardware(H100)


def test_variants_become_sessions_with_their_flags(tune):
    ids = [s.session_id for s in tune.sessions()]
    assert ids[0] == "h100-80gbx1_qwen3.8-27b_fp8-base" and len(ids) == 8
    by = {s.variant: s for s in tune.sessions()}
    base, chunk, mtp = by["base"].vllm_args(), by["chunk32k"].vllm_args(), by["mtp2"].vllm_args()
    assert "--max-num-batched-tokens" not in base
    assert chunk[chunk.index("--max-num-batched-tokens") + 1] == "32768"
    assert base[base.index("--max-num-seqs") + 1] == "768"  # precision args still apply
    assert mtp[mtp.index("--max-num-seqs") + 1] == "256"  # variant overrides precision
    assert '"method": "mtp"' in mtp[mtp.index("--speculative-config") + 1]
    assert all(s.served_model == "Qwen/Qwen3.8-27B-FP8" for s in tune.sessions())


def test_variant_filter(tune):
    from gpubench.cli import filter_config

    only = filter_config(tune, None, None, variants=["base", "kvfp8"])
    assert [s.variant for s in only.sessions()] == ["base", "kvfp8"]
    import typer

    with pytest.raises(typer.BadParameter):
        filter_config(tune, None, None, variants=["nope"])


def test_profiles_do_not_change_the_config_hash(tune):
    assert tune.config_hash() == tune.model_copy(update={"profiles": []}).config_hash()


def _pt(c, ttft_ms):
    raw = {"completed": 8, "failed": 0, "duration": 100, "request_throughput": 0.08,
           "output_throughput": 40, "total_token_throughput": 8000, "median_itl_ms": 20,
           "p95_itl_ms": 30, "p99_itl_ms": 40}
    for m in ("ttft", "tpot", "e2el"):
        raw[f"p95_{m}_ms"] = ttft_ms if m == "ttft" else 20
        raw[f"p99_{m}_ms"] = ttft_ms if m == "ttft" else 20
    wl = Workload(name="w", input_len=100000, output_len=512)
    return build_perf_point([raw], wl, c, 8, 1, SLO(ttft_p99_ms=30000), None, None)


def test_capacity_search_is_exact_at_small_user_counts():
    """Phase 1 finding: 2 passed, 4 failed, and 3 was never measured."""
    sweep = {1: _pt(1, 9000), 2: _pt(2, 17000), 4: _pt(4, 32000)}
    probes = []

    def measure(c):
        probes.append(c)
        return _pt(c, 25000 if c == 3 else 99999)

    users, _ = find_max_users(sweep, measure, CapacitySearch())
    assert probes == [3] and users == 3
    # Large counts keep the 4-user tolerance (no extra probes between 100 and 104).
    big = {c: _pt(c, 1000 if c <= 100 else 99999) for c in (64, 100, 104)}
    assert find_max_users(big, lambda c: 1 / 0, CapacitySearch())[0] == 100


def test_nsys_summary_groups_kernels():
    from gpubench.profile import summarise_kernels

    s = summarise_kernels((FIXTURES / "nsys_kern_sum.csv").read_text())
    groups = {g["group"]: g["pct"] for g in s["groups"]}
    assert groups["FP8/BF16 matmul"] == pytest.approx(0.41)
    assert groups["full attention"] == pytest.approx(0.27)
    assert groups["linear attention (GDN)"] == pytest.approx(0.18)
    assert s["top_kernels"][0]["group"] == "FP8/BF16 matmul"
    assert sum(g["pct"] for g in s["groups"]) == pytest.approx(1.0)


def test_profiled_server_wraps_vllm_in_nsys(tune, tmp_path):
    from gpubench.profile import ProfiledDockerServer

    nsys = tmp_path / "opt/nvidia/nsight-systems/2026.1/target-linux-x64/nsys"
    nsys.parent.mkdir(parents=True)
    nsys.write_text("")
    s = ProfiledDockerServer(tune.sessions()[0], tmp_path / "l", tmp_path, tmp_path,
                             nsys=nsys, report="/work/profile/x")
    cmd = s.docker_cmd()
    assert cmd[cmd.index("--entrypoint") + 1] == "/opt/nsys/target-linux-x64/nsys"
    assert "--capture-range=cudaProfilerApi" in cmd and "--cuda-graph-trace=node" in cmd
    assert cmd[cmd.index("--profiler-config") + 1] == '{"profiler": "cuda"}'
    assert cmd[cmd.index("serve") + 1] == "Qwen/Qwen3.8-27B-FP8"


def test_variant_report(tmp_path):
    from gpubench.sample import sample_runs
    from gpubench.schema import CapacityResult
    from gpubench.variants import report

    base = sample_runs()[1]
    wl = "toolcall-100k-512"
    pts = [_pt(1, 9000).model_copy(update={"workload": wl}),
           _pt(2, 17000).model_copy(update={"workload": wl})]
    cap = CapacityResult(workload=wl, slo=SLO(ttft_p99_ms=30000), max_users_slo=2,
                         max_users_kv_cache=6, output_throughput_at_max_users=40,
                         probed={1: True, 2: True})
    for name, users in [("base", 2), ("chunk32k", 3)]:
        r = base.model_copy(update={"perf": pts, "capacity": [cap.model_copy(
            update={"max_users_slo": users})], "model": base.model.model_copy(
            update={"variant": name})})
        d = tmp_path / name
        d.mkdir()
        (d / "result.json").write_text(r.model_dump_json())
    text = report([tmp_path / "base", tmp_path / "chunk32k"], "base", wl, {"gcp": 6.571})
    assert "| chunk32k | 3 | +1 |" in text and "| base | 2 | – |" in text
    assert "$/1k req (gcp)" in text
