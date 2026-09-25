import json
from pathlib import Path

import pytest

from gpubench.accuracy import parse_lm_eval_results
from gpubench.capacity import find_max_users, kv_cache_max_users
from gpubench.config import SLO, AccuracyTask, CapacitySearch, Hardware, Workload, load_config
from gpubench.dataset import build_site_data
from gpubench.perf import bench_serve_args, build_perf_point
from gpubench.schema import RunResult
from gpubench.server import parse_kv_cache_tokens
from gpubench.telemetry import parse_samples

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
WL = Workload(name="chat-128-128", input_len=128, output_len=128)


H100 = Hardware(gpu_type="H100-80GB", provider="gcp", price_per_hour_usd=6.57)


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "qwen3-8b.yaml").with_hardware(H100)


def raw(**overrides):
    d = json.loads((FIXTURES / "bench_serve_c16.json").read_text())
    d.update(overrides)
    return d


@pytest.mark.parametrize("name", ["qwen3-8b", "qwen3-14b", "qwen3-8b-quick", "smoke"])
def test_configs_load(name):
    cfg = load_config(ROOT / "configs" / f"{name}.yaml")
    assert cfg.sessions()


def test_sessions_and_vllm_args(cfg):
    sessions = cfg.sessions()
    assert [s.session_id for s in sessions] == [
        "h100-80gbx1_qwen3-8b_bf16", "h100-80gbx1_qwen3-8b_fp8",
    ]
    bf16, fp8 = (s.vllm_args() for s in sessions)
    assert "--dtype" in bf16 and "bfloat16" in bf16
    assert fp8[fp8.index("--quantization") + 1] == "fp8"
    i = bf16.index("--default-chat-template-kwargs")
    assert json.loads(bf16[i + 1]) == {"enable_thinking": False}
    assert bf16[bf16.index("--tensor-parallel-size") + 1] == "1"


def test_bench_args_request_only_p95_p99(cfg):
    args = bench_serve_args(cfg.sessions()[0], WL, 16, 80, 42, "x.json", 10)
    assert args[args.index("--metric-percentiles") + 1] == "95,99"
    assert args[args.index("--max-concurrency") + 1] == "16"
    assert "--ignore-eos" in args


def test_build_perf_point_takes_median_of_repeats():
    raws = [raw(output_throughput=t, median_itl_ms=itl)
            for t, itl in [(800, 17), (820, 18), (900, 30)]]
    p = build_perf_point(raws, WL, 16, 80, gpu_count=1, slo=SLO(),
                         telemetry=None, price_per_hour_usd=3.6)
    assert p.output_throughput == 820
    assert p.itl_ms.median == 18
    assert p.tokens_per_s_per_user == pytest.approx(1000 / 18)
    assert p.ttft_ms.p99 == 120 and p.ttft_ms.p95 == 90
    # $3.6/h at 820 tok/s -> 3.6 / (820*3600) * 1e6
    assert p.usd_per_1m_output_tokens == pytest.approx(3.6 / (820 * 3600) * 1e6)
    assert p.slo_pass


def test_slo_fails_on_itl():
    p = build_perf_point([raw(median_itl_ms=80)], WL, 16, 80, 1, SLO(), None, None)
    assert not p.slo_pass


def _point(c, passes):
    return build_perf_point([raw(median_itl_ms=10 if passes else 99)], WL, c, 80, 1, SLO(),
                            None, None)


def test_find_max_users_bisects():
    true_max = 90
    sweep = {c: _point(c, c <= true_max) for c in [1, 16, 64, 128, 256]}
    probes = []

    def measure(c):
        probes.append(c)
        return _point(c, c <= true_max)

    max_users, points = find_max_users(sweep, measure, CapacitySearch(resolution=4))
    assert true_max - 4 <= max_users <= true_max
    assert all(64 < c < 128 for c in probes)
    assert set(sweep) <= set(points)


def test_find_max_users_none_pass_and_search_disabled():
    none_pass = {c: _point(c, False) for c in [1, 16]}
    assert find_max_users(none_pass, lambda c: None, CapacitySearch())[0] == 0
    all_pass = {c: _point(c, True) for c in [1, 16]}
    assert find_max_users(all_pass, lambda c: None, CapacitySearch(enabled=False))[0] == 16


def test_find_max_users_extends_past_sweep_top():
    """Sweep tops out at 512 and passes (the real Qwen3-8B chat case): probe upward."""
    true_max = 600
    sweep = {c: _point(c, True) for c in [1, 64, 512]}
    probes = []

    def measure(c):
        probes.append(c)
        return _point(c, c <= true_max)

    max_users, _ = find_max_users(sweep, measure, CapacitySearch(resolution=4), ceiling=2944)
    assert probes[0] == 1024
    assert true_max - 4 <= max_users <= true_max


def test_find_max_users_stops_at_ceiling():
    sweep = {c: _point(c, True) for c in [1, 16]}
    max_users, points = find_max_users(sweep, lambda c: _point(c, True), CapacitySearch(),
                                       ceiling=40)
    assert max_users == 40 and max(points) == 40


def test_kv_cache_parsing_and_users():
    log = ("INFO 09-24 [kv_cache_utils.py:2396] GPU KV cache size: 456,320 tokens, "
           "Maximum concurrency for 16,384 tokens per request: 27.85x")
    tokens = parse_kv_cache_tokens(log)
    assert tokens == 456320
    assert kv_cache_max_users(tokens, WL) == 456320 // 256
    assert parse_kv_cache_tokens("no such line") is None


def test_telemetry_parse():
    csv = ("2026/09/24 12:00:00.000, 0, 650.5, 70000, 98, 1980\n"
           "2026/09/24 12:00:00.500, 0, 700.5, 72000, 99, 1980\n"
           "2026/09/24 12:00:01.000, 0, [N/A], 72000, 99, 1980\n")
    s = parse_samples(csv)
    assert s.avg_power_w == pytest.approx(675.5)
    assert s.peak_memory_gb == pytest.approx(72000 / 1024)


def test_lm_eval_parse():
    results = json.loads((FIXTURES / "lm_eval_gsm8k.json").read_text())
    r = parse_lm_eval_results(results, AccuracyTask(name="gsm8k"))
    assert (r.metric, r.value, r.num_samples) == ("exact_match", 0.88, 1319)


def test_sample_data_validates_and_real_runs_replace_samples(tmp_path):
    from gpubench.sample import sample_runs

    samples = sample_runs()
    assert samples and all(r.sample for r in samples)
    RunResult.model_validate_json(samples[0].model_dump_json())

    build_site_data(samples, tmp_path)
    assert len(json.loads((tmp_path / "index.json").read_text())) == len(samples)

    real = samples[0].model_copy(update={"sample": False, "run_id": "real"})
    build_site_data([*samples, real], tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [r["run_id"] for r in index] == ["real"]
    assert [p.name for p in (tmp_path / "runs").iterdir()] == ["real.json"]


def test_power_averaged_only_over_benchmark_window():
    """Real H100 trace: ~11 s near-idle client startup, then ~5 s of load at ~470 W."""
    from datetime import datetime, timedelta

    csv = (FIXTURES / "h100_telemetry_c16.csv").read_text()
    whole = parse_samples(csv)
    end = datetime(2026, 9, 24, 12, 33, 21, 600000)
    windowed = parse_samples(csv, [(end - timedelta(seconds=4.8), end)])
    assert whole.avg_power_w < 250
    assert windowed.avg_power_w > 400
    assert windowed.peak_memory_gb == whole.peak_memory_gb


def test_smoke_runs_never_published(tmp_path):
    from gpubench.sample import sample_runs

    smoke = sample_runs()[0].model_copy(update={"sample": False, "campaign": "smoke"})
    build_site_data([smoke], tmp_path)
    assert json.loads((tmp_path / "index.json").read_text()) == []


def test_filter_config_narrows_precision_and_workload(cfg):
    import typer

    from gpubench.cli import filter_config

    fp8 = filter_config(cfg, ["fp8"], None)
    assert [s.session_id for s in fp8.sessions()] == ["h100-80gbx1_qwen3-8b_fp8"]
    assert fp8.config_hash() != cfg.config_hash()
    assert cfg.models[0].precisions == ["bf16", "fp8"]  # original untouched

    chat = filter_config(cfg, ["bf16"], ["chat-128-128"], skip_accuracy=True)
    assert [w.name for w in chat.perf.workloads] == ["chat-128-128"]
    assert not chat.accuracy.enabled

    with pytest.raises(typer.BadParameter):
        filter_config(cfg, None, ["nope"])
    with pytest.raises(typer.BadParameter):
        filter_config(cfg, ["fp16"], None)


def test_accuracy_only_rerun_merges_into_perf_run(tmp_path):
    from datetime import timedelta

    from gpubench.sample import sample_runs
    from gpubench.schema import AccuracyResult

    perf_run = sample_runs()[0].model_copy(update={"sample": False, "run_id": "perf"})
    acc_run = perf_run.model_copy(update={
        "run_id": "acc", "perf": [], "capacity": [],
        "created_at": perf_run.created_at + timedelta(hours=2),
        "accuracy": [AccuracyResult(task="gsm8k", metric="exact_match", value=0.5),
                     AccuracyResult(task="new_task", metric="acc", value=0.7)],
    })
    build_site_data([perf_run, acc_run], tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert [r["run_id"] for r in index] == ["perf"]
    merged = RunResult.model_validate_json((tmp_path / "runs" / "perf.json").read_text())
    assert merged.perf == perf_run.perf and merged.merged_from == ["acc"]
    scores = {a.task: a.value for a in merged.accuracy}
    assert scores["gsm8k"] == 0.5 and scores["new_task"] == 0.7  # later run wins / added
    assert scores["ifeval"] == 0.83  # untouched tasks kept


def test_skip_perf_filter(cfg):
    from gpubench.cli import filter_config

    acc_only = filter_config(cfg, ["bf16"], None, skip_perf=True)
    assert acc_only.perf.workloads == [] and acc_only.accuracy.enabled


def test_arc_rescored_from_samples():
    """Real Qwen3-8B output: 'The best answer is C' must match target 'C'."""
    from gpubench.accuracy import _arc_letter, rescore_samples

    assert _arc_letter("The best answer is C") == "C"
    assert _arc_letter("The best answer is **B**") == "B"
    assert _arc_letter("D") == "D"
    assert _arc_letter("To find the average speed, we use the formula:") is None
    r = rescore_samples(AccuracyTask(name="arc_challenge_chat"),
                        FIXTURES / "samples_arc_challenge_chat_sample.jsonl")
    assert r.metric == "exact_match_letter" and r.num_samples == 40
    assert r.value > 0.7


def test_failed_probe_keeps_lower_bound():
    """Real FP8 case: all sweep levels pass, the 1024-user probe crashes (fd limit)."""
    sweep = {c: _point(c, True) for c in [1, 64, 512]}

    def measure(c):
        raise RuntimeError("Too many open files")

    max_users, points = find_max_users(sweep, measure, CapacitySearch(), ceiling=2944)
    assert max_users == 512 and set(points) == {1, 64, 512}


def test_chat_only_rerun_merges_per_workload(tmp_path):
    from datetime import timedelta

    from gpubench.sample import sample_runs

    full = sample_runs()[0].model_copy(update={"sample": False, "run_id": "full"})
    chat_pts = [p.model_copy(update={"output_throughput": 1.0})
                for p in full.perf if p.workload == "chat-128-128"]
    chat_cap = [c.model_copy(update={"max_users_slo": 777})
                for c in full.capacity if c.workload == "chat-128-128"]
    rerun = full.model_copy(update={
        "run_id": "chat", "perf": chat_pts, "capacity": chat_cap, "accuracy": [],
        "created_at": full.created_at + timedelta(hours=9),
    })
    build_site_data([full, rerun], tmp_path)
    m = RunResult.model_validate_json((tmp_path / "runs" / "full.json").read_text())
    assert {p.workload for p in m.perf} == {p.workload for p in full.perf}
    assert all(p.output_throughput == 1.0 for p in m.perf if p.workload == "chat-128-128")
    caps = {c.workload: c.max_users_slo for c in m.capacity}
    assert caps["chat-128-128"] == 777 and len(caps) == len(full.capacity)
    assert m.accuracy == full.accuracy and m.merged_from == ["chat"]


def test_missing_capacity_derived_from_sweep(tmp_path):
    """Real FP8 case: the chat capacity step crashed, leaving sweep points but no capacity."""
    from gpubench.sample import sample_runs

    full = sample_runs()[0].model_copy(update={"sample": False, "run_id": "fp8"})
    lost = full.model_copy(update={
        "capacity": [c for c in full.capacity if c.workload != "chat-128-128"],
    })
    build_site_data([lost], tmp_path)
    m = RunResult.model_validate_json((tmp_path / "runs" / "fp8.json").read_text())
    names = [c.workload for c in m.capacity]
    assert names == [c.workload for c in lost.capacity] + ["chat-128-128"]

    chat_pts = [p for p in full.perf if p.workload == "chat-128-128"]
    sweep = sorted(chat_pts, key=lambda p: p.concurrency)
    expected = 0
    for p in sweep:  # highest level before the first failure
        if not p.slo_pass:
            break
        expected = p.concurrency
    chat = next(c for c in m.capacity if c.workload == "chat-128-128")
    orig = next(c for c in full.capacity if c.workload == "chat-128-128")
    assert chat.max_users_slo == expected
    assert chat.max_users_kv_cache == orig.max_users_kv_cache
    assert chat.probed == {p.concurrency: p.slo_pass for p in sweep}
