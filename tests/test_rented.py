"""Rented-box features: hardware detection, runtimes, resume, submit, cost model, v1 compat."""

import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from gpubench.config import Hardware, load_config
from gpubench.cost import ApiOffer, GpuOffer, build_cost
from gpubench.dataset import check_result, check_run_dir, tables, write_tables
from gpubench.hardware import normalise_gpu_name, parse_gpu_query
from gpubench.runner import find_previous, prepare_resume
from gpubench.schema import RunResult
from gpubench.server import DockerServer, NativeServer
from gpubench.submit import pack_raw, pr_body, stage_runs

ROOT = Path(__file__).resolve().parents[1]
H100 = Hardware(gpu_type="H100-80GB", provider="runpod", price_per_hour_usd=2.69)


def real_run(**update) -> RunResult:
    from gpubench.sample import sample_runs

    run = sample_runs()[1]  # Qwen3-8B fp8
    return run.model_copy(update={"sample": False, "run_id": "r1", "git_commit": "abc123",
                                  "hardware": H100, **update})


@pytest.mark.parametrize("name,mib,expected", [
    ("NVIDIA H100 80GB HBM3", 81559, "H100-80GB"),
    ("NVIDIA H100 PCIe", 81559, "H100-PCIe-80GB"),
    ("NVIDIA H100 NVL", 95830, "H100-NVL-94GB"),
    ("NVIDIA H200", 143771, "H200-141GB"),
    ("NVIDIA GH200 480GB", 97871, "GH200-96GB"),
    ("NVIDIA A100-SXM4-80GB", 81920, "A100-80GB"),
    ("NVIDIA A100-PCIE-40GB", 40960, "A100-PCIe-40GB"),
    ("NVIDIA L40S", 46068, "L40S-48GB"),
    ("NVIDIA L4", 23034, "L4-24GB"),
    ("NVIDIA GeForce RTX 4090", 24564, "RTX4090-24GB"),
    ("NVIDIA B200", 183359, "B200-180GB"),
])
def test_gpu_names_normalised(name, mib, expected):
    assert normalise_gpu_name(name, mib) == expected


def test_parse_gpu_query():
    out = "NVIDIA H100 80GB HBM3, 81559\nNVIDIA H100 80GB HBM3, 81559\n"
    assert parse_gpu_query(out) == [("NVIDIA H100 80GB HBM3", 81559.0)] * 2


def test_campaign_configs_are_hardware_agnostic():
    for path in (ROOT / "configs").glob("*.yaml"):
        cfg = load_config(path)
        assert cfg.hardware is None, path
        assert cfg.engine.version == "0.30.0"


def test_v1_result_loads_with_provider_from_cloud():
    v1 = json.loads(real_run().model_dump_json())
    v1["schema_version"] = 1
    v1["hardware"] = {"gpu_type": "H100-80GB", "gpu_count": 1, "node_count": 1,
                      "machine_type": "a3-highgpu-1g", "cloud": "gcp", "zone": "us-central1-a",
                      "provisioning": "spot", "price_per_hour_usd": 6.57}
    del v1["complete"]
    del v1["software"]["runtime"]
    r = RunResult.model_validate(v1)
    assert r.hardware.provider == "gcp" and r.complete and r.software.runtime is None


def _session(tmp_path):
    cfg = load_config(ROOT / "configs" / "smoke.yaml").with_hardware(H100)
    return cfg.sessions()[0]


def test_docker_and_native_commands(tmp_path):
    s = _session(tmp_path)
    docker = DockerServer(s, tmp_path / "v.log", tmp_path / "hf", tmp_path)
    cmd = docker.docker_cmd()
    assert cmd[cmd.index("--ulimit") + 1] == "nofile=65535:65535"
    assert "vllm/vllm-openai:v0.30.0" in cmd
    assert cmd[-2:] == ["--host", "127.0.0.1"]
    assert docker.raw_dir == "/work/raw"

    native = NativeServer(s, tmp_path / "v.log", tmp_path / "hf", tmp_path,
                          venv=tmp_path / "venv")
    cmd = native.serve_cmd()
    assert cmd[0] == str(tmp_path / "venv/bin/vllm") and cmd[1:3] == ["serve", "Qwen/Qwen3-8B"]
    assert cmd[-2:] == ["--host", "127.0.0.1"]
    assert native.raw_dir == str((tmp_path / "raw").resolve())
    assert native._env()["HF_HOME"] == str(tmp_path / "hf")


def test_resume_finds_matching_run_and_drops_partial_workload(tmp_path):
    s = _session(tmp_path)
    run = real_run(run_id=f"20260925T000000Z_{s.session_id}", config_hash=s.config.config_hash(),
                   complete=False)
    run_dir = tmp_path / run.run_id
    run_dir.mkdir()
    # Workload "decode" was cut off: points but no capacity entry.
    partial = run.model_copy(update={
        "capacity": [c for c in run.capacity if c.workload != "decode-128-1024"]})
    (run_dir / "result.json").write_text(partial.model_dump_json())
    found_dir, found = find_previous(tmp_path, s)
    assert found_dir == run_dir
    resumed = prepare_resume(found)
    assert "decode-128-1024" not in {p.workload for p in resumed.perf}
    assert {p.workload for p in resumed.perf} == {c.workload for c in resumed.capacity}

    # A different config (e.g. another price) never resumes this run.
    other = load_config(ROOT / "configs" / "smoke.yaml").with_hardware(
        H100.model_copy(update={"price_per_hour_usd": 9.99})).sessions()[0]
    assert find_previous(tmp_path, other) is None


def test_check_result_rules():
    assert check_result(real_run()) == []
    assert any("sample" in p for p in check_result(real_run(sample=True)))
    assert any("smoke" in p for p in check_result(real_run(campaign="smoke")))
    assert any("finish" in p for p in check_result(real_run(complete=False)))
    assert check_result(real_run(complete=False), allow_incomplete=True) == []
    assert any("provenance" in p for p in check_result(real_run(git_commit="abc-dirty")))
    unknown = H100.model_copy(update={"provider": "unknown"})
    assert any("provider" in p for p in check_result(real_run(hardware=unknown)))


def test_submission_staging(tmp_path):
    run = real_run(run_id="20260926T000000Z_h100-80gbx1_qwen3-8b_fp8")
    src = tmp_path / "results" / run.run_id
    (src / "raw").mkdir(parents=True)
    (src / "result.json").write_text(run.model_dump_json())
    (src / "raw" / "chat_c1_r0.json").write_text("{}")
    (src / "vllm.log").write_text("log")
    (src / "accuracy" / "gsm8k").mkdir(parents=True)
    (src / "accuracy" / "gsm8k" / "samples_gsm8k.jsonl").write_text("{}\n" * 1000)
    (src / "accuracy" / "gsm8k" / "results_x.json").write_text("{}")

    names = tarfile.open(fileobj=__import__("io").BytesIO(pack_raw(src))).getnames()
    assert "raw/chat_c1_r0.json" in names and "vllm.log" in names
    assert "accuracy/gsm8k/results_x.json" in names
    assert not any(n.endswith(".jsonl") or n == "result.json" for n in names)

    stage = tmp_path / "stage"
    stage_runs([src], stage, allow_incomplete=False)
    assert check_run_dir(stage / "runs" / run.run_id) == []
    assert "runpod" in pr_body([run]) and run.run_id in pr_body([run])


def test_cost_model_and_tables(tmp_path):
    run = real_run()
    gpu = [GpuOffer(provider="lambda", gpu_type="H100-80GB", usd_per_gpu_hour=3.0,
                    source="x", checked_at="2026-09-25"),
           GpuOffer(provider="other", gpu_type="A100-80GB", usd_per_gpu_hour=1.0,
                    source="x", checked_at="2026-09-25")]
    api = [ApiOffer(model="Qwen/Qwen3-8B", provider="cheap", usd_per_1m_input=0.05,
                    usd_per_1m_output=0.2, source="x", fetched_at="2026-09-25")]
    cost = build_cost([run], gpu, api)
    dep = next(d for d in cost["deployments"] if d["workload"] == "chat-128-128")
    # A100 offer does not apply to an H100 benchmark; the run's own price is included.
    assert {o["provider"] for o in dep["offers"]} == {"lambda", "runpod"}
    lam = next(o for o in dep["offers"] if o["provider"] == "lambda")
    assert lam["usd_per_1m_output_tokens"] == pytest.approx(
        3.0 / 3600 / dep["output_tokens_per_s"] * 1e6)
    assert lam["usd_per_1k_requests"] == pytest.approx(3.0 / 3600 / dep["requests_per_s"] * 1e3)

    comp = next(c for c in cost["comparisons"] if c["workload"] == "chat-128-128")
    assert comp["hosted"]["provider"] == "runpod"  # $2.69 < $3.00
    api_per_req = (128 * 0.05 + 128 * 0.2) / 1e6
    assert comp["api_cheapest"]["usd_per_1k_requests"] == pytest.approx(api_per_req * 1e3)
    assert comp["breakeven_requests_per_day"] == pytest.approx(2.69 * 24 / api_per_req)

    write_tables(tables([run], cost), tmp_path)
    import sqlite3

    db = sqlite3.connect(tmp_path / "silybench.sqlite")
    n = db.execute("select count(*) from hosted_costs where gpu_type='H100-80GB'").fetchone()[0]
    assert n == 2 * len(cost["deployments"])
    assert (tmp_path / "csv" / "comparisons.csv").read_text().startswith("model,workload")


def test_plan_estimate_matches_measured_runs():
    """Calibrated on the Qwen3-8B H100 BF16 run: 7.85 h of measured perf time."""
    from gpubench.plan import estimate

    cfg = load_config(ROOT / "configs" / "qwen3-8b.yaml").with_hardware(H100)
    from gpubench.cli import filter_config

    bf16 = filter_config(cfg, ["bf16"], None, skip_accuracy=True)
    est = estimate(bf16)
    assert 5.0 < est.perf_hours < 11.0


def test_price_files_parse():
    data = yaml.safe_load("offers:\n- {provider: a, gpu_type: H100-80GB, usd_per_gpu_hour: 2,"
                          " source: s, checked_at: 2026-09-25}\n")
    assert GpuOffer.model_validate(data["offers"][0]).checked_at.year == 2026
    assert datetime.now(UTC)
