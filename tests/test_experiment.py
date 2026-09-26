"""The Qwen3.8-27B experiment features: checkpoints, per-workload overrides, custom prompts,
sweep cap, fingerprint and verify-host rules, planning from recorded durations."""

import copy
import json
from pathlib import Path

import pytest
import yaml

from gpubench import experiment as ex
from gpubench.config import SLO, Hardware, Workload, load_config
from gpubench.fingerprint import parse_nvidia_smi_xml
from gpubench.perf import bench_serve_args
from gpubench.prompts import fit_to_tokens, sha256_file, trajectory_to_messages
from gpubench.telemetry import parse_samples

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
H100 = Hardware(gpu_type="H100-80GB", provider="gcp", price_per_hour_usd=6.57)


@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "qwen3.8-27b-h100.yaml").with_hardware(H100)


def test_fp8_serves_official_checkpoint(cfg):
    bf16, fp8 = cfg.sessions()
    assert bf16.served_model == "Qwen/Qwen3.8-27B" and not bf16.prequantized
    assert fp8.served_model == "Qwen/Qwen3.8-27B-FP8" and fp8.prequantized
    assert "--quantization" not in fp8.vllm_args()  # read from the checkpoint
    assert "--dtype" in bf16.vllm_args()
    args = fp8.vllm_args()
    assert "--language-model-only" in args and "--no-enable-prefix-caching" in args
    assert args[args.index("--tool-call-parser") + 1] == "qwen3_xml"


def test_per_workload_overrides(cfg):
    perf = cfg.perf
    tc = next(w for w in perf.workloads if w.name == "toolcall-100k-512")
    chat = next(w for w in perf.workloads if w.name == "chat-128-128")
    assert perf.slo_for(tc) == SLO(ttft_p99_ms=30000, itl_median_ms=50)
    assert perf.slo_for(chat) == SLO()
    assert perf.concurrency_for(tc) == [1, 2, 4, 8]
    assert perf.num_prompts(1, tc) == 10 and perf.num_prompts(1, chat) == 50
    assert perf.warmups_for(tc) == 1 and perf.warmups_for(chat) == 10


def test_custom_dataset_bench_args(cfg):
    tc = next(w for w in cfg.perf.workloads if w.dataset == "custom")
    args = bench_serve_args(cfg.sessions()[1], tc, 2, 10, 42, "x.json", 1,
                            raw_dir="/r", datasets_dir="/d")
    assert args[args.index("--model") + 1] == "Qwen/Qwen3.8-27B-FP8"
    assert args[args.index("--dataset-name") + 1] == "custom"
    assert args[args.index("--dataset-path") + 1] == "/d/toolcall-100k.jsonl"
    assert args[args.index("--custom-output-len") + 1] == "512"
    assert {"--skip-chat-template", "--no-oversample", "--ignore-eos"} <= set(args)


def test_custom_workload_needs_hash():
    with pytest.raises(ValueError):
        Workload(name="x", dataset="custom", input_len=10, output_len=1, dataset_path="a.jsonl")


def test_sweep_skips_levels_far_past_kv_limit(cfg):
    chat = next(w for w in cfg.perf.workloads if w.name == "chat-128-128")
    assert cfg.perf.levels(chat, None) == ([1, 4, 16, 32, 64, 128, 256, 512], [])
    run, skipped = cfg.perf.levels(chat, 40)  # 2x40 = 80 users
    assert run == [1, 4, 16, 32, 64] and skipped == [128, 256, 512]
    assert cfg.perf.levels(chat, 0) == ([1, 4, 16, 32, 64, 128, 256, 512], [])
    tc = next(w for w in cfg.perf.workloads if w.dataset == "custom")
    assert cfg.perf.levels(tc, 1) == ([1, 2], [4, 8])  # keeps >= 1 level


def test_trajectory_becomes_tool_calls():
    traj = [
        {"role": "system", "system_prompt": "COMMANDS: open, edit", "text": None},
        {"role": "user", "text": "ISSUE: fix the bug"},
        {"role": "ai", "text": "Let's look.\n```\nopen foo.py\n```"},
        {"role": "user", "text": "[file contents]"},
        {"role": "ai", "text": "Done."},
    ]
    docs, msgs = trajectory_to_messages(traj)
    assert docs == "COMMANDS: open, edit"
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool", "assistant"]
    call = msgs[1]["tool_calls"][0]["function"]
    assert call == {"name": "run_command", "arguments": {"command": "open foo.py"}}
    assert msgs[1]["content"] == "Let's look."
    assert msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]


def test_fit_to_tokens_is_exact_and_ends_on_tool_result():
    # A 1-char = 1-token toy renderer.
    render = lambda ms: "".join(f"<{m['role']}>{m['content']}" for m in ms)  # noqa: E731
    count = len
    msgs = [{"role": "user", "content": "task"}]
    for i in range(20):
        msgs += [{"role": "assistant", "content": f"call{i}"},
                 {"role": "tool", "content": "x" * 50}]
    msgs.append({"role": "assistant", "content": "trailing"})
    text = fit_to_tokens(render, count, msgs, 500)
    assert text is not None and len(text) == 500
    assert text.startswith("<user>task") and text.rstrip("x").endswith("<tool>")
    assert fit_to_tokens(render, count, msgs[:3], 500) is None  # too short


def test_parse_nvidia_smi_xml_drops_serials():
    info = parse_nvidia_smi_xml((FIXTURES / "nvidia_smi_h100_sxm.xml").read_text())
    g = info["gpus"][0]
    assert info["driver"] == "580.173.02" and info["count"] == 1
    assert g["name"] == "NVIDIA H100 80GB HBM3" and g["pci_device_id"] == "233010DE"
    assert g["memory_total_mib"] == 81559 and g["default_power_limit_w"] == 700
    assert g["ecc_mode"] == "Enabled" and g["mig_mode"] == "Disabled"
    assert g["active_clock_limits"] == []  # gpu_idle is not a limit
    assert "serial" not in json.dumps(info) and "GPU-1111" not in json.dumps(info)


def _fingerprint():
    return {
        "cloud": {"provider": "gcp", "machine_type": "a3-highgpu-1g", "zone": "us-central1-a"},
        "host": {"cpu_model": "Intel Xeon Platinum 8481C", "vcpus": 26, "ram_gib": 234.0},
        "gpu": parse_nvidia_smi_xml((FIXTURES / "nvidia_smi_h100_sxm.xml").read_text()),
        "measured": {"hbm_copy_gbs": 2900.0, "bf16_tflops": 700.0, "fp8_tflops": 1300.0,
                     "h2d_gbs": 50.0, "d2h_gbs": 50.0, "hf_download_mbps": 400.0},
        "conditions": {"after_load_temperature_c": [60.0], "after_load_clock_limits": [[]]},
    }


EXP = ex.Experiment.model_validate(yaml.safe_load("""
id: 2026-10-qwen3.8-27b-h100
title: t
status: published
bench: {tag: exp-2026-10-qwen3.8-27b-h100, config: configs/qwen3.8-27b-h100.yaml}
environments:
  - {provider: gcp, machine_type: a3-highgpu-1g, zone: us-central1-a, provisioning: spot}
"""))


def test_verify_host_passes_on_same_hardware():
    ref, cur = _fingerprint(), _fingerprint()
    cur["measured"]["bf16_tflops"] = 690.0  # -1.4%: within tolerance
    checks = ex.check_environment(EXP, cur["cloud"]) + ex.compare_fingerprints(ref, cur)
    assert ex.verdict(checks), ex.format_checks(checks)


def test_verify_host_fails_on_pcie_variant_and_slow_memory():
    ref, cur = _fingerprint(), copy.deepcopy(_fingerprint())
    cur["gpu"]["gpus"][0]["pci_device_id"] = "233110DE"  # H100 PCIe
    cur["measured"]["hbm_copy_gbs"] = 1800.0
    failed = {c.field for c in ex.compare_fingerprints(ref, cur) if c.status == "FAIL"}
    assert failed == {"gpu.pci_device_id", "measured.hbm_copy_gbs"}


def test_verify_host_refuses_other_providers():
    checks = ex.check_environment(EXP, {"provider": "runpod"})
    assert not ex.verdict(checks) and "only run on gcp" in checks[0].note
    other_zone = ex.check_environment(EXP, {"provider": "gcp", "machine_type": "a3-highgpu-1g",
                                            "zone": "us-east5-a"})
    assert ex.verdict(other_zone) and other_zone[-1].status == "WARN"


def test_plan_from_recorded_durations():
    from gpubench.sample import sample_runs

    run = sample_runs()[0]
    hours = ex.recorded_hours([run])[run.run_id]
    expected = (sum(p.duration_s * p.repeats + 15 * p.repeats for p in run.perf) + 300 + 120)
    assert hours == pytest.approx(expected / 3600)


def test_telemetry_reads_temperature_and_thermal_throttle():
    csv = ("2026/09/24 12:00:00.000, 0, 650.5, 70000, 98, 1980, 61, 2619, 0x0000000000000004\n"
           "2026/09/24 12:00:00.500, 0, 700.5, 72000, 99, 1755, 83, 2619, 0x0000000000000024\n")
    s = parse_samples(csv)
    assert s.max_temp_c == 83 and s.thermal_throttle_fraction == 0.5  # 0x4 = power cap only


def test_staged_dataset_is_hash_checked(tmp_path):
    import hashlib

    from gpubench.prompts import stage_dataset

    f = tmp_path / "p.jsonl"
    f.write_text('{"prompt": "a"}\n')
    digest = hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha256_file(f) == digest
    wl = Workload(name="x", dataset="custom", input_len=1, output_len=1,
                  dataset_path=str(f), sha256=digest)
    assert stage_dataset(wl, tmp_path / "run" / "datasets").read_bytes() == f.read_bytes()
    bad = wl.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(ValueError, match="sha256"):
        stage_dataset(bad, tmp_path / "other")


def test_sanitized_bundle_drops_private_state(tmp_path):
    import io
    import tarfile

    from gpubench.publish import pack_run, sanitize_bundle

    bundle = tmp_path / "code.tgz"
    with tarfile.open(bundle, "w:gz") as tar:
        for name, body in [("./bench/gpubench/cli.py", b"code"),
                           ("./infra/terraform/base/terraform.tfvars", b"billing = 1"),
                           ("./infra/terraform/envs/x/terraform.tfstate", b"{}")]:
            info = tarfile.TarInfo(name)
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    out, removed = sanitize_bundle(bundle, tmp_path / "out")
    assert sorted(removed) == ["./infra/terraform/base/terraform.tfvars",
                               "./infra/terraform/envs/x/terraform.tfstate"]
    with tarfile.open(out) as tar:
        names = tar.getnames()
        manifest = tar.extractfile("MANIFEST.sha256").read().decode()
    assert "./bench/gpubench/cli.py" in names and not any("terraform.tf" in n for n in names)
    assert "./bench/gpubench/cli.py" in manifest and "terraform.tfvars" in manifest  # noted

    run = tmp_path / "20260101T000000Z_x"
    (run / "accuracy").mkdir(parents=True)
    (run / "result.json").write_text("{}")
    (run / "accuracy" / "samples_x.jsonl").write_text("{}\n")
    with tarfile.open(pack_run(run, tmp_path / "o2")) as tar:
        assert f"{run.name}/accuracy/samples_x.jsonl" in tar.getnames()  # nothing left out


def test_per_precision_max_num_seqs_caps_the_sweep(cfg):
    """Smoke run finding: hybrid models need max-num-seqs <= Mamba state blocks."""
    bf16, fp8 = cfg.sessions()
    assert bf16.max_num_seqs == 320 and fp8.max_num_seqs == 768
    a = bf16.vllm_args()
    assert a[a.index("--max-num-seqs") + 1] == "320"
    chat = next(w for w in cfg.perf.workloads if w.name == "chat-128-128")
    # KV would allow ~976 chat requests, but only 320 run at once: 2x320 caps the sweep.
    run, skipped = cfg.perf.levels(chat, min(976, bf16.max_num_seqs))
    assert run[-1] == 512 and skipped == []
    run, skipped = cfg.perf.levels(chat, min(976, 200))
    assert skipped == [512]


def test_resume_retries_failed_workloads_of_a_finished_run(cfg):
    """Smoke run finding: a workload that failed must be retried by --resume."""
    from gpubench.quality import QualityResult
    from gpubench.runner import missing_work
    from gpubench.sample import sample_runs

    s = cfg.sessions()[0]
    run = sample_runs()[0].model_copy(update={"complete": True})
    names = [w.name for w in cfg.perf.workloads]
    have = [c.model_copy(update={"workload": n}) for c, n in zip(run.capacity * 2, names[1:],
                                                                 strict=False)]
    q = QualityResult(workload="toolcall-100k-512", dataset_sha256="x", responses="q")
    partial = run.model_copy(update={"capacity": have, "quality": [q]})
    assert missing_work(s, partial) == ["toolcall-100k-512"]
    assert missing_work(s, partial.model_copy(update={"quality": []})) == [
        "toolcall-100k-512", "toolcall-100k-512 quality"]
