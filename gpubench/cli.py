"""gpubench CLI. On a rented GPU box (see AGENTS.md):

  gpubench doctor                                   # is this machine ready?
  gpubench plan configs/qwen3-8b.yaml --price-per-hour 2.69   # hours and $ before starting
  gpubench run configs/qwen3-8b.yaml --provider runpod --price-per-hour 2.69 --detach --resume
  gpubench status                                   # where is the detached run?
  gpubench submit results/*/                        # open a PR to the data repo

Maintainers: `gpubench dataset build` (run by the data repo's CI), `aggregate` (GCS -> local).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import typer

from gpubench.config import BenchConfig, load_config
from gpubench.dataset import dataset_app
from gpubench.schema import RunResult

app = typer.Typer(add_completion=False, no_args_is_help=True)
app.add_typer(dataset_app, name="dataset")
prompts_app = typer.Typer(no_args_is_help=True, help="Build prompt datasets for custom workloads.")
app.add_typer(prompts_app, name="prompts")


quality_app = typer.Typer(no_args_is_help=True,
                          help="Correctness of custom long-context scenarios.")
app.add_typer(quality_app, name="quality")


@quality_app.command("make")
def quality_make(
    model: str = typer.Option(..., help="HF model whose tokenizer and chat template to use"),
    tokens: int = typer.Option(100_000, help="Context length of the perf prompts"),
    count: int = typer.Option(100, help="Recall questions and drift prompts (each)"),
    out: Path = typer.Option(Path("datasets/toolcall-100k-quality.jsonl")),
    shards: int = typer.Option(4),
    seed: int = typer.Option(42, help="Must match the perf dataset's seed"),
) -> None:
    """Build recall + drift items from the same contexts as the tool-calling perf prompts."""
    from gpubench.quality import build_quality_dataset

    build_quality_dataset(model, tokens, count, out, seed=seed, shards=shards, echo=typer.echo)


@quality_app.command("run")
def quality_run(
    config: Path,
    workload: str = typer.Option(..., help="Workload whose quality suite to run"),
    base_url: str = typer.Option("http://localhost:8000", help="A running OpenAI-compatible "
                                 "server, e.g. your optimized build"),
    model: str = typer.Option(..., help="Served model name"),
    out: Path = typer.Option(Path("quality-runs/candidate"), help="Where responses go"),
    concurrency: int = typer.Option(1),
) -> None:
    """Run a quality suite against any server (the before/after loop of an optimization)."""
    from gpubench.prompts import ensure_file
    from gpubench.quality import run_quality

    cfg = load_config(config)
    wl = next((w for w in cfg.perf.workloads if w.name == workload), None)
    if wl is None or wl.quality is None:
        raise typer.BadParameter(f"{workload} has no quality suite in {config}")
    dataset = ensure_file(wl.quality.dataset_path, wl.quality.dataset_url, wl.quality.sha256,
                          f"{workload}-quality")
    result = run_quality(base_url, model, workload, dataset, out, concurrency,
                         wl.quality.recall_max_tokens, wl.quality.drift_max_tokens,
                         echo=typer.echo)
    (out / "quality.json").write_text(result.model_dump_json(indent=2))


@quality_app.command("compare")
def quality_compare(
    baseline: Path = typer.Argument(..., help="Run dir (or dir with quality/) of the baseline"),
    candidate: Path = typer.Argument(..., help="Run dir of the optimized build"),
    workload: str = typer.Option("toolcall-100k-512"),
) -> None:
    """Is the optimized build as correct as the baseline? Exit 1 if not."""
    from gpubench.quality import compare_responses, load_items

    def responses(d: Path) -> list[dict]:
        f = d / "quality" / f"{workload}.jsonl"
        if not f.exists():
            raise typer.BadParameter(f"{f} not found")
        return load_items(f)

    report = compare_responses(responses(baseline), responses(candidate))
    for k in ("same_tool_call", "mean_kl", "top1_agreement", "median_divergence_token",
              "recall_baseline", "recall_candidate"):
        v = report[k]
        typer.echo(f"  {k:24s} {v:.4f}" if isinstance(v, float) else f"  {k:24s} {v}")
    for k, ok in report["checks"].items():
        typer.echo(f"{'✓' if ok else '✗'} {k}")
    typer.echo("PASS" if report["pass"] else "FAIL")
    if not report["pass"]:
        raise typer.Exit(1)


@prompts_app.command("make-toolcall")
def make_toolcall(
    model: str = typer.Option(..., help="HF model whose tokenizer and chat template to use"),
    tokens: int = typer.Option(100_000, help="Exact prompt length in tokens"),
    count: int = typer.Option(200, help="Number of prompts"),
    out: Path = typer.Option(Path("datasets/toolcall-100k.jsonl")),
    shards: int = typer.Option(2, help="Source parquet shards to read (of 12, ~440 MB each)"),
    seed: int = typer.Option(42),
) -> None:
    """Build the long-context tool-calling prompts from real SWE-agent sessions."""
    from gpubench.prompts import build_toolcall_dataset

    build_toolcall_dataset(model, tokens, count, out, seed=seed, shards=shards, echo=typer.echo)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_REPO = "tgarg01/silybench-data"


@app.callback()
def main(verbose: bool = typer.Option(False, "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def filter_config(
    cfg: BenchConfig,
    precisions: list[str] | None,
    workloads: list[str] | None,
    skip_accuracy: bool = False,
    skip_perf: bool = False,
    skip_workloads: list[str] | None = None,
    quality_only: bool = False,
    variants: list[str] | None = None,
) -> BenchConfig:
    """Narrow a campaign (e.g. re-run only FP8). The config hash then reflects what actually ran."""
    cfg = cfg.model_copy(deep=True)
    if precisions:
        for m in cfg.models:
            m.precisions = [p for p in m.precisions if p in precisions]
            m.variants = {k: v for k, v in m.variants.items() if v.precision in precisions}
        cfg.models = [m for m in cfg.models if m.precisions or m.variants]
    if variants:
        known = {n for m in cfg.models for n in m.variants}
        unknown = set(variants) - known
        if unknown:
            raise typer.BadParameter(f"unknown variants: {sorted(unknown)} (have {sorted(known)})")
        for m in cfg.models:
            m.precisions = []
            m.variants = {k: v for k, v in m.variants.items() if k in variants}
        cfg.models = [m for m in cfg.models if m.variants]
    if workloads:
        unknown = set(workloads) - {w.name for w in cfg.perf.workloads}
        if unknown:
            raise typer.BadParameter(f"unknown workloads: {sorted(unknown)}")
        cfg.perf.workloads = [w for w in cfg.perf.workloads if w.name in workloads]
    if skip_workloads:
        unknown = set(skip_workloads) - {w.name for w in cfg.perf.workloads}
        if unknown:
            raise typer.BadParameter(f"unknown workloads: {sorted(unknown)}")
        cfg.perf.workloads = [w for w in cfg.perf.workloads if w.name not in skip_workloads]
    if quality_only:
        cfg.perf.enabled = False
        cfg.accuracy.enabled = False
        if not any(w.quality for w in cfg.perf.workloads):
            raise typer.BadParameter("--quality-only: no selected workload has a quality suite")
        return cfg
    if skip_accuracy:
        cfg.accuracy.enabled = False
    if skip_perf:
        cfg.perf.workloads = []
    if not cfg.sessions() or (not cfg.perf.workloads and not cfg.accuracy.enabled):
        raise typer.BadParameter("filters left nothing to run")
    return cfg


@app.command()
def validate(
    config: Path,
    precision: list[str] = typer.Option(None),
    workload: list[str] = typer.Option(None),
    skip_accuracy: bool = typer.Option(False),
    skip_perf: bool = typer.Option(False),
    skip_workload: list[str] = typer.Option(None),
    skip_quality: bool = typer.Option(False),
    quality_only: bool = typer.Option(False),
    variant: list[str] = typer.Option(None),
) -> None:
    """Validate a config and print the sessions and perf points it expands to."""
    cfg = filter_config(load_config(config), precision, workload, skip_accuracy, skip_perf,
                        skip_workload, quality_only, variant)
    perf = cfg.perf
    typer.echo(f"campaign {cfg.name}")
    for w in perf.workloads:
        slo = perf.slo_for(w)
        typer.echo(f"  workload {w.name}: {w.input_len} in / {w.output_len} out ({w.dataset}), "
                   f"users {perf.concurrency_for(w)} x {perf.repeats_for(w)} repeats, "
                   f"SLO TTFT p99 <= {slo.ttft_p99_ms:g} ms, ITL <= {slo.itl_median_ms:g} ms")
    for s in cfg.sessions():
        typer.echo(f"  session {s.session_id}")
        typer.echo(f"    vllm serve {s.served_model} {' '.join(s.vllm_args())}")
        tasks = [t.name for t in cfg.accuracy.tasks] if cfg.accuracy.enabled else []
        typer.echo(f"    accuracy: {', '.join(tasks) or 'disabled'}")


def resolve_hardware(
    cfg: BenchConfig,
    provider: str | None,
    price_per_hour: float | None,
    provisioning: str | None,
    gpu_type: str | None,
    gpu_count: int | None,
    machine_type: str | None,
    zone: str | None,
) -> BenchConfig:
    """Attach this machine's hardware: detected GPUs + provider/price flags. A config that
    pins `hardware` (old GCP configs) is used as-is apart from explicit overrides."""
    from gpubench.hardware import detect_hardware

    if cfg.hardware is not None and provider is None:
        hw = cfg.hardware.model_copy()
        for field, value in [("price_per_hour_usd", price_per_hour),
                             ("provisioning", provisioning), ("gpu_type", gpu_type),
                             ("gpu_count", gpu_count), ("machine_type", machine_type),
                             ("zone", zone)]:
            if value is not None:
                setattr(hw, field, value)
        return cfg.with_hardware(hw)
    if provider is None:
        raise typer.BadParameter("pass --provider (runpod, lambda, vast, gcp, aws, ...)")
    try:
        hw = detect_hardware(provider, price_per_hour, provisioning or "on-demand",
                             gpu_type, gpu_count, machine_type, zone)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    return cfg.with_hardware(hw)


PROVIDER = typer.Option(None, help="Where this GPU is rented: runpod, lambda, vast, gcp, ...")
PRICE = typer.Option(None, "--price-per-hour", help="USD/h for the whole machine")
DATA_ROOT = typer.Option(None, help="Local silybench-data checkout (default: GitHub)")
PROVISIONING = typer.Option(None, help="on-demand (default), spot, reserved or flex-start")
GPU_TYPE = typer.Option(None, help="Override the detected GPU type, e.g. H100-80GB")
GPU_COUNT = typer.Option(None, help="Override the detected GPU count")


@app.command()
def plan(
    config: Path,
    price_per_hour: float = PRICE,
    gpu_type: str = GPU_TYPE,
    gpu_count: int = GPU_COUNT,
    precision: list[str] = typer.Option(None),
    workload: list[str] = typer.Option(None),
    skip_accuracy: bool = typer.Option(False),
    skip_perf: bool = typer.Option(False),
    experiment: str = typer.Option(None, help="Reproducing this experiment: use its recorded "
                                   "durations instead of the model"),
    data_root: Path = DATA_ROOT,
) -> None:
    """Estimate how long a campaign takes and what it costs, before starting it."""
    from gpubench.config import Hardware
    from gpubench.hardware import detect_hardware
    from gpubench.plan import estimate, format_estimate

    if experiment:
        from gpubench import experiment as ex

        exp, _ = ex.load_experiment(experiment, data_root)
        runs = [r for r in ex.load_runs(exp, data_root)
                if not precision or r.model.precision in precision]
        hours = ex.recorded_hours(runs)
        for run_id, h in hours.items():
            typer.echo(f"  {run_id}: {h:.1f} h")
        total = sum(hours.values())
        typer.echo(f"TOTAL {total:.1f} h, as measured when {experiment} ran")
        if price_per_hour:
            typer.echo(f"GPU cost ~${total * price_per_hour:.0f} at ${price_per_hour}/h "
                       "(+ model download; Spot preemptions add restarts)")
        return

    cfg = filter_config(load_config(config), precision, workload, skip_accuracy, skip_perf)
    try:
        hw = detect_hardware("unknown", price_per_hour, gpu_type=gpu_type, gpu_count=gpu_count)
    except ValueError:
        hw = Hardware(gpu_type=gpu_type or "H100-80GB", gpu_count=gpu_count or 1)
        typer.echo(f"(no GPU detected; estimating for {hw.gpu_count}x {hw.gpu_type})")
    cfg = cfg.with_hardware(hw)
    typer.echo(f"campaign {cfg.name} on {hw.gpu_count}x {hw.gpu_type}")
    typer.echo(format_estimate(estimate(cfg), price_per_hour))


@app.command()
def doctor(
    config: Path = typer.Option(None, help="Also check disk/tokens for this campaign"),
    out: Path = typer.Option(Path("results")),
    hf_cache: Path = typer.Option(None, help="Model cache (default: $HF_HOME or the largest disk)"),
    runtime: str = typer.Option("auto", help="auto, docker or native"),
) -> None:
    """Check that this machine can run a benchmark (GPU, runtime, disk, limits, tokens)."""
    from gpubench.doctor import run_checks
    from gpubench.paths import cache_root
    from gpubench.paths import hf_cache as default_hf_cache

    cfg = load_config(config) if config else None
    typer.echo(f"big files go to {cache_root()} (override: SILYBENCH_CACHE)")
    ok = run_checks(cfg, out, hf_cache or default_hf_cache(), runtime, echo=typer.echo)
    raise typer.Exit(0 if ok else 1)


@app.command()
def run(
    config: Path,
    provider: str = PROVIDER,
    price_per_hour: float = PRICE,
    provisioning: str = PROVISIONING,
    gpu_type: str = GPU_TYPE,
    gpu_count: int = GPU_COUNT,
    machine_type: str = typer.Option(None, help="Provider's instance type, if any"),
    zone: str = typer.Option(None, help="Region/zone/datacenter, if known"),
    runtime: str = typer.Option("auto", help="auto (docker if it has GPU access), docker, native"),
    resume: bool = typer.Option(False, help="Skip finished sessions/workloads/tasks in --out"),
    detach: bool = typer.Option(False, help="Run in the background (survives SSH disconnects)"),
    bucket: str = typer.Option(None, help="Also upload each finished run to this GCS bucket"),
    out: Path = typer.Option(Path("results"), help="Local results directory"),
    hf_cache: Path = typer.Option(None, help="Model cache (default: $HF_HOME or the largest disk)"),
    precision: list[str] = typer.Option(None, help="Only these precisions (repeatable)"),
    workload: list[str] = typer.Option(None, help="Only these perf workloads (repeatable)"),
    skip_accuracy: bool = typer.Option(False, help="Skip the accuracy suite"),
    skip_perf: bool = typer.Option(False, help="Skip performance (accuracy-only re-run)"),
    experiment: str = typer.Option(None, help="Experiment id these runs belong to"),
    no_fingerprint: bool = typer.Option(False, help="Skip the hardware fingerprint"),
    skip_workload: list[str] = typer.Option(None, help="Run every workload except these"),
    skip_quality: bool = typer.Option(False, help="Skip the quality suites"),
    quality_only: bool = typer.Option(False, help="Only the quality suites (no perf points)"),
    variant: list[str] = typer.Option(None, help="Only these serving variants (repeatable)"),
    profile_only: bool = typer.Option(False, help="Only the config's nsys profiles (no perf)"),
) -> None:
    """Run a campaign on this machine (needs NVIDIA GPUs; Docker optional)."""
    from gpubench.plan import estimate
    from gpubench.progress import Progress
    from gpubench.runner import run_session
    from gpubench.server import pick_runtime
    from gpubench.storage import upload_run

    cfg = filter_config(load_config(config), precision, workload, skip_accuracy, skip_perf,
                        skip_workload, quality_only, variant)
    cfg = resolve_hardware(cfg, provider, price_per_hour, provisioning, gpu_type, gpu_count,
                           machine_type, zone)
    hw = cfg.hardware
    if hw.price_per_hour_usd is None:
        typer.echo("warning: no --price-per-hour; $ per token will be computed later from "
                   "the data repo's price table only", err=True)
    out.mkdir(parents=True, exist_ok=True)

    if detach:
        argv = [a for a in sys.argv[1:] if a != "--detach"]
        log_path = out / "run.log"
        with log_path.open("a") as log_file:
            proc = subprocess.Popen(
                [sys.executable, "-m", "gpubench.cli", *argv],
                stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        typer.echo(f"started in the background (pid {proc.pid}); log: {log_path}")
        typer.echo("check progress with: uv run gpubench status")
        return

    from gpubench.paths import hf_cache as default_hf_cache

    hf_cache = hf_cache or default_hf_cache()
    os.environ.setdefault("HF_HOME", str(hf_cache))  # lm-eval's datasets land there too
    rt = pick_runtime(runtime)
    typer.echo(f"{hw.gpu_count}x {hw.gpu_type} on {hw.provider}, runtime {rt}, "
               f"model cache {hf_cache}")
    est = estimate(cfg)
    progress = Progress(out / "progress.json")
    progress.start_campaign(cfg.name, [s.session_id for s in cfg.sessions()], est.perf_points,
                            round(est.hours, 1), hw.price_per_hour_usd)
    fp = None if no_fingerprint else campaign_fingerprint(cfg, rt, out, hf_cache, progress)
    if experiment:
        # Reproducing (or creating) a published experiment: refuse the wrong provider/hardware
        # before spending hours measuring on it.
        from gpubench import experiment as ex

        exp, ref = ex.load_experiment(experiment, data_root_env())
        checks = ex.check_environment(exp, (fp or {}).get("cloud", {}))
        if ref and fp:
            checks += ex.compare_fingerprints(ref, fp)
        typer.echo(ex.format_checks(checks))
        if not ex.verdict(checks):
            progress.finish(["host does not match " + experiment])
            typer.echo(f"FAIL: this host doesn't have the exact hardware used in {experiment}. "
                       "Try again later on another machine, or choose another experiment.",
                       err=True)
            raise typer.Exit(3)
    failed = []
    finished: dict[str, RunResult] = {}
    if profile_only:
        for spec in cfg.profiles:
            profile_standalone(cfg, spec, out, hf_cache, bucket, progress, failed)
        progress.finish(failed)
        if failed:
            raise typer.Exit(1)
        return
    for session in cfg.sessions():
        try:
            result = run_session(session, out, hf_cache, bucket, runtime=rt, resume=resume,
                                 progress=progress, experiment=experiment, fingerprint=fp,
                                 quality=not skip_quality)
        except Exception:
            logging.exception("FAILED session %s; continuing", session.session_id)
            failed.append(session.session_id)
            continue
        finished[session.session_id] = result
        typer.echo(f"finished {out / result.run_id}")
        if bucket:
            typer.echo(f"uploaded {upload_run(out / result.run_id, bucket)}")
    for spec in cfg.profiles:
        run_profile(cfg, spec, finished, out, hf_cache, rt, bucket, progress, failed)
    progress.finish(failed)
    if failed:
        typer.echo(f"failed sessions: {', '.join(failed)}", err=True)
        raise typer.Exit(1)
    typer.echo("campaign complete. Next: uv run gpubench submit " + str(out) + "/*/")


def run_profile(cfg: BenchConfig, spec, finished: dict, out: Path, hf_cache: Path,
                runtime: str, bucket: str | None, progress, failed: list[str]) -> None:
    """Nsight Systems capture for one ProfileSpec; stored in the matching run's result.json."""
    from gpubench.profile import profile_point
    from gpubench.storage import upload_result

    session = next((s for s in cfg.sessions()
                    if s.precision == spec.precision and s.variant == spec.variant), None)
    workload = next((w for w in cfg.perf.workloads if w.name == spec.workload), None)
    if session is None or workload is None or session.session_id not in finished:
        return  # filtered out, or its session failed
    result = finished[session.session_id]
    if any(p.tool == "nsys" and p.workload == spec.workload
           and p.concurrency == spec.concurrency for p in result.profiles):
        return  # already captured (resume)
    if runtime != "docker":
        typer.echo("nsys profiling needs the docker runtime; skipping", err=True)
        return
    progress.update(stage="profile", detail=f"{session.session_id} {spec.workload}")
    work_dir = out / result.run_id
    try:
        result.profiles.append(profile_point(session, workload, spec.concurrency, work_dir,
                                             hf_cache, spec.output_len, echo=typer.echo))
    except Exception:
        logging.exception("FAILED nsys profile of %s; continuing", session.session_id)
        failed.append(f"profile {session.session_id}")
        return
    (work_dir / "result.json").write_text(result.model_dump_json(indent=2))
    if bucket:
        upload_result(work_dir, bucket, cfg.name)


def profile_standalone(cfg: BenchConfig, spec, out: Path, hf_cache: Path, bucket: str | None,
                       progress, failed: list[str]) -> None:
    """--profile-only: capture into its own directory <UTC>_<session>_nsys/ (no perf run)."""
    from datetime import UTC, datetime

    from gpubench.profile import profile_point
    from gpubench.storage import upload_result

    session = next((s for s in cfg.sessions()
                    if s.precision == spec.precision and s.variant == spec.variant), None)
    workload = next((w for w in cfg.perf.workloads if w.name == spec.workload), None)
    if session is None or workload is None:
        return
    work_dir = out / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}_{session.session_id}_nsys"
    work_dir.mkdir(parents=True, exist_ok=True)
    progress.update(stage="profile", detail=f"{session.session_id} {spec.workload}")
    try:
        prof = profile_point(session, workload, spec.concurrency, work_dir, hf_cache,
                             spec.output_len, echo=typer.echo)
        (work_dir / "profile.json").write_text(prof.model_dump_json(indent=2))
    except Exception:
        logging.exception("FAILED nsys profile of %s", session.session_id)
        failed.append(f"profile {session.session_id}")
    if bucket:
        upload_result(work_dir, bucket, cfg.name)


def data_root_env() -> Path | None:
    root = os.environ.get("SILYBENCH_DATA_DIR")
    return Path(root) if root else None


def campaign_fingerprint(cfg: BenchConfig, runtime: str, out: Path, hf_cache: Path,
                         progress) -> dict:
    """Fingerprint this machine once per campaign (GPU idle, before vLLM starts); reused on
    --resume so every run of the campaign carries the same fingerprint."""
    from gpubench.fingerprint import collect, hf_shard_url, summary
    from gpubench.server import make_server

    path = out / "fingerprint.json"
    progress.update(stage="fingerprint", detail="GPU microbenchmark")
    session = cfg.sessions()[0]
    server = make_server(runtime, session, out / "fingerprint-vllm.log", hf_cache, out)
    fp = collect(server.python_cmd(), hf_shard_url(session.served_model), echo=typer.echo)
    typer.echo(f"fingerprint: {summary(fp)}")
    if not path.exists():
        path.write_text(json.dumps(fp, indent=2))
        return fp
    # Resumed (e.g. a new Spot VM after preemption): this machine must be the same hardware
    # as the one the campaign started on, or its numbers can't be merged with the rest.
    from gpubench import experiment as ex

    first = json.loads(path.read_text())
    boots = out / "fingerprints"
    boots.mkdir(exist_ok=True)
    (boots / f"{fp['collected_at'].replace(':', '')}.json").write_text(json.dumps(fp, indent=2))
    checks = ex.compare_fingerprints(first, fp)
    typer.echo(ex.format_checks(checks))
    if not ex.verdict(checks):
        progress.finish(["resumed on different hardware"])
        typer.echo("FAIL: this machine differs from the one the campaign started on; results "
                   "can't be mixed. Relaunch (another host) or start the campaign fresh.",
                   err=True)
        raise typer.Exit(3)
    return first


@app.command()
def fingerprint(
    out: Path = typer.Option(None, help="Write the JSON here"),
    runtime: str = typer.Option("auto"),
    config: Path = typer.Option(Path("configs/smoke.yaml"), help="Its engine is used to run "
                                "the microbenchmark"),
) -> None:
    """Fingerprint this machine: exact GPU identity + measured bandwidth/throughput."""
    from gpubench.config import Hardware
    from gpubench.fingerprint import collect, summary
    from gpubench.paths import hf_cache
    from gpubench.server import make_server, pick_runtime

    cfg = load_config(config).with_hardware(Hardware(gpu_type="any"))
    server = make_server(pick_runtime(runtime), cfg.sessions()[0], Path("/dev/null"),
                         hf_cache(), Path("."))
    fp = collect(server.python_cmd(), None, echo=typer.echo)
    typer.echo(summary(fp))
    if out:
        out.write_text(json.dumps(fp, indent=2))
        typer.echo(f"wrote {out}")


@app.command("verify-host")
def verify_host(
    experiment: str = typer.Option(..., help="Experiment id, e.g. 2026-10-qwen3.8-27b-h100"),
    data_root: Path = DATA_ROOT,
    runtime: str = typer.Option("auto"),
    fingerprint_file: Path = typer.Option(None, help="Use this fingerprint instead of "
                                          "measuring (testing)"),
) -> None:
    """Is this machine the exact hardware the experiment ran on? Exit 1 if not."""
    from gpubench import experiment as ex

    exp, ref = ex.load_experiment(experiment, data_root)
    if fingerprint_file:
        cur = json.loads(fingerprint_file.read_text())
    else:
        from gpubench.fingerprint import collect
        from gpubench.paths import hf_cache
        from gpubench.server import make_server, pick_runtime

        cfg = load_config(REPO_ROOT / exp.bench.config)
        from gpubench.config import Hardware

        cfg = cfg.with_hardware(Hardware(gpu_type="any"))
        server = make_server(pick_runtime(runtime), cfg.sessions()[0], Path("/dev/null"),
                             hf_cache(), Path("."))
        cur = collect(server.python_cmd() if ref else None, None, echo=typer.echo)
    checks = ex.check_environment(exp, cur.get("cloud", {}))
    if ref:
        checks += ex.compare_fingerprints(ref, cur)
    else:
        typer.echo("(this experiment has no reference fingerprint yet; checking the "
                   "environment only)")
    typer.echo(ex.format_checks(checks))
    if ex.verdict(checks):
        typer.echo(f"\nPASS: this host matches experiment {experiment}")
        return
    failed = [c.field for c in checks if c.status == "FAIL"]
    typer.echo(f"\nFAIL: this host doesn't have the exact hardware used in {experiment} "
               f"({', '.join(failed)}). Try again later on another machine, or choose an "
               "experiment that was run on this GPU/provider.")
    raise typer.Exit(1)


experiment_app = typer.Typer(no_args_is_help=True, help="Published experiments.")
app.add_typer(experiment_app, name="experiment")


@experiment_app.command("list")
def experiment_list(data_root: Path = DATA_ROOT) -> None:
    """Published experiments (id, model, hardware, provider)."""
    from gpubench import experiment as ex

    for e in ex.list_experiments(data_root):
        if e.get("status") != "published":
            continue
        envs = ", ".join(f"{v['provider']} {v.get('machine_type') or ''}".strip()
                         for v in e["environments"])
        typer.echo(f"{e['id']}: {e['title']} [{envs}]")


@experiment_app.command("manifest")
def experiment_manifest(config: Path) -> None:
    """Print the models/scenarios blocks of experiment.yaml for a campaign config."""
    import yaml

    from gpubench.experiment import manifest_from_config

    class NoAliases(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True

    typer.echo(yaml.dump(manifest_from_config(load_config(config)), Dumper=NoAliases,
                         sort_keys=False, default_flow_style=None, width=100))


@experiment_app.command("show")
def experiment_show(exp_id: str, data_root: Path = DATA_ROOT) -> None:
    """Everything needed to reproduce an experiment exactly."""
    from gpubench import experiment as ex
    from gpubench.fingerprint import summary

    exp, ref = ex.load_experiment(exp_id, data_root)
    typer.echo(f"{exp.id}: {exp.title} ({exp.status})")
    typer.echo(f"  code:    {exp.bench.repo} @ tag {exp.bench.tag}"
               + (f" ({exp.bench.commit[:12]})" if exp.bench.commit else ""))
    typer.echo(f"  config:  {exp.bench.config}")
    for env in exp.environments:
        typer.echo(f"  where:   {env.provider} {env.machine_type or ''} {env.zone or ''} "
                   f"{env.provisioning or ''}, image {env.image}, runtime {env.runtime}")
    if ref:
        typer.echo(f"  hardware: {summary(ref)}")
    if exp.phases:
        typer.echo("  phases:  " + " ; then ".join(f'--phase "{p}"' for p in exp.phases))
    if exp.runs:
        hours = ex.recorded_hours(ex.load_runs(exp, data_root))
        typer.echo(f"  runs:    {', '.join(exp.runs)} (~{sum(hours.values()):.1f} h measured)")


@app.command()
def compare(
    run_dirs: list[Path],
    experiment: str = typer.Option(...),
    data_root: Path = DATA_ROOT,
) -> None:
    """Compare a reproduction with the published runs of an experiment."""
    from gpubench import experiment as ex

    exp, _ = ex.load_experiment(experiment, data_root)
    refs = {(r.model.hf_id, r.model.precision): r for r in ex.load_runs(exp, data_root)}
    for d in run_dirs:
        rep = RunResult.model_validate_json((d / "result.json").read_text())
        ref = refs.get((rep.model.hf_id, rep.model.precision))
        if ref is None:
            typer.echo(f"{rep.run_id}: no published run for this model/precision")
            continue
        typer.echo(f"== {rep.run_id} vs {ref.run_id}")
        typer.echo(ex.format_checks(ex.compare_runs(ref, rep)))


@app.command()
def status(out: Path = typer.Option(Path("results")), lines: int = 15) -> None:
    """Show the progress of the (detached) run in --out."""
    from gpubench.progress import Progress, pid_alive

    path = out / "progress.json"
    if not path.exists():
        typer.echo(f"no run found ({path} missing)")
        raise typer.Exit(1)
    st = Progress(path).state
    alive = pid_alive(st.get("pid"))
    state = "FINISHED" if st.get("finished") else ("RUNNING" if alive else "STOPPED (crashed "
                                                  "or killed; re-run with --resume)")
    done, planned = st.get("points_done", 0), st.get("planned_points") or 0
    typer.echo(f"campaign {st.get('campaign')}: {state}")
    typer.echo(f"  session {st.get('session')}  stage: {st.get('stage')} {st.get('detail', '')}")
    typer.echo(f"  perf points {done}/~{planned}  started {st.get('started_at')}  "
               f"updated {st.get('updated_at')}  estimate ~{st.get('estimate_hours')} h")
    if st.get("failed_sessions"):
        typer.echo(f"  failed sessions: {', '.join(st['failed_sessions'])}")
    log_path = out / "run.log"
    if log_path.exists() and lines:
        typer.echo(f"--- last {lines} lines of {log_path}")
        typer.echo("\n".join(log_path.read_text(errors="replace").splitlines()[-lines:]))


@app.command()
def submit(
    run_dirs: list[Path],
    data_repo: str = typer.Option(DEFAULT_DATA_REPO, help="GitHub owner/repo of the data repo"),
    dry_run: bool = typer.Option(False, help="Only build the submission in --stage"),
    stage: Path = typer.Option(Path("submission"), help="Where --dry-run writes files"),
    allow_incomplete: bool = typer.Option(False, help="Submit runs that did not finish"),
) -> None:
    """Validate finished runs and open a pull request adding them to the data repo.
    Runs of a published experiment get a comparison with its numbers in the PR."""
    from gpubench.submit import SubmitError, submit_runs

    dirs = [d for d in run_dirs if (d / "result.json").exists()]
    if not dirs:
        raise typer.BadParameter("no <run_dir>/result.json found in the given paths")
    extra = ""
    first = RunResult.model_validate_json((dirs[0] / "result.json").read_text())
    if first.experiment:
        from gpubench import experiment as ex

        try:
            exp, _ = ex.load_experiment(first.experiment, data_root_env())
            refs = {(r.model.hf_id, r.model.precision): r for r in ex.load_runs(exp)}
            parts = []
            for d in dirs:
                rep = RunResult.model_validate_json((d / "result.json").read_text())
                ref = refs.get((rep.model.hf_id, rep.model.precision))
                if ref:
                    parts.append(f"**{rep.model.precision} vs published {ref.run_id}**\n```\n"
                                 + ex.format_checks(ex.compare_runs(ref, rep)) + "\n```")
            extra = "### Comparison with the published experiment\n" + "\n".join(parts)
        except FileNotFoundError:
            pass
    try:
        url = submit_runs(dirs, data_repo, dry_run, stage, allow_incomplete, echo=typer.echo,
                          extra=extra)
    except SubmitError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(url)


@app.command("publish-raw")
def publish_raw(
    release: str = typer.Option(..., help="Release tag on the data repo, e.g. exp-<experiment>"),
    experiment: str = typer.Option(..., help="Experiment id (assets.json goes in its folder)"),
    title: str = typer.Option(None, help="Release title (default: the tag)"),
    run_dir: list[Path] = typer.Option(None, help="Complete run directories (repeatable)"),
    bundle: list[Path] = typer.Option(None, help="Code bundles that ran (sanitized first)"),
    file: list[Path] = typer.Option(None, help="Other files: VM logs, prompt datasets, ..."),
    data_root: Path = typer.Option(..., help="Local silybench-data checkout to update"),
    data_repo: str = typer.Option(DEFAULT_DATA_REPO),
    dry_run: bool = typer.Option(False, help="Build the files, don't upload"),
    stage: Path = typer.Option(Path("publish-stage")),
    redact: list[str] = typer.Option(None, help="Strings to redact in code bundles "
                                     "(e.g. your GCP project id)"),
) -> None:
    """Maintainers: upload everything captured to a data-repo release and record it."""
    from gpubench import publish

    files, contents = [], {}
    for d in run_dir or []:
        f = publish.pack_run(d, stage)
        files.append(f)
        contents[f.name] = (f"complete run directory {d.name}: result.json, per-repeat "
                            "vllm bench JSON, GPU telemetry, vLLM/lm-eval logs, accuracy "
                            "results and per-question samples")
    for b in bundle or []:
        f, removed = publish.sanitize_bundle(b, stage, tuple(redact or ()))
        files.append(f)
        contents[f.name] = ("exact code bundle that ran on the VM (MANIFEST.sha256 lists every "
                            f"file; {len(removed)} private terraform state/vars files removed)")
    for x in file or []:
        if publish.is_private(str(x)):
            raise typer.BadParameter(f"refusing to publish private file {x}")
        files.append(x)
        contents[x.name] = ("VM log" if x.suffix == ".log" else
                            "prompt dataset" if x.suffix == ".jsonl" else x.name)
    total = sum(f.stat().st_size for f in files) / 1e6
    typer.echo(f"{len(files)} files, {total:.0f} MB")
    if dry_run:
        for f in files:
            typer.echo(f"  {f.name}: {contents[f.name]}")
        return
    publish.ensure_release(data_repo, release, title or release,
                           f"Raw data for experiment `{experiment}`. See "
                           f"experiments/{experiment}/ in this repo.")
    assets = publish.upload(data_repo, release, files, contents, echo=typer.echo)
    exp_dir = data_root / "experiments" / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    existing = []
    if (exp_dir / "assets.json").exists():
        from gpubench.schema import RawAsset

        existing = [RawAsset.model_validate(a)
                    for a in json.loads((exp_dir / "assets.json").read_text())]
    merged = {a.name: a for a in existing} | {a.name: a for a in assets}
    publish.write_index(list(merged.values()), exp_dir / "assets.json")
    for a in assets:
        run_id = a.name.removesuffix(".tar.gz")
        result = data_root / "runs" / run_id / "result.json"
        if result.exists():
            publish.attach_assets(result, [a])
            typer.echo(f"recorded {a.name} in runs/{run_id}/result.json")
    typer.echo(f"recorded {len(assets)} assets in {exp_dir / 'assets.json'}")


variants_app = typer.Typer(no_args_is_help=True, help="Compare serving variants.")
app.add_typer(variants_app, name="variants")


@variants_app.command("report")
def variants_report(
    run_dirs: list[Path],
    baseline: str = typer.Option("base", help="Variant (or precision) to compare against"),
    workload: str = typer.Option("toolcall-100k-512"),
    price: list[str] = typer.Option(["gcp-spot=6.571"], help="label=USD/h (repeatable)"),
    out: Path = typer.Option(None, help="Also write the markdown here"),
) -> None:
    """Capacity, latency, cost and quality of each variant vs the baseline (markdown)."""
    from gpubench.variants import report

    prices = {k: float(v) for k, v in (p.split("=", 1) for p in price)}
    dirs = [d for d in run_dirs if (d / "result.json").exists()]
    text = report(dirs, baseline, workload, prices)
    typer.echo(text)
    if out:
        out.write_text(text)


@app.command("install-engine")
def install_engine(config: Path) -> None:
    """Pre-install the native vLLM (pip) for a campaign, so `run` starts measuring sooner."""
    from gpubench.server import ensure_native_vllm

    venv = ensure_native_vllm(load_config(config).engine.version)
    typer.echo(f"vLLM ready in {venv}")


@app.command()
def aggregate(
    bucket: str = typer.Option(None, help="Pull result.json files from this GCS bucket first"),
    results: Path = typer.Option(Path("results"), help="Local dir containing */result.json"),
    out: Path = typer.Option(Path("derived"), help="Where to write index.json + runs/"),
) -> None:
    """Merge run results into index.json + runs/*.json (what the website reads)."""
    from gpubench.dataset import build_site_data, load_runs
    from gpubench.storage import download_results

    if bucket:
        download_results(bucket, results)
    runs = load_runs(results)
    build_site_data(runs, out)
    typer.echo(f"wrote {len(runs)} runs to {out}")


@app.command()
def sample(site_data: Path = typer.Option(Path("derived"))) -> None:
    """Write synthetic placeholder runs (clearly flagged) so a site builds before real data."""
    from gpubench.dataset import build_site_data
    from gpubench.sample import sample_runs

    runs = sample_runs()
    build_site_data(runs, site_data)
    typer.echo(f"wrote {len(runs)} SAMPLE runs to {site_data}")


@app.command()
def rescore(
    run_dir: Path,
    bucket: str = typer.Option(None, help="Upload the updated result.json to this bucket"),
) -> None:
    """Re-score tasks in RESCORERS from a run's saved lm-eval samples and update result.json."""
    from gpubench.accuracy import RESCORERS, latest_samples, rescore_samples
    from gpubench.config import AccuracyTask
    from gpubench.storage import upload_result

    result = RunResult.model_validate_json((run_dir / "result.json").read_text())
    for i, old in enumerate(result.accuracy):
        if old.task not in RESCORERS:
            continue
        task = AccuracyTask(name=old.task, num_fewshot=old.num_fewshot, limit=old.limit)
        new = rescore_samples(task, latest_samples(run_dir / "accuracy" / old.task, old.task))
        result.accuracy[i] = new
        typer.echo(f"{old.task}: {old.metric} {old.value:.4f} -> {new.metric} {new.value:.4f}")
    (run_dir / "result.json").write_text(result.model_dump_json(indent=2))
    if bucket:
        upload_result(run_dir, bucket)
        typer.echo(f"uploaded gs://{bucket}/runs/{run_dir.name}/result.json")


@app.command()
def schema(out: Path = typer.Option(Path("schema") / "result.schema.json")) -> None:
    """Export the RunResult JSON Schema (published in the data repo)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(RunResult.model_json_schema(), indent=2))
    typer.echo(f"wrote {out}")


if __name__ == "__main__":
    app()
