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
) -> BenchConfig:
    """Narrow a campaign (e.g. re-run only FP8). The config hash then reflects what actually ran."""
    cfg = cfg.model_copy(deep=True)
    if precisions:
        for m in cfg.models:
            m.precisions = [p for p in m.precisions if p in precisions]
        cfg.models = [m for m in cfg.models if m.precisions]
    if workloads:
        unknown = set(workloads) - {w.name for w in cfg.perf.workloads}
        if unknown:
            raise typer.BadParameter(f"unknown workloads: {sorted(unknown)}")
        cfg.perf.workloads = [w for w in cfg.perf.workloads if w.name in workloads]
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
) -> None:
    """Validate a config and print the sessions and perf points it expands to."""
    cfg = filter_config(load_config(config), precision, workload, skip_accuracy, skip_perf)
    perf = cfg.perf
    points = len(perf.workloads) * len(perf.concurrency)
    typer.echo(f"campaign {cfg.name}")
    for s in cfg.sessions():
        typer.echo(f"  session {s.session_id}")
        typer.echo(f"    vllm serve {s.model.hf_id} {' '.join(s.vllm_args())}")
        typer.echo(f"    perf: {points} sweep points x {perf.repeats} repeats (+ capacity search)")
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
) -> None:
    """Estimate how long a campaign takes and what it costs, before starting it."""
    from gpubench.config import Hardware
    from gpubench.hardware import detect_hardware
    from gpubench.plan import estimate, format_estimate

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
) -> None:
    """Run a campaign on this machine (needs NVIDIA GPUs; Docker optional)."""
    from gpubench.plan import estimate
    from gpubench.progress import Progress
    from gpubench.runner import run_session
    from gpubench.server import pick_runtime
    from gpubench.storage import upload_run

    cfg = filter_config(load_config(config), precision, workload, skip_accuracy, skip_perf)
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
    failed = []
    for session in cfg.sessions():
        try:
            result = run_session(session, out, hf_cache, bucket, runtime=rt, resume=resume,
                                 progress=progress)
        except Exception:
            logging.exception("FAILED session %s; continuing", session.session_id)
            failed.append(session.session_id)
            continue
        typer.echo(f"finished {out / result.run_id}")
        if bucket:
            typer.echo(f"uploaded {upload_run(out / result.run_id, bucket)}")
    progress.finish(failed)
    if failed:
        typer.echo(f"failed sessions: {', '.join(failed)}", err=True)
        raise typer.Exit(1)
    typer.echo("campaign complete. Next: uv run gpubench submit " + str(out) + "/*/")


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
    """Validate finished runs and open a pull request adding them to the data repo."""
    from gpubench.submit import SubmitError, submit_runs

    dirs = [d for d in run_dirs if (d / "result.json").exists()]
    if not dirs:
        raise typer.BadParameter("no <run_dir>/result.json found in the given paths")
    try:
        url = submit_runs(dirs, data_repo, dry_run, stage, allow_incomplete, echo=typer.echo)
    except SubmitError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(url)


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
