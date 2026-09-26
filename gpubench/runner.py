"""Orchestrates one campaign: for each serving session, start vLLM and run everything."""

from __future__ import annotations

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from gpubench import __version__
from gpubench.accuracy import lm_eval_version, run_task
from gpubench.capacity import capacity_result, find_max_users, kv_cache_max_users
from gpubench.config import ServingSession
from gpubench.perf import make_point_runner
from gpubench.progress import Progress
from gpubench.prompts import ensure_file
from gpubench.quality import run_quality
from gpubench.schema import ModelInfo, PerfPoint, RunResult, SoftwareInfo
from gpubench.server import Runtime, make_server
from gpubench.storage import upload_result
from gpubench.telemetry import driver_info

log = logging.getLogger("gpubench")


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_commit() -> str | None:
    """Commit of the code being run (+"-dirty" if modified). The GCP bundle has no .git, so
    up.sh stamps a file instead."""
    stamp = REPO_ROOT / ".gpubench-commit"
    if stamp.exists():
        return stamp.read_text().strip() or None
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    commit = out.stdout.strip()
    if not commit:
        return None
    dirty = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--quiet", "HEAD", "--", "gpubench", "configs"],
        capture_output=True,
    ).returncode != 0
    return f"{commit}-dirty" if dirty else commit


def resolve_revision(hf_id: str, revision: str) -> str:
    """Pin 'main' to a commit sha so results are reproducible."""
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(hf_id, revision=revision).sha or revision
    except Exception:
        return revision


def find_previous(out_root: Path, session: ServingSession) -> tuple[Path, RunResult] | None:
    """Latest earlier run of this exact session (same config hash) under out_root."""
    config_hash = session.config.config_hash()
    for path in sorted(out_root.glob(f"*_{session.session_id}/result.json"), reverse=True):
        try:
            result = RunResult.model_validate_json(path.read_text())
        except ValueError:
            continue
        if result.config_hash == config_hash:
            return path.parent, result
    return None


def missing_work(session: ServingSession, result: RunResult) -> list[str]:
    """Workloads/quality suites the config asks for that the run doesn't have (e.g. they
    failed); a resumed campaign retries them."""
    perf = session.config.perf
    have = {c.workload for c in result.capacity}
    have_q = {q.workload for q in result.quality}
    missing = [w.name for w in perf.workloads if perf.enabled and w.name not in have]
    missing += [f"{w.name} quality" for w in perf.workloads
                if w.quality is not None and w.name not in have_q]
    return missing


def prepare_resume(result: RunResult) -> RunResult:
    """Keep finished workloads (those with a capacity entry) and accuracy tasks; drop the
    perf points of a workload that was cut off mid-way so it is re-measured from scratch."""
    done = {c.workload for c in result.capacity}
    return result.model_copy(update={
        "perf": [p for p in result.perf if p.workload in done],
        "complete": False,
    })


def run_session(
    session: ServingSession,
    out_root: Path,
    hf_cache: Path,
    bucket: str | None = None,
    runtime: Runtime = "docker",
    resume: bool = False,
    progress: Progress | None = None,
    experiment: str | None = None,
    fingerprint: dict | None = None,
    quality: bool = True,
) -> RunResult:
    cfg = session.config
    hw = session.hardware
    progress = progress or Progress(out_root / "progress.json")

    previous = find_previous(out_root, session) if resume else None
    if previous and previous[1].complete and missing_work(session, previous[1]):
        log.info("previous run %s is missing %s; resuming it", previous[1].run_id,
                 ", ".join(missing_work(session, previous[1])))
        previous = (previous[0], previous[1].model_copy(update={"complete": False}))
    if previous and previous[1].complete:
        log.info("=== session %s already complete in %s; skipping", session.session_id,
                 previous[0])
        return previous[1]

    if previous:
        work_dir, result = previous[0], prepare_resume(previous[1])
        log.info("=== resuming session %s in %s", session.session_id, work_dir)
    else:
        started = datetime.now(UTC)
        run_id = f"{started:%Y%m%dT%H%M%SZ}_{session.session_id}"
        work_dir = out_root / run_id
        work_dir.mkdir(parents=True, exist_ok=True)
        log.info("=== session %s -> %s", session.session_id, work_dir)
        driver, cuda = driver_info()
        result = RunResult(
            run_id=run_id,
            campaign=cfg.name,
            created_at=started,
            git_commit=git_commit(),
            config_hash=cfg.config_hash(),
            complete=False,
            experiment=experiment,
            fingerprint=fingerprint,
            hardware=hw,
            parallelism=cfg.parallelism,
            software=SoftwareInfo(
                engine_image=cfg.engine.image,
                lm_eval_version=lm_eval_version(),
                gpubench_version=__version__,
                nvidia_driver=driver,
                cuda_version=cuda,
                runtime=runtime,
            ),
            model=ModelInfo(
                hf_id=session.model.hf_id,
                checkpoint=session.served_model if session.prequantized else None,
                revision=resolve_revision(session.served_model, session.model.revision),
                precision=session.precision,
                max_model_len=session.model.max_model_len,
                thinking=session.model.thinking,
            ),
            serving_args=session.vllm_args(),
        )

    def save() -> None:
        # Written (and pushed to GCS if configured) after every point, so a preemption or a
        # dropped SSH session loses at most one point; `--resume` picks up from here.
        (work_dir / "result.json").write_text(result.model_dump_json(indent=2))
        if bucket:
            upload_result(work_dir, bucket, cfg.name)

    save()
    progress.update(run_id=result.run_id, session=session.session_id, stage="starting vLLM",
                    detail=session.served_model)
    server = make_server(runtime, session, work_dir / "vllm.log", hf_cache, work_dir)
    try:
        server.start()
    except Exception:
        save()  # mirror vllm.log (the reason) before giving up on this session
        raise
    with server:
        if any(w.dataset == "custom" for w in cfg.perf.workloads) and cfg.perf.enabled:
            result.software.bench_extras = server.ensure_bench_extras()
        result.software.engine_version = server.version()
        result.software.engine_image_digest = server.image_digest()
        result.kv_cache_tokens = server.kv_cache_tokens()
        log.info("vLLM %s (%s), KV cache %s tokens", result.software.engine_version, runtime,
                 result.kv_cache_tokens)

        measure = make_point_runner(
            session, lambda args: server.exec(args, check=True), work_dir,
            raw_dir=server.raw_dir, datasets_dir=server.datasets_dir,
        )
        perf = cfg.perf
        done_workloads = {c.workload for c in result.capacity}
        # A failing workload or task is logged and skipped so the rest of the campaign still runs.
        for workload in perf.workloads if perf.enabled else []:
            if workload.name in done_workloads:
                log.info("perf %s already measured; skipping", workload.name)
                continue
            try:
                def measure_and_save(c: int, w=workload, phase: str = "perf") -> PerfPoint:
                    # Every point (sweep or capacity probe) is logged and saved as soon as it lands.
                    log.info("%s %s c=%d", phase, w.name, c)
                    progress.update(stage=phase, detail=f"{w.name} @ {c} users")
                    point = measure(w, c)
                    result.perf.append(point)
                    progress.point_done()
                    save()
                    return point

                kv_users = kv_cache_max_users(result.kv_cache_tokens, workload)
                # Past vLLM's running-request cap, extra users only queue too.
                if session.max_num_seqs:
                    kv_users = min(kv_users or session.max_num_seqs, session.max_num_seqs)
                levels, skipped = perf.levels(workload, kv_users)
                if skipped:
                    log.info("%s: skipping %s users (> %sx the %s that fit in the KV cache)",
                             workload.name, skipped, perf.max_kv_multiple, kv_users)
                sweep = {c: measure_and_save(c) for c in levels}
                max_users, points = find_max_users(
                    sweep, lambda c: measure_and_save(c, phase="probe"), perf.capacity,
                    ceiling=2 * kv_users if kv_users else None,
                )
                cap = capacity_result(workload, perf.slo_for(workload), max_users, points,
                                      result.kv_cache_tokens)
                cap.skipped_levels = skipped
                result.capacity.append(cap)
                log.info("capacity %s: %d users (SLO)", workload.name, max_users)
                save()
            except Exception:
                log.exception("FAILED perf workload %s; continuing", workload.name)

        # Correctness of custom long-context workloads, while the same server is up.
        done_quality = {q.workload for q in result.quality}
        for workload in perf.workloads:
            suite = workload.quality
            if suite is None or not quality or workload.name in done_quality:
                continue
            progress.update(stage="quality", detail=workload.name)
            try:
                dataset = ensure_file(suite.dataset_path, suite.dataset_url, suite.sha256,
                                      f"{workload.name}-quality")
                kv_users = kv_cache_max_users(result.kv_cache_tokens, workload) or 1
                result.quality.append(run_quality(
                    server.base_url, session.served_model, workload.name, dataset, work_dir,
                    concurrency=max(1, min(4, kv_users)),
                    recall_max_tokens=suite.recall_max_tokens,
                    drift_max_tokens=suite.drift_max_tokens, echo=log.info,
                    limit=suite.limit))
                save()
            except Exception:
                log.exception("FAILED quality suite of %s; continuing", workload.name)

        if cfg.accuracy.enabled:
            done_tasks = {a.task for a in result.accuracy}
            for task in cfg.accuracy.tasks:
                if task.name in done_tasks:
                    log.info("accuracy %s already scored; skipping", task.name)
                    continue
                log.info("accuracy %s", task.name)
                progress.update(stage="accuracy", detail=task.name)
                try:
                    result.accuracy.append(run_task(session, task, work_dir))
                    save()
                except Exception:
                    log.exception("FAILED accuracy task %s; continuing", task.name)

    result.perf.sort(key=lambda p: (p.workload, p.concurrency))
    result.complete = True
    save()
    progress.update(stage="session done", detail="")
    return result
