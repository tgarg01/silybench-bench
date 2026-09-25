"""`gpubench submit`: package finished runs and open a pull request on the data repo.

Adds runs/<run_id>/result.json and raw.tar.gz (raw `vllm bench serve` JSON, GPU telemetry,
vLLM and lm-eval logs and scores; per-sample lm-eval outputs are left out to stay small).
Uses the GitHub CLI: pushes a branch if you can write to the data repo, otherwise forks it.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Callable
from pathlib import Path

from gpubench.dataset import MAX_RAW_BYTES, check_result
from gpubench.schema import RunResult

# Kept out of raw.tar.gz: large, per-example model outputs (re-creatable by re-running).
EXCLUDE_SUFFIXES = (".jsonl",)
EXCLUDE_NAMES = {"result.json"}


class SubmitError(Exception):
    pass


def pack_raw(run_dir: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in sorted(run_dir.rglob("*")):
            if not f.is_file() or f.name in EXCLUDE_NAMES or f.suffix in EXCLUDE_SUFFIXES:
                continue
            tar.add(f, arcname=str(f.relative_to(run_dir)))
    data = buf.getvalue()
    if len(data) > MAX_RAW_BYTES:
        raise SubmitError(f"{run_dir.name}: raw.tar.gz would be {len(data) / 1e6:.0f} MB "
                          f"(max {MAX_RAW_BYTES / 1e6:.0f} MB)")
    return data


def stage_runs(run_dirs: list[Path], dest: Path, allow_incomplete: bool) -> list[RunResult]:
    """Write runs/<run_id>/{result.json,raw.tar.gz} under dest after validating each run."""
    results = []
    for run_dir in run_dirs:
        result = RunResult.model_validate_json((run_dir / "result.json").read_text())
        problems = check_result(result, allow_incomplete)
        if problems:
            raise SubmitError(f"{run_dir.name}: " + "; ".join(problems))
        out = dest / "runs" / result.run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "result.json").write_text(result.model_dump_json(indent=2))
        (out / "raw.tar.gz").write_bytes(pack_raw(run_dir))
        results.append(result)
    return results


def pr_body(results: list[RunResult]) -> str:
    hw = results[0].hardware
    lines = [
        f"Benchmark runs from `gpubench submit` on **{hw.gpu_count}x {hw.gpu_type}** "
        f"({hw.provider}, {hw.provisioning}"
        + (f", ${hw.price_per_hour_usd}/h" if hw.price_per_hour_usd else "") + ").",
        "",
        "| run | model | precision | vLLM | runtime | workload | max users (SLO) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for c in r.capacity or [None]:
            lines.append(
                f"| `{r.run_id}` | {r.model.hf_id} | {r.model.precision} "
                f"| {r.software.engine_version} | {r.software.runtime} "
                f"| {c.workload if c else '-'} | {c.max_users_slo if c else '-'} |"
            )
    acc = {(r.model.precision, a.task): a.value for r in results for a in r.accuracy}
    if acc:
        lines += ["", "Accuracy: " + ", ".join(f"{p}/{t} {v:.3f}" for (p, t), v in acc.items())]
    lines += ["", f"bench commit: `{results[0].git_commit}`",
              "", "🤖 Submitted with [silybench](https://github.com/tgarg01/silybench-bench)"]
    return "\n".join(lines)


def _gh(*args: str, cwd: Path | None = None, capture: bool = True) -> str:
    try:
        out = subprocess.run(["gh", *args], cwd=cwd, check=True, text=True,
                             capture_output=capture)
    except FileNotFoundError as e:
        raise SubmitError("GitHub CLI `gh` not found; run ./setup.sh") from e
    except subprocess.CalledProcessError as e:
        raise SubmitError(f"gh {' '.join(args)} failed: {e.stderr or e.stdout}") from e
    return (out.stdout or "").strip()


def _git(*args: str, cwd: Path) -> None:
    try:
        subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise SubmitError(f"git {' '.join(args)} failed: {e.stderr}") from e


def submit_runs(
    run_dirs: list[Path],
    data_repo: str,
    dry_run: bool,
    stage: Path,
    allow_incomplete: bool = False,
    echo: Callable[[str], None] = print,
) -> str:
    if dry_run:
        if stage.exists():
            shutil.rmtree(stage)
        results = stage_runs(run_dirs, stage, allow_incomplete)
        (stage / "PR_BODY.md").write_text(pr_body(results))
        for f in sorted(stage.rglob("*")):
            if f.is_file():
                echo(f"  {f.relative_to(stage)}  ({f.stat().st_size / 1e3:.0f} kB)")
        return f"dry run: submission staged in {stage}"

    if subprocess.run(["gh", "auth", "status"], capture_output=True).returncode != 0:
        raise SubmitError("not logged in to GitHub: run `gh auth login` (or set GH_TOKEN)")
    _gh("auth", "setup-git")  # let `git push` use gh's token
    login = _gh("api", "user", "--jq", ".login")
    user_id = _gh("api", "user", "--jq", ".id")
    perms = json.loads(_gh("api", f"repos/{data_repo}", "--jq", ".permissions") or "{}")
    can_push = bool(perms.get("push"))

    with tempfile.TemporaryDirectory() as tmp:
        repo_dir = Path(tmp) / "data"
        if can_push:
            _gh("repo", "clone", data_repo, str(repo_dir), "--", "--depth", "1")
            head_prefix = ""
        else:
            echo(f"no write access to {data_repo}; submitting from your fork")
            _gh("repo", "fork", data_repo, "--clone=false")
            fork = f"{login}/{data_repo.split('/')[1]}"
            _gh("repo", "sync", fork, "--source", data_repo)
            _gh("repo", "clone", fork, str(repo_dir), "--", "--depth", "1")
            head_prefix = f"{login}:"
        results = stage_runs(run_dirs, repo_dir, allow_incomplete)
        branch = f"runs/{results[0].run_id}"
        ident = ["-c", f"user.name={login}",
                 "-c", f"user.email={user_id}+{login}@users.noreply.github.com"]
        _git("checkout", "-b", branch, cwd=repo_dir)
        _git("add", "runs", cwd=repo_dir)
        hw = results[0].hardware
        title = (f"Add {len(results)} run(s): {results[0].model.hf_id} on "
                 f"{hw.gpu_count}x {hw.gpu_type} ({hw.provider})")
        _git(*ident, "commit", "-m", title, cwd=repo_dir)
        _git("push", "-u", "origin", branch, cwd=repo_dir)
        return _gh("pr", "create", "--repo", data_repo, "--head", f"{head_prefix}{branch}",
                   "--title", title, "--body", pr_body(results), cwd=repo_dir)
