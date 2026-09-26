"""Optional GCS sink for the GCP Terraform path (`--bucket`). Rented boxes use `submit`."""

from __future__ import annotations

import subprocess
from pathlib import Path


def upload_run(run_dir: Path, bucket: str) -> str:
    """Copy a run directory (result.json, raw bench JSON, logs, telemetry) to GCS."""
    dest = f"gs://{bucket}/runs/{run_dir.name}/"
    subprocess.run(["gcloud", "storage", "cp", "-r", f"{run_dir}/*", dest], check=True)
    return dest


def upload_result(run_dir: Path, bucket: str, campaign: str = "default") -> None:
    """Mirror the whole run directory (incremental) after every measured point, so a Spot
    preemption loses at most one point and no raw file: a relaunched VM pulls it back and
    `--resume` continues. Campaign-level files (fingerprint, progress) go alongside."""
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", str(run_dir),
         f"gs://{bucket}/runs/{run_dir.name}"],
        check=False, capture_output=True,
    )
    for name in ("fingerprint.json", "progress.json"):
        f = run_dir.parent / name
        if f.exists():
            subprocess.run(
                ["gcloud", "storage", "cp", str(f), f"gs://{bucket}/campaign-state/{campaign}/{name}"],
                check=False, capture_output=True,
            )


def download_results(bucket: str, dest: Path) -> None:
    """Fetch only the result.json of each run (raw files stay in GCS)."""
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gcloud", "storage", "rsync", "-r", "-x", r"^(?!.*result\.json$).*",
         f"gs://{bucket}/runs", str(dest)],
        check=True,
    )
