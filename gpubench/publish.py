"""Maintainers: publish everything a run captured as GitHub release assets on silybench-data.

The data repo keeps small files in git (result.json, a small raw.tar.gz). The complete run
directory, including per-question accuracy samples, VM logs, code bundles and (later) Nsight
reports, is too big for git and goes to a release named after the experiment. Each run's
result.json then lists its assets with URL, sha256 and size, so the website links them and
anyone can verify a download.
"""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path

from gpubench.schema import RawAsset, RunResult

# Never published, even if an old code bundle contains them (they hold project/billing ids).
PRIVATE_PATTERNS = (".tfstate", "terraform.tfvars", ".auto.tfvars", ".env", ".pem",
                    ".terraform/")


def is_private(name: str) -> bool:
    return any(p in name for p in PRIVATE_PATTERNS)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack_run(run_dir: Path, out_dir: Path) -> Path:
    """<run_id>.tar.gz of the whole run directory (nothing left out)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{run_dir.name}.tar.gz"
    with tarfile.open(out, "w:gz") as tar:
        for f in sorted(run_dir.rglob("*")):
            if f.is_file() and not is_private(str(f)):
                tar.add(f, arcname=f"{run_dir.name}/{f.relative_to(run_dir)}")
    return out


def sanitize_bundle(bundle: Path, out_dir: Path,
                    redact: tuple[str, ...] = ()) -> tuple[Path, list[str]]:
    """Copy of a code bundle without private files, plus a MANIFEST.sha256 of every original
    file (so the published code is provably what ran). Strings in `redact` (e.g. a GCP project
    id in a README) are replaced in text files; those files are flagged in the manifest.
    Returns (path, removed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / bundle.name
    removed, manifest, redacted = [], [], []
    with tarfile.open(bundle, "r:gz") as src, tarfile.open(out, "w:gz") as dst:
        for member in src.getmembers():
            if is_private(member.name):
                removed.append(member.name)
                continue
            data = src.extractfile(member) if member.isfile() else None
            if data is not None:
                content = data.read()
                manifest.append(f"{hashlib.sha256(content).hexdigest()}  {member.name}")
                clean = content
                for r in redact:
                    clean = clean.replace(r.encode(), b"<redacted>")
                if clean != content:
                    redacted.append(member.name)
                    member.size = len(clean)
                dst.addfile(member, io.BytesIO(clean))
            else:
                dst.addfile(member)
        note = ("# sha256 of every file in the original bundle that ran on the VM.\n"
                "# Removed before publishing (private project/billing state, not code):\n"
                + "".join(f"#   {r}\n" for r in removed)
                + "# Account identifiers replaced by <redacted> (hash above is the original):\n"
                + "".join(f"#   {r}\n" for r in redacted) + "\n".join(manifest) + "\n")
        info = tarfile.TarInfo("MANIFEST.sha256")
        info.size = len(note.encode())
        dst.addfile(info, io.BytesIO(note.encode()))
    return out, removed


def _gh(*args: str) -> str:
    return subprocess.run(["gh", *args], check=True, capture_output=True, text=True).stdout


def ensure_release(repo: str, tag: str, title: str, notes: str) -> None:
    if subprocess.run(["gh", "release", "view", tag, "-R", repo],
                      capture_output=True).returncode != 0:
        _gh("release", "create", tag, "-R", repo, "--title", title, "--notes", notes)


def upload(repo: str, tag: str, files: list[Path], contents: dict[str, str],
           echo: Callable[[str], None] = print) -> list[RawAsset]:
    assets = []
    for f in files:
        echo(f"uploading {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
        _gh("release", "upload", tag, str(f), "--clobber", "-R", repo)
        assets.append(RawAsset(
            name=f.name,
            url=f"https://github.com/{repo}/releases/download/{tag}/{f.name}",
            sha256=_sha256(f), bytes=f.stat().st_size,
            contents=contents.get(f.name, ""),
        ))
    return assets


def attach_assets(result_path: Path, assets: list[RawAsset]) -> None:
    """Add/replace assets (by name) in a result.json."""
    r = RunResult.model_validate_json(result_path.read_text())
    keep = {a.name: a for a in r.raw_assets}
    keep.update({a.name: a for a in assets})
    r.raw_assets = list(keep.values())
    result_path.write_text(r.model_dump_json(indent=2) + "\n")


def write_index(assets: list[RawAsset], out: Path) -> None:
    out.write_text(json.dumps([a.model_dump() for a in assets], indent=2) + "\n")
