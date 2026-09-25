"""Accuracy evals with lm-evaluation-harness against the running vLLM server."""

from __future__ import annotations

import json
import math
import re
import subprocess
from pathlib import Path

from gpubench.config import AccuracyTask, ServingSession
from gpubench.schema import AccuracyResult

# Primary metric reported per task (lm-eval result key, without the ",filter" suffix).
PRIMARY_METRIC = {
    "gsm8k": ("exact_match", "flexible-extract"),
    "ifeval": ("prompt_level_strict_acc", "none"),
    "mmlu_pro": ("exact_match", "custom-extract"),
    "gpqa_diamond_cot_zeroshot": ("exact_match", "flexible-extract"),
    "arc_challenge_chat": ("exact_match", "remove_whitespace"),
    "minerva_math500": ("math_verify", "none"),
}


# Tasks whose lm-eval scoring doesn't work over a chat API, re-scored from the saved samples.
# arc_challenge_chat relies on `gen_prefix` ("The best answer is") being pre-filled into the
# assistant turn; chat completions can't do that, so the model writes the whole sentence and
# exact_match against the bare letter fails (~6%). We extract the letter instead.
_ARC_LETTER = re.compile(r"best answer is\W*([A-D])\b", re.IGNORECASE)
_BARE_LETTER = re.compile(r"^\W*([A-D])\W*$")


def _arc_letter(response: str) -> str | None:
    found = _ARC_LETTER.findall(response) or _BARE_LETTER.findall(response)
    return found[-1].upper() if found else None


RESCORERS = {"arc_challenge_chat": ("exact_match_letter", _arc_letter)}


def rescore_samples(task: AccuracyTask, samples_path: Path) -> AccuracyResult:
    """Score from lm-eval's samples_*.jsonl; unparseable answers count as wrong."""
    metric, extract = RESCORERS[task.name]
    rows = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    correct = sum(extract(r["resps"][0][0]) == str(r["target"]).strip().upper() for r in rows)
    n = len(rows)
    p = correct / n
    return AccuracyResult(
        task=task.name, metric=metric, value=p,
        stderr=math.sqrt(p * (1 - p) / (n - 1)) if n > 1 else None,
        num_fewshot=task.num_fewshot, limit=task.limit, num_samples=n,
    )


def lm_eval_cmd(session: ServingSession, task: AccuracyTask, output_dir: Path) -> list[str]:
    acc = session.config.accuracy
    port = session.config.engine.port
    model_args = ",".join([
        f"model={session.model.hf_id}",
        f"base_url=http://localhost:{port}/v1/chat/completions",
        f"num_concurrent={acc.num_concurrent}",
        "max_retries=3",
        "tokenized_requests=False",
    ])
    cmd = [
        "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", task.name,
        "--apply_chat_template",
        "--gen_kwargs", f"max_gen_toks={acc.max_gen_toks},temperature=0",
        "--output_path", str(output_dir),
        "--log_samples",
    ]
    if task.num_fewshot is not None:
        cmd += ["--num_fewshot", str(task.num_fewshot), "--fewshot_as_multiturn"]
    if task.limit is not None:
        cmd += ["--limit", str(task.limit)]
    return cmd


def parse_lm_eval_results(results: dict, task: AccuracyTask) -> AccuracyResult:
    """Extract the primary metric from an lm-eval results_*.json dict."""
    task_results = results["results"][task.name]
    metric, filt = PRIMARY_METRIC.get(task.name, (None, None))
    if metric is None:  # unknown task: first non-stderr numeric metric
        key = next(
            k for k, v in task_results.items()
            if isinstance(v, int | float) and "_stderr" not in k and k != "alias"
        )
        metric, _, filt = key.partition(",")
    key = f"{metric},{filt}"
    stderr = task_results.get(f"{metric}_stderr,{filt}")
    n = results.get("n-samples", {}).get(task.name, {})
    return AccuracyResult(
        task=task.name,
        metric=metric,
        value=float(task_results[key]),
        stderr=float(stderr) if isinstance(stderr, int | float) else None,
        num_fewshot=task.num_fewshot,
        limit=task.limit,
        num_samples=n.get("effective"),
    )


def run_task(session: ServingSession, task: AccuracyTask, work_dir: Path) -> AccuracyResult:
    out_dir = work_dir / "accuracy" / task.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # lm-eval's tqdm bars redraw one line with \r and grow without bound; on the VM that line
    # crashed the GCE startup-script log forwarder ("bufio.Scanner: token too long") and the
    # resulting broken pipe killed the whole run. Keep its output in a file instead.
    log_path = out_dir / "lm_eval.log"
    with log_path.open("w") as log_file:
        proc = subprocess.run(
            lm_eval_cmd(session, task, out_dir), stdout=log_file, stderr=subprocess.STDOUT
        )
    if proc.returncode != 0:
        tail = log_path.read_text(errors="replace")[-2000:]
        raise RuntimeError(f"lm_eval {task.name} exited {proc.returncode}; log tail:\n{tail}")
    # lm-eval writes <output_path>/<model_name_sanitized>/results_<timestamp>.json
    result_files = sorted(out_dir.rglob("results_*.json"))
    if not result_files:
        raise FileNotFoundError(f"no lm-eval results under {out_dir}")
    if task.name in RESCORERS:
        return rescore_samples(task, latest_samples(out_dir, task.name))
    return parse_lm_eval_results(json.loads(result_files[-1].read_text()), task)


def latest_samples(out_dir: Path, task_name: str) -> Path:
    files = sorted(out_dir.rglob(f"samples_{task_name}_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no lm-eval samples for {task_name} under {out_dir}")
    return files[-1]


def lm_eval_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("lm-eval")
    except Exception:
        return None
