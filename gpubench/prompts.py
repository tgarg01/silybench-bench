"""Prompt datasets for `custom` workloads: build them once, then verify and stage them per run.

The tool-calling scenario sends real agent transcripts, not random tokens. Every benchmark run
downloads the same file and checks its sha256 first, so the prompts are byte-identical on every
machine that reproduces an experiment.

Building (maintainers, once per dataset; CPU only):

    uv run --extra prompts gpubench prompts make-toolcall --model Qwen/Qwen3.8-27B \\
        --tokens 100000 --count 200 --out datasets/toolcall-100k.jsonl

Source: nebius/SWE-agent-trajectories (CC-BY-4.0), 80k SWE-agent sessions in which a model
fixes GitHub issues through shell/editor commands. Each session becomes an OpenAI-style
tool-calling conversation (commands -> tool_calls, observations -> tool results). Sessions from
the same repository are chained into one long agent context until it exceeds the target, and
the middle is trimmed so every prompt is exactly `tokens` tokens under the model's tokenizer,
ending with the generation prompt (the model is about to decide its next tool call).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import httpx

from gpubench.config import Workload

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = "nebius/SWE-agent-trajectories"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dataset(workload: Workload) -> Path:
    """Local path of a custom workload's prompt file, downloaded if missing, hash-checked."""
    path = Path(workload.dataset_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        if not workload.dataset_url:
            raise FileNotFoundError(
                f"{path} missing and workload {workload.name} has no dataset_url")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", workload.dataset_url, follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.replace(path)
    digest = sha256_file(path)
    if digest != workload.sha256:
        raise ValueError(f"{path}: sha256 {digest} != expected {workload.sha256}; "
                         "delete it to re-download")
    return path


def ensure_file(path_str: str, url: str | None, sha256: str, what: str) -> Path:
    """Local copy of a hash-pinned file (downloaded from `url` if missing)."""
    return ensure_dataset(Workload(name=what, dataset="custom", input_len=1, output_len=1,
                                   dataset_path=path_str, dataset_url=url, sha256=sha256))


def stage_dataset(workload: Workload, datasets_dir: Path) -> Path:
    """Put the verified prompt file where the load generator reads it (the run's datasets/)."""
    src = ensure_dataset(workload)
    dest = datasets_dir / src.name
    if not dest.exists():
        datasets_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest.hardlink_to(src)
        except OSError:
            shutil.copyfile(src, dest)
    return dest


# --- building the tool-calling dataset --------------------------------------------------------

RUN_COMMAND_TOOL = {
    "type": "function",
    "function": {
        "name": "run_command",
        "description": "",  # filled with the SWE-agent interface docs from the session
        "parameters": {
            "type": "object",
            "properties": {"command": {
                "type": "string",
                "description": "One shell command or special interface command (open, goto, "
                               "scroll_down, edit, create, search_dir, find_file, submit, ...)",
            }},
            "required": ["command"],
        },
    },
}

_CODE_BLOCK = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


def trajectory_to_messages(traj: list[dict]) -> tuple[str, list[dict]]:
    """SWE-agent session -> (interface docs, OpenAI-style messages with tool calls).

    The agent's turns are "thought + one command in a code block"; the command becomes a
    run_command tool call and the next user turn (the command's output) its tool result.
    """
    docs = ""
    messages: list[dict] = []
    pending_call: str | None = None
    for i, turn in enumerate(traj):
        role, text = turn.get("role"), turn.get("text") or ""
        if role == "system":
            docs = turn.get("system_prompt") or text
            continue
        if role == "ai":
            blocks = _CODE_BLOCK.findall(text)
            if blocks:
                call_id = f"call_{i}"
                thought = _CODE_BLOCK.sub("", text).strip()
                messages.append({
                    "role": "assistant", "content": thought,
                    "tool_calls": [{"id": call_id, "type": "function", "function": {
                        "name": "run_command", "arguments": {"command": blocks[-1].strip()}}}],
                })
                pending_call = call_id
            else:
                messages.append({"role": "assistant", "content": text})
                pending_call = None
        elif role == "user":
            if pending_call:
                messages.append({"role": "tool", "tool_call_id": pending_call, "content": text})
                pending_call = None
            else:
                messages.append({"role": "user", "content": text})
    return docs, messages


class Renderer:
    """The model's own chat template + tokenizer, without transformers."""

    def __init__(self, model: str, revision: str = "main"):
        import jinja2
        from huggingface_hub import hf_hub_download
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        from tokenizers import Tokenizer

        self.tokenizer = Tokenizer.from_file(hf_hub_download(model, "tokenizer.json",
                                                             revision=revision))
        template = Path(hf_hub_download(model, "chat_template.jinja", revision=revision))

        def raise_exception(msg: str):
            raise jinja2.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = lambda x, indent=None, **_: json.dumps(
            x, ensure_ascii=False, indent=indent)
        env.globals["raise_exception"] = raise_exception
        self.template = env.from_string(template.read_text())

    def render(self, messages: list[dict], tools: list[dict], **kwargs) -> str:
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=True,
                                    **kwargs)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)


def fit_to_tokens(render: Callable[[list[dict]], str], count: Callable[[str], int],
                  messages: list[dict], target: int) -> str | None:
    """Make the rendered prompt exactly `target` tokens: drop the oldest turns after the first
    user message while the prompt stays >= target, then trim the largest tool result.
    None if the conversation is too short or exactness can't be reached."""
    fitted = fit_messages(render, count, messages, target)
    return fitted[0] if fitted else None


def fit_messages(render: Callable[[list[dict]], str], count: Callable[[str], int],
                 messages: list[dict], target: int) -> tuple[str, list[dict]] | None:
    """fit_to_tokens, also returning the fitted messages."""
    msgs = [dict(m) for m in messages]
    # End on a tool result / user turn: the model is about to decide its next call.
    while msgs and msgs[-1]["role"] == "assistant":
        msgs.pop()
    if count(render(msgs)) < target:
        return None
    head = 1  # keep the first user message (the task)

    def drop_oldest(ms: list[dict]) -> list[dict]:
        out = ms[:head] + ms[head + 1:]
        # A tool result must follow the assistant call that produced it.
        while len(out) > head and out[head]["role"] == "tool":
            out = out[:head] + out[head + 1:]
        return out

    while len(msgs) > head + 2:
        shorter = drop_oldest(msgs)
        if count(render(shorter)) < target:
            break
        msgs = shorter
    tools = [i for i, m in enumerate(msgs) if m["role"] == "tool" and m["content"]]
    if not tools:
        return None
    idx = max(tools, key=lambda i: len(msgs[i]["content"]))
    original = msgs[idx]["content"]

    def with_tail(n: int) -> list[dict]:
        ms = [dict(m) for m in msgs]
        ms[idx]["content"] = "[...]\n" + original[len(original) - n:]
        return ms

    lo, hi = 0, len(original)  # largest kept tail with count <= target
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(render(with_tail(mid))) <= target:
            lo = mid
        else:
            hi = mid - 1
    ms = with_tail(lo)
    # Token boundaries can leave it a token or two short; pad the trimmed result with spaces.
    for _ in range(16):
        text = render(ms)
        n = count(text)
        if n == target:
            return text, ms
        if n > target:
            return None
        ms[idx]["content"] = " " + ms[idx]["content"]
    return None


def fitted_contexts(model: str, tokens: int, count_prompts: int, seed: int, shards: int,
                    echo: Callable[[str], None] = print):
    """The deterministic agent contexts behind the tool-calling datasets:
    yields (renderer, tool, text, messages) for each of `count_prompts` contexts."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    renderer = Renderer(model)
    by_repo: dict[str, list[list[dict]]] = defaultdict(list)
    for i in range(shards):
        path = hf_hub_download(SOURCE_DATASET, f"data/train-{i:05d}-of-00012.parquet",
                               repo_type="dataset")
        table = pq.read_table(path, columns=["instance_id", "trajectory"])
        for iid, traj in zip(table.column("instance_id").to_pylist(),
                             table.column("trajectory").to_pylist(), strict=True):
            by_repo[iid.rsplit("-", 1)[0]].append(traj)
    rng = random.Random(seed)
    repos = sorted(by_repo)
    rng.shuffle(repos)
    echo(f"{sum(len(v) for v in by_repo.values())} sessions from {len(repos)} repositories")

    made = 0
    for repo in repos:
        if made >= count_prompts:
            break
        sessions = by_repo[repo][:]
        rng.shuffle(sessions)
        docs, messages = "", []
        for traj in sessions:  # chain sessions of one repo into one long agent context
            d, m = trajectory_to_messages(traj)
            docs = docs or d
            messages += m
            tool = json.loads(json.dumps(RUN_COMMAND_TOOL))
            tool["function"]["description"] = docs
            render = lambda ms, t=tool: renderer.render(ms, [t])  # noqa: E731
            if renderer.count(render(messages)) >= tokens:
                fitted = fit_messages(render, renderer.count, messages, tokens)
                if fitted is not None:
                    made += 1
                    yield renderer, tool, fitted[0], fitted[1]
                break
    if made < count_prompts:
        raise RuntimeError(f"only {made} prompts reached {tokens} tokens; use more --shards")


def build_toolcall_dataset(model: str, tokens: int, count_prompts: int, out: Path,
                           seed: int = 42, shards: int = 2,
                           echo: Callable[[str], None] = print) -> str:
    """Write `count_prompts` prompts of exactly `tokens` tokens to `out`; return its sha256."""
    prompts = [text for _, _, text, _ in
               fitted_contexts(model, tokens, count_prompts, seed, shards, echo)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}, ensure_ascii=False) + "\n")
    digest = sha256_file(out)
    echo(f"wrote {len(prompts)} prompts x {tokens} tokens to {out} (sha256 {digest})")
    return digest
