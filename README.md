# silybench-bench

Benchmark open LLMs on **any rented GPU**, then publish the numbers. It measures latency,
concurrent-user capacity, throughput, energy, accuracy and **$ per token**, serving with a pinned
vLLM. Results go to the public dataset [silybench-data](https://github.com/tgarg01/silybench-data),
which powers the self-hosting vs API cost comparison on the silybench site.

## Quick start: let Claude do it

1. Rent a GPU box with SSH (RunPod, Vast.ai, Lambda, Hyperbolic, any cloud VM or your own machine).
   Give it about 100 GB of disk.
2. Install [Claude Code](https://claude.com/claude-code) on it, start `claude`, and paste:

> Clone https://github.com/tgarg01/silybench-bench and follow its AGENTS.md to benchmark this GPU. Ask me for the provider, hourly price and campaign, and show me the cost estimate before starting.

Claude runs setup and pre-flight checks, estimates the time and cost for your OK, runs the
campaign in the background (it survives disconnects and resumes after a crash), and opens a
pull request with the results. Then it reminds you to shut the box down.

## Or by hand

```bash
git clone https://github.com/tgarg01/silybench-bench && cd silybench-bench
./setup.sh configs/qwen3-8b-quick.yaml                         # deps, runtime, doctor
uv run gpubench plan configs/qwen3-8b-quick.yaml --price-per-hour 2.69
uv run gpubench run  configs/qwen3-8b-quick.yaml --provider runpod --price-per-hour 2.69 --resume --detach
uv run gpubench status                                         # re-run until FINISHED
gh auth login && uv run gpubench submit results/*/             # PR to silybench-data
```

| Campaign | What | ~1x H100 |
|---|---|---|
| `configs/smoke.yaml` | Pipeline check (not publishable) | 15 min |
| `configs/qwen3-8b-quick.yaml` | Qwen3-8B BF16+FP8, 3 workloads, 1 repeat, no accuracy | 1.5 h |
| `configs/qwen3-8b.yaml` | Full Qwen3-8B: 5 workloads, 8 concurrency levels × 3 repeats, capacity search, 6 accuracy tasks | 12-16 h |
| `configs/qwen3-14b.yaml` | Full Qwen3-14B | 20+ h |

## How it works

| | |
|---|---|
| Runtimes | **docker**: the pinned `vllm/vllm-openai` image, used where Docker can reach the GPUs. **native**: `pip install vllm==<same version>` in its own uv venv, for RunPod and Vast pods, which are containers without Docker. The runtime used is recorded in each result. |
| Hardware | Detected with `nvidia-smi` and normalised (e.g. `H100-80GB`, `H100-PCIe-80GB`, `A100-80GB`), so runs from different providers line up. Provider and price come from flags. |
| Load | `vllm bench serve` with random prompts of fixed input/output length (`--ignore-eos`) at each concurrency level, 3 repeats, median taken. |
| Capacity | Bisection for the most concurrent users with **p99 TTFT ≤ 2 s** and **median inter-token latency ≤ 50 ms**. The KV-cache limit is reported too. |
| Accuracy | lm-evaluation-harness against the same server: MMLU-Pro, GPQA-Diamond, GSM8K, MATH-500, IFEval, ARC-Challenge. |
| Robustness | `result.json` is saved after every point. `--resume` skips finished work. `--detach` survives SSH drops. vLLM listens on localhost only. |
| Provenance | Each result records the git commit (dirty checkouts are rejected), the vLLM version and image digest, the driver and CUDA versions, the model revision SHA, and every serving flag. |

`gpubench --help` lists every command. `result.json` follows `gpubench/schema.py` (published as JSON
Schema in the data repo).

## Layout

| Path | What |
|---|---|
| `gpubench/` | The package: config → vLLM server → perf sweep → capacity search → lm-eval → `result.json`; plus `submit`, `doctor`, `plan`, and `dataset`/`cost` (used by the data repo's CI) |
| `configs/` | Campaigns: models, precisions, workloads, SLO, accuracy tasks. No hardware; that's detected. |
| `AGENTS.md` | The runbook agents follow on a rented box |
| `infra/` | Optional fully automated **GCP** path (Terraform Spot VM that runs a campaign and deletes itself). See below. |

## GCP (automated, optional)

```bash
gcloud auth login && gcloud auth application-default login
cp infra/terraform/base/terraform.tfvars.example infra/terraform/base/terraform.tfvars   # fill in
terraform -chdir=infra/terraform/base init && terraform -chdir=infra/terraform/base apply
cp infra/terraform/envs/h100-1g/terraform.tfvars.example infra/terraform/envs/h100-1g/terraform.tfvars
infra/scripts/up.sh h100-1g configs/smoke.yaml     # VM runs the campaign, uploads to GCS, deletes itself
infra/scripts/logs.sh h100-1g
uv run gpubench aggregate --bucket <bucket> --out /tmp/agg   # or submit the run dirs from GCS
```

## Develop

```bash
uv sync && uv run pytest && uv run ruff check .
for c in configs/*.yaml; do uv run gpubench validate $c; done
```

Adding a model: copy a campaign config and change `models`. Adding a GPU type: nothing to change; just
run it. Multi-GPU: set `parallelism` (tp/pp/dp/ep). GPU count and parallelism are recorded in the
schema.
