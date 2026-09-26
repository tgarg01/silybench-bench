# silybench-bench

Benchmark open LLMs on **any rented GPU**, then publish the numbers. It measures latency,
concurrent-user capacity, throughput, energy, accuracy and **$ per token**, serving with a pinned
vLLM. Results go to the public dataset [silybench-data](https://github.com/tgarg01/silybench-data),
which powers the self-hosting vs API cost comparison on the silybench site.

## Reproduce a published experiment: let Claude do it

Each experiment on the silybench site lists exactly where and how it ran: provider, machine type,
zone, pinned boot image, GPU fingerprint, and the git tag of this repo. To re-measure it yourself,
start [Claude Code](https://claude.com/claude-code) and paste:

> Clone https://github.com/tgarg01/silybench-bench and follow its AGENTS.md to reproduce experiment 2026-10-qwen3.8-27b-h100. Show me the cost before creating anything.

Claude:
1. checks out the experiment's exact code;
2. refuses providers the experiment didn't use;
3. provisions the same machine and image, and verifies that the GPU, driver, power limit and
   measured memory/matmul speed match the reference fingerprint. If they don't, it stops and tells
   you to try later or pick another GPU;
4. shows the cost, based on the experiment's measured durations, and waits for your OK;
5. runs with automatic resume across Spot preemptions;
6. compares your numbers with the published ones and opens a "Reproduction of …" pull request.

By hand: `uv run gpubench experiment show <id>`, then follow [AGENTS.md](AGENTS.md).

| Campaign | What |
|---|---|
| `configs/qwen3.8-27b-h100.yaml` | **Published experiment** `2026-10-qwen3.8-27b-h100`: Qwen3.8-27B BF16 + official FP8 checkpoint, 6 scenarios including 100k-token tool-calling contexts |
| `configs/smoke.yaml` | Pipeline check (not publishable) |
| `configs/qwen3-8b*.yaml`, `qwen3-14b.yaml` | Pipeline-validation campaigns (Qwen3-8B runs are kept as pipeline data) |

## How it works

| | |
|---|---|
| Runtimes | **docker**: the pinned `vllm/vllm-openai` image, used where Docker can reach the GPUs. **native**: `pip install vllm==<same version>` in its own uv venv, for RunPod and Vast pods, which are containers without Docker. The runtime used is recorded in each result. |
| Hardware | Detected with `nvidia-smi` and normalised (e.g. `H100-80GB`, `H100-PCIe-80GB`, `A100-80GB`), so runs from different providers line up. Provider and price come from flags. |
| Load | `vllm bench serve` with random prompts of fixed input/output length (`--ignore-eos`) at each concurrency level, 3 repeats, median taken. |
| Capacity | Bisection for the most concurrent users with **p99 TTFT ≤ 2 s** and **median inter-token latency ≤ 50 ms**. The KV-cache limit is reported too. |
| Accuracy | lm-evaluation-harness against the same server: MMLU-Pro, GPQA-Diamond, GSM8K, MATH-500, IFEval, ARC-Challenge. |
| Hardware fingerprint | `nvidia-smi -q` identity (PCI device id tells H100 SXM from PCIe/NVL, power limit, VBIOS, ECC/MIG), driver/CUDA, CPU/RAM, cloud machine type/zone/image, measured HBM bandwidth, BF16/FP8 matmul TFLOPS, PCIe and download speed, temperatures. `gpubench verify-host` compares a machine with an experiment's reference. |
| Robustness | The run directory is saved (and mirrored to GCS on GCP) after every point. `--resume` skips finished work. `watch.sh` relaunches Spot VMs. `--detach` survives SSH drops. vLLM listens on localhost only. |
| Scenarios | Random prompts of exact lengths, plus `custom` JSONL datasets checked by sha256. The 100k tool-calling prompts are built from real SWE-agent sessions (`gpubench prompts make-toolcall`). Per-scenario SLO, concurrency and repeats. |
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
