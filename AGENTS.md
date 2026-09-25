# Runbook for AI agents: benchmark this GPU box and submit the results

You are on a rented GPU machine (RunPod, Vast.ai, Lambda, Hyperbolic, bare metal, a cloud VM, ...).
A person asked you to run a silybench benchmark. The results go into the public dataset
behind the silybench site, a self-hosting vs API cost comparison, so follow these steps exactly
and don't improvise with the numbers.

**Ground rules**
- The person pays for this machine by the hour. Say what each long step costs before you start it,
  and **never start a campaign without their explicit OK** on the estimate (step 3).
- Don't edit `gpubench/` or `configs/`. Runs from modified code are rejected (`git_commit` ends in
  `-dirty`). If something is broken, report it and stop; don't patch around it.
- Never paste tokens (HF_TOKEN, GitHub) into files, commits or chat. Ask the person to `export` them
  or run `gh auth login` themselves.
- All commands run from the repo root: `cd silybench-bench`.

## 1. Setup (5-10 min)
```bash
./setup.sh configs/<campaign>.yaml      # or just ./setup.sh if the campaign isn't chosen yet
```
It installs uv, Python deps, lm-eval, gh and tmux. It picks the vLLM runtime: Docker with the NVIDIA
runtime if the box has it, otherwise **native**, a pip-installed vLLM of the same version (RunPod
and Vast pods are containers with no Docker). It puts model weights on the largest disk and ends by
running `gpubench doctor`.

## 2. Doctor: fix every ✗
```bash
uv run gpubench doctor --config configs/<campaign>.yaml
```
| ✗ / ! | What to do |
|---|---|
| no NVIDIA GPU visible | Wrong machine/template. Stop and tell the person. |
| disk: … need ~N GB | Ask the person to enlarge the volume, or `export SILYBENCH_CACHE=/path/on/big/disk` and re-run doctor. |
| lm-eval missing | `uv sync --extra eval` |
| HF_TOKEN not set (GPQA) | Optional. The GPQA task is skipped without it. Ask the person to accept the dataset terms at huggingface.co/datasets/Idavidrein/gpqa and `export HF_TOKEN=...`. |
| gh not logged in | Needed only in step 5. Ask the person to run `gh auth login` (device code flow works over SSH). |
| open-files hard limit < 65535 | Warning only. Capacity above ~limit/2 users can't be probed, so results become lower bounds. |

## 3. Choose the campaign, estimate the cost, and get approval
Ask the person for:
1. **Provider**: `runpod`, `vast`, `lambda`, `hyperbolic`, `gcp`, `aws`, `azure`, `coreweave`, `nebius`, `together`, … (lowercase).
2. **Hourly price in USD for this whole machine**, from their provider dashboard. It is recorded with
   the results, and the site reprices every benchmark with each provider's current list price anyway.
3. **Pricing type**: `on-demand` (default), `spot`, `reserved`.
4. **Campaign**:

| config | what | ~time on 1x H100 |
|---|---|---|
| `configs/smoke.yaml` | 15-minute pipeline check. **Not publishable.** Run it first on a new provider/GPU type. | 0.2 h |
| `configs/qwen3-8b-quick.yaml` | Qwen3-8B BF16+FP8, 3 workloads, 1 repeat, no accuracy | 1.5 h |
| `configs/qwen3-8b.yaml` | Full Qwen3-8B: 5 workloads × 8 concurrency levels × 3 repeats + capacity search + 6 accuracy tasks | 12-16 h |
| `configs/qwen3-14b.yaml` | Full Qwen3-14B | ~20+ h |

Then show them the estimate:
```bash
uv run gpubench plan configs/<campaign>.yaml --price-per-hour <USD/h>
```
Tell them: "This will take about X hours and cost about $Y (±50%). Start?" **Wait for a yes.**
If the full campaign is too expensive, offer `--precision fp8`, `--workload <name>` or `--skip-accuracy`
(the same flags work for `plan` and `run`).

## 4. Run it detached, then monitor
```bash
uv run gpubench run configs/<campaign>.yaml --provider <provider> --price-per-hour <USD/h> \
  [--provisioning spot] --resume --detach
```
- `--detach` keeps the run going if SSH or your session drops. `--resume` makes re-running the same
  command safe: finished sessions, workloads and accuracy tasks are skipped.
- The first session downloads the model and, on the native runtime, installs vLLM, so expect
  5-15 min before the first measurement.

Check progress (cheap; do it every ~10-20 min, not in a tight loop):
```bash
uv run gpubench status
```
- `RUNNING`: fine. Report `stage`, `perf points N/~M` and elapsed vs estimate to the person.
- `STOPPED`: the process died. Read `results/run.log` (tail), then re-run the **same** `run`
  command. `--resume` continues where it stopped.
- `FINISHED` with failed sessions: the log says why. Common fixes are below.
- A workload or accuracy task failing is logged and skipped; the rest of the campaign continues.

## 5. Submit the results as a pull request
```bash
uv run gpubench submit results/*/ --dry-run    # validate and show what would be submitted
uv run gpubench submit results/*/              # opens a PR on github.com/tgarg01/silybench-data
```
- This needs `gh auth login` (the person's GitHub account). Without write access it forks the data
  repo automatically.
- Smoke runs and unfinished runs are refused on purpose. To submit a partly failed campaign, add
  `--allow-incomplete` and tell the person what is missing.
- Give the person the PR URL. The data repo's CI validates the run, and after merge the website
  rebuilds with the new numbers.

## 6. Stop paying
Remind the person to **stop or terminate the pod/VM** now. Results live in the PR, so the box is
disposable. (`results/` is local only; if the PR failed, copy it off first:
`tar czf results.tgz results/` then scp/download it.)

## Troubleshooting (known failures)
| Symptom in `results/run.log` / `vllm.log` | Cause | Fix |
|---|---|---|
| `vLLM exited during startup`, `CUDA out of memory` | Another process holds GPU memory | `nvidia-smi` to find it; kill it; `--resume` |
| `No space left on device` | Weights on a small root disk | `export SILYBENCH_CACHE=/big/disk/silybench-cache`, `--resume` |
| `Too many open files` during a capacity probe | fd limit | Automatically caught: capacity is recorded as a lower bound. Nothing to do. |
| pip install of vLLM fails (native) | Driver too old for the vLLM wheel's CUDA | Report the driver (`nvidia-smi`) and stop; the provider template needs a newer driver. |
| `gated` / 401 on `Idavidrein/gpqa` | No HF_TOKEN, or terms not accepted | GPQA is skipped; the other tasks still count. |
| Spot/preemptible box vanished | Preemption | New box: clone, `./setup.sh`, copy `results/` back if you saved it, same `run --resume` |
| `vllm: command not found` in native mode | Engine install interrupted | `rm -rf <cache>/engines/vllm-*` then `uv run gpubench install-engine configs/<campaign>.yaml` |

## What gets measured (for answering the person's questions)
- **Latency**: p95/p99 time-to-first-token, time-per-output-token, end-to-end; median, p95 and p99
  inter-token latency, per workload and concurrency.
- **Capacity**: the most concurrent users served while p99 TTFT ≤ 2 s and median inter-token
  latency ≤ 50 ms (≥ 20 tokens/s per user). This drives the cost numbers. Also: the KV-cache limit.
- **Throughput, tokens/joule, $ per 1M tokens** at the price they gave.
- **Accuracy** with lm-evaluation-harness (MMLU-Pro, GPQA-Diamond, GSM8K, MATH-500, IFEval, ARC-C),
  so quantization (FP8) trade-offs are visible.
