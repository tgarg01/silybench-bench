# Runbook for AI agents: reproduce a silybench experiment exactly

A person asked you to reproduce a published silybench experiment: the same model, code, provider,
machine, hardware and measuring method, so they can check the numbers on the website themselves.
**Their confidence is the product**, so follow these steps exactly, don't improvise, and never
"make it work" on hardware or a provider the experiment didn't use.

**Ground rules**
- The person pays for the GPU by the hour. Show time and cost before anything is created, and
  **never provision a GPU or start a run without their explicit OK**.
- Don't edit `gpubench/`, `configs/` or `infra/`. Runs from modified code are rejected (`-dirty`
  commit). If something breaks, report it and stop.
- Never write tokens or credentials into files, commits or chat. The person runs `gcloud auth login`,
  `gh auth login` or `export HF_TOKEN=...` themselves.
- Run commands from the repo root, and prefix gpubench with `uv run` (after `uv sync`).

## 1. Pick the experiment
```bash
uv sync
uv run gpubench experiment list                 # published experiments
uv run gpubench experiment show <experiment-id> # exact code, provider, machine, image, hardware, measured hours
```
Tell the person what `show` says, in plain words. For example: "Qwen3.8-27B on 1× H100 SXM 80GB,
GCP a3-highgpu-1g Spot in us-central1-a, image common-cu129-ubuntu-2404-nvidia-580-v20260909,
silybench-bench tag exp-2026-10-qwen3.8-27b-h100, about 23 GPU-hours."

## 2. Provider gate (hard stop)
The experiment lists the provider(s) it was run on (`where:` in `show`). If the person wants a
different provider, or is on a machine at a different provider, **stop** and say:

> This experiment was only run on <provider>. Reproducing it on <their provider> isn't supported:
> different hosts, drivers and networking change the numbers. Please use <provider>, or pick an
> experiment that was run on <their provider>.

Don't offer workarounds. (To benchmark a *new* provider or GPU, see section B. It isn't a
reproduction and is labelled differently on the site.)

## 3. Exact code
```bash
git fetch --tags
git checkout <tag from show>
git rev-parse HEAD      # must equal the commit printed by `show`
uv sync --extra eval
```

## 4. Provision and run: GCP experiments
GCP experiments are driven from **the person's own computer** (you run there, not on the VM).
Terraform creates the same VM type, zone and pinned boot image in the person's GCP project, and
the VM runs the campaign by itself.

1. **Prerequisites.** Check each; ask the person to fix anything missing:
   - `gcloud` and `terraform` are installed. `gcloud auth login` and
     `gcloud auth application-default login` are done. A project is selected with billing enabled.
   - The project has Spot (preemptible) H100 quota ≥ 1 in the zone's region: ask the person to
     check "Preemptible NVIDIA H100 GPUs" for us-central1 in the console (IAM & Admin → Quotas).
     It isn't listed by `gcloud compute regions describe`. If it is 0, they request an increase;
     stop until then.
   - One-time base resources (bucket, service account, budget):
     `cp infra/terraform/base/terraform.tfvars.example infra/terraform/base/terraform.tfvars`.
     Fill it in with the person, then
     `terraform -chdir=infra/terraform/base init && terraform -chdir=infra/terraform/base apply`
     (show the plan and get an OK).
   - `cp infra/terraform/envs/h100-1g/terraform.tfvars.example infra/terraform/envs/h100-1g/terraform.tfvars`,
     then set `project_id`, `bucket` and `price_per_hour` (their Spot price for the machine).
2. **Cost estimate from the real measured durations:**
   ```bash
   uv run gpubench plan --experiment <id> --price-per-hour <their price>
   ```
   Say "about X hours, about $Y, plus restarts if Spot capacity is reclaimed. Start?" **Wait for yes.**
3. **Launch with the watchdog.** It relaunches after Spot preemptions and resumes from the last
   measured point. Pass every phase that `show` prints (`phases:`), in order, one `--phase` each.
   For `2026-10-qwen3.8-27b-h100`: the 100k tool-calling scenario on both precisions (with its
   quality suite), FP8 + MTP on it, the multi-turn agent sessions with and without prefix caching,
   then the other five scenarios:
   ```bash
   infra/scripts/watch.sh h100-1g <config from show> --experiment <id> --image <image from show> \
     --phase "--session bf16 --session fp8 --workload toolcall-100k-512" \
     --phase "--session fp8-mtp --workload toolcall-100k-512" \
     --phase "--session fp8 --session fp8-pc --session fp8-mtp-pc --workload agent-sessions-100k" \
     --phase "--session bf16 --session fp8 --skip-workload toolcall-100k-512 --skip-workload agent-sessions-100k"
   ```
   To reproduce only part of an experiment, pass just those phases (and tell the person which
   published numbers it covers).
   Every relaunched VM re-fingerprints itself and must match the first one exactly (±5% on the
   measured speeds). If it doesn't, watch.sh retries up to 3 hosts, then stops.
   On boot the VM fingerprints its hardware and runs **verify-host** against the experiment's
   reference before measuring anything. If the hardware differs (another GPU variant, power
   limit, driver, memory bandwidth off by more than 5%, …), the run stops, the VM deletes itself
   and watch.sh exits with FAIL. Then tell the person:

   > This host doesn't have the exact hardware used in <id> (<failed fields>). Please try again
   > later, when GCP may place the VM on another host, or choose a different GPU/experiment.
4. **Monitor.** watch.sh prints progress every 5 min: stage, workload and points done. Relay it
   occasionally; don't poll in a tight loop. The full log is at
   `gcloud storage ls gs://<bucket>/logs/`.

## 5. Compare and submit
```bash
gcloud storage rsync -r gs://<bucket>/runs results/     # mirrored run directories
uv run gpubench compare results/<run_dirs>/ --experiment <id>
gh auth login            # the person does this
uv run gpubench submit results/<run_dirs>/ --dry-run
uv run gpubench submit results/<run_dirs>/
```
`compare` checks max users at the SLO, throughput, inter-token latency and p99 TTFT against the
published runs (±10%). Explain every WARN honestly. The PR is titled "Reproduction of <id>" and
includes the comparison.

## 6. Tear down
```bash
infra/scripts/down.sh h100-1g        # safe if the VM already deleted itself
gcloud compute instances list        # must show no gpubench VM
```
Tell the person that the results bucket costs cents per month, and that they can delete it.

## Checking correctness before and after an optimization (custom long-context scenarios)
Perf runs force a fixed output length, so they say nothing about correctness. Scenarios with a
`quality:` suite (the 100k tool-calling one) run it right after their perf points, on the same
server:
- **recall**: 100 exact-answer questions about tool outputs 20-80% deep inside the 100k context
  ("What was the first line of the output of the command `…`?");
- **drift**: 100 decision points (greedy decoding, top-20 logprobs) whose tool calls and token
  distributions are the baseline.

The published run's `quality/toolcall-100k-512.jsonl` (in the experiment's release) is the
**before**. To check an optimized build (new kernel, engine flag, quantization, …):
```bash
# serve the optimized build on the same GPU type, same checkpoint, then:
uv run gpubench quality run configs/qwen3.8-27b-h100.yaml --workload toolcall-100k-512 \
  --base-url http://localhost:8000 --model Qwen/Qwen3.8-27B-FP8 --out quality-runs/candidate
uv run gpubench quality compare <baseline run dir> quality-runs/candidate
```
`compare` passes when:
- ≥ 97% of the tool calls are identical;
- mean KL divergence ≤ 0.02 and ≥ 98% top-1 token agreement before the first divergence;
- recall drops by ≤ 2 points.

Report every number, not just PASS/FAIL. Compare each precision with its own baseline: FP8 vs
FP8, BF16 vs BF16.

## Experiments on other providers (e.g. a RunPod pod)
When an experiment's `where:` is a pod or VM you connect to by SSH, you run **on that machine**:
1. The person rents exactly the listed machine type (same provider, same GPU type), installs
   Claude Code, and asks you to follow this file.
2. `./setup.sh`, then `uv run gpubench verify-host --experiment <id>`. On FAIL, give the exact
   message from step 4.3 and tell the person to terminate the pod.
3. `uv run gpubench plan --experiment <id> --price-per-hour <price>`, and wait for the OK.
4. `uv run gpubench run <config> --provider <provider> --price-per-hour <price> --experiment <id> --resume --detach`,
   then `uv run gpubench status` every 10-20 min, then steps 5-6 (results are in `results/`).

## B. Benchmark new hardware (not a reproduction)
To measure a GPU/provider that no experiment covers yet: `./setup.sh`,
`uv run gpubench plan <config> --price-per-hour <price>` (and wait for OK), then
`uv run gpubench run <config> --provider <p> --price-per-hour <price> --resume --detach`, then
`gpubench submit`. The maintainers decide whether it becomes a new published experiment.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| watch.sh: `FAIL` / exit 3 in the VM log | verify-host: different hardware | Try later, or a different experiment (see 4.3) |
| watch.sh relaunches repeatedly | Spot capacity is being reclaimed | Normal; each relaunch resumes. If it's too slow, tell the person and suggest a quieter time |
| `Quota 'PREEMPTIBLE_NVIDIA_H100_GPUS' exceeded` | no Spot H100 quota | The person requests quota in the GCP console |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | no Spot H100 free in the zone right now | Wait and retry. A different zone is only acceptable if verify-host still passes (it warns on zone) |
| `sha256 … != expected` for a dataset | corrupted or partial download | Delete the file in `datasets/`; it re-downloads |
| image not found | the pinned boot image was deprecated by Google | Stop: exact reproduction is no longer possible on GCP; tell the person |
| `Too many open files` in a capacity probe | fd limit | Handled automatically (a lower-bound capacity is recorded) |

## What gets measured
- **Latency** per scenario and number of users: p95/p99 time to first token (TTFT), time per
  output token (TPOT), end-to-end latency; median/p95/p99 inter-token latency.
- **Multi-turn agent sessions** (`dataset: sessions`): each user runs one 10-turn session at a
  time, every turn resending the growing context. Also records first-turn vs later-turn TTFT and,
  with prefix caching on, the cache hit rate (the cache is reset before every measurement).
- **Capacity**: the most concurrent users with p99 TTFT ≤ the scenario's limit (2 s, or 30 s for
  100k-token prompts) and median inter-token latency ≤ 50 ms. This drives the cost comparison.
- **Throughput, power, tokens per joule, temperature and thermal throttling** per point.
- **Hardware fingerprint**: GPU identity, measured HBM/matmul/PCIe speed, CPU/RAM, network, temperatures.
- **Accuracy** (lm-evaluation-harness) when the experiment includes it.
