#!/usr/bin/env bash
# The AGENTS.md benchmark flow end to end, on the GPU-free mock runtime (CI and local).
# Proves the wiring: plan -> run (detached) -> status -> resume -> quality compare ->
# submit refuses mock runs -> the data repo's dataset build. Real vLLM/GPU behaviour is
# covered by the GCP smoke run, not here.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$(mktemp -d)
G="uv run python -m gpubench.cli"
FLAGS=(--provider ci --gpu-type H100-80GB --price-per-hour 1 --runtime mock --no-fingerprint --out "$OUT")

echo "== plan"
$G plan configs/ci/mock.yaml --gpu-type H100-80GB --price-per-hour 1
echo "== quality phase first (like the experiment's 100k-first order), then the rest"
$G run configs/ci/mock.yaml "${FLAGS[@]}" --workload toolcall-mini --resume --detach
for _ in $(seq 1 120); do
  if $G status --out "$OUT" --lines 0 | grep -q "FINISHED"; then break; fi
  sleep 2
done
STATUS=$($G status --out "$OUT" --lines 5)
echo "$STATUS"
[[ $STATUS == *FINISHED* ]]
$G run configs/ci/mock.yaml "${FLAGS[@]}" --skip-workload toolcall-mini --resume

echo "== resume: nothing left to do"
RESUME=$($G run configs/ci/mock.yaml "${FLAGS[@]}" --workload toolcall-mini --resume 2>&1)
[[ $RESUME == *"already complete"* ]]
python3 - "$OUT" <<'PY'
import json, sys, glob
runs = [json.load(open(p)) for p in glob.glob(sys.argv[1] + "/*/result.json")]
assert len(runs) == 2, runs
by = {tuple(sorted({p["workload"] for p in r["perf"]})): r for r in runs}
tool = by[("toolcall-mini",)]
assert tool["complete"] and tool["quality"][0]["recall_accuracy"] == 1.0, tool["quality"]
assert tool["capacity"][0]["max_users_slo"] >= 1
chat = by[("chat-32-32",)]
cap = chat["capacity"][0]
assert 16 <= cap["max_users_slo"] < 128 and len(cap["probed"]) > 3, cap  # bisection ran
assert all(r["software"]["runtime"] == "mock" for r in runs)
print("results OK:", cap["max_users_slo"], "users (SLO) on the mock")
PY

echo "== quality compare (a run against itself must PASS)"
TOOL_RUN=$(grep -l '"toolcall-mini"' "$OUT"/*/result.json | head -1 | xargs dirname)
$G quality compare "$TOOL_RUN" "$TOOL_RUN" --workload toolcall-mini

echo "== submit must refuse mock runs"
SUBMIT=$($G submit "$OUT"/*/ --dry-run --stage "$OUT/stage" 2>&1 || true)
if [[ $SUBMIT == *"mock runtime"* ]]; then
  echo "refused as expected"
else
  echo "submit did not refuse a mock run" >&2; exit 1
fi

echo "== serving variants + report"
VOUT=$(mktemp -d)
$G run configs/ci/mock-variants.yaml --provider ci --gpu-type H100-80GB --price-per-hour 1 \
  --runtime mock --no-fingerprint --out "$VOUT"
REPORT=$($G variants report "$VOUT"/*/ --baseline base --workload toolcall-mini --price ci=1)
echo "$REPORT" | head -8
[[ $REPORT == *"| alt |"* && $REPORT == *"| base |"* && $REPORT == *"PASS"* ]]

echo "== data repo build over these runs"
DATA=$(mktemp -d)
mkdir -p "$DATA/runs" "$DATA/prices"
for d in "$OUT"/*/; do [[ -f $d/result.json ]] && mkdir -p "$DATA/runs/$(basename "$d")" && cp "$d/result.json" "$DATA/runs/$(basename "$d")/"; done
printf 'offers:\n- {provider: ci, gpu_type: H100-80GB, usd_per_gpu_hour: 2.0, source: ci, checked_at: 2026-01-01}\n' > "$DATA/prices/gpu_hourly.yaml"
printf 'offers:\n- {model: silybench/mock-8B, provider: api, usd_per_1m_input: 0.1, usd_per_1m_output: 0.4, source: ci, fetched_at: 2026-01-01}\n' > "$DATA/prices/api.yaml"
$G dataset build --root "$DATA"
test -s "$DATA/derived/cost.json" && test -s "$DATA/derived/silybench.sqlite"
echo "AGENT FLOW OK"
