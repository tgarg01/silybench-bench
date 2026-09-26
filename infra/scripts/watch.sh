#!/usr/bin/env bash
# Keep a long GCP Spot campaign going: launch it, and every 5 min check that the VM is alive.
# If Spot preempted it (VM gone, campaign not DONE), relaunch; the new VM pulls the mirrored
# results from GCS, re-verifies it is the same hardware, and `gpubench run --resume` continues.
#
#   infra/scripts/watch.sh h100-1g configs/qwen3.8-27b-h100.yaml [--experiment ID] \
#     [--phase "--workload toolcall-100k-512"] [--phase "--skip-workload toolcall-100k-512"] \
#     [--max-relaunches 10] [other up.sh args...]
#
# Each --phase is a set of `gpubench run` filters, run in order (one VM launch each); without
# --phase the whole campaign runs at once. Runs of all phases are merged when published.
set -euo pipefail
ENV=${1:?usage: watch.sh <env> <config> [--phase ARGS]... [--max-relaunches N] [up.sh args]}
CONFIG=${2:?usage: watch.sh <env> <config> [--phase ARGS]... [--max-relaunches N] [up.sh args]}
shift 2
MAX=10
MAX_MISMATCH=3
PHASES=()
UP_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-relaunches) MAX=$2; shift 2 ;;
    --phase) PHASES+=("$2"); shift 2 ;;
    *) UP_ARGS+=("$1"); shift ;;
  esac
done
[[ ${#PHASES[@]} -gt 0 ]] || PHASES=("")

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TF="$ROOT/infra/terraform/envs/$ENV"
BUCKET=$(sed -n 's/^bucket *= *"\(.*\)"/\1/p' "$TF/terraform.tfvars")
NAME="gpubench-$ENV"
CAMPAIGN=$(cd "$ROOT" && PYTHONPATH=. uv run python -c "from gpubench.config import load_config; print(load_config('$ROOT/$CONFIG').name)")
STATE="gs://$BUCKET/campaign-state/$CAMPAIGN"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

run_phase() {
  local phase_args=$1 launches=0 mismatches=0
  local up=("$ROOT/infra/scripts/up.sh" "$ENV" "$CONFIG" --yes "${UP_ARGS[@]}")
  [[ -n $phase_args ]] && up+=(--run-args "$phase_args")
  log "phase: ${phase_args:-whole campaign}"
  "${up[@]}"
  while true; do
    sleep 300
    if gcloud storage ls "$STATE/DONE" >/dev/null 2>&1; then
      log "phase finished: ${phase_args:-whole campaign}"
      return 0
    fi
    if gcloud storage ls "$STATE/FAILED" >/dev/null 2>&1; then
      log "campaign FAILED (not a preemption): read the log before relaunching:"
      log "  gcloud storage ls gs://$BUCKET/logs/"
      exit 1
    fi
    if gcloud storage ls "$STATE/HWMISMATCH" >/dev/null 2>&1; then
      gcloud storage rm "$STATE/HWMISMATCH" >/dev/null 2>&1 || true
      mismatches=$((mismatches + 1))
      if (( mismatches > MAX_MISMATCH )); then
        log "HARDWARE MISMATCH $mismatches times: GCP keeps placing the VM on hardware that"
        log "differs from the reference. Try again later, or choose another experiment/GPU."
        exit 3
      fi
      log "VM hardware differed from the reference (mismatch $mismatches/$MAX_MISMATCH); trying another host"
    fi
    if [[ -n $(gcloud compute instances list --filter="name=$NAME" --format="value(name)") ]]; then
      progress=$(gcloud storage cat "$STATE/progress.json" 2>/dev/null \
        | python3 -c "import json,sys; s=json.load(sys.stdin); print(s.get('stage'), s.get('detail'), f\"{s.get('points_done')}/{s.get('planned_points')}\")" 2>/dev/null || echo "starting")
      log "running: $progress"
      continue
    fi
    if (( launches >= MAX )); then
      log "VM gone and $MAX relaunches used up; check logs: gcloud storage ls gs://$BUCKET/logs/"
      exit 1
    fi
    launches=$((launches + 1))
    log "VM gone before the phase finished (Spot preemption?): relaunch $launches/$MAX"
    "$ROOT/infra/scripts/down.sh" "$ENV" >/dev/null 2>&1 || true
    "${up[@]}" --relaunch
  done
}

for phase in "${PHASES[@]}"; do
  run_phase "$phase"
done
log "campaign $CAMPAIGN: all phases finished"
