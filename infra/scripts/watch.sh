#!/usr/bin/env bash
# Keep a long GCP Spot campaign going: launch it, and every 5 min check that the VM is alive.
# If Spot preempted it (VM gone, campaign not DONE), relaunch; the new VM pulls the mirrored
# results from GCS and `gpubench run --resume` continues where the old one stopped.
#   infra/scripts/watch.sh h100-1g configs/qwen3.8-27b-h100.yaml [--max-relaunches 10] [up.sh args...]
set -euo pipefail
ENV=${1:?usage: watch.sh <env> <config> [--max-relaunches N] [up.sh args]}
CONFIG=${2:?usage: watch.sh <env> <config> [--max-relaunches N] [up.sh args]}
shift 2
MAX=10
UP_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --max-relaunches) MAX=$2; shift 2 ;;
    *) UP_ARGS+=("$1"); shift ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TF="$ROOT/infra/terraform/envs/$ENV"
BUCKET=$(sed -n 's/^bucket *= *"\(.*\)"/\1/p' "$TF/terraform.tfvars")
NAME="gpubench-$ENV"
CAMPAIGN=$(cd "$ROOT" && PYTHONPATH=. uv run python -c "from gpubench.config import load_config; print(load_config('$ROOT/$CONFIG').name)")
DONE="gs://$BUCKET/campaign-state/$CAMPAIGN/DONE"
log() { echo "[$(date -u +%FT%TZ)] $*"; }

"$ROOT/infra/scripts/up.sh" "$ENV" "$CONFIG" --yes "${UP_ARGS[@]}"
launches=0
while true; do
  sleep 300
  if gcloud storage ls "$DONE" >/dev/null 2>&1; then
    log "campaign $CAMPAIGN finished"
    exit 0
  fi
  if gcloud storage ls "gs://$BUCKET/campaign-state/$CAMPAIGN/FAILED" >/dev/null 2>&1; then
    log "campaign $CAMPAIGN FAILED (not a preemption): read the log before relaunching:"
    log "  gcloud storage ls gs://$BUCKET/logs/"
    exit 1
  fi
  if [[ -n $(gcloud compute instances list --filter="name=$NAME" --format="value(name)") ]]; then
    progress=$(gcloud storage cat "gs://$BUCKET/campaign-state/$CAMPAIGN/progress.json" 2>/dev/null \
      | python3 -c "import json,sys; s=json.load(sys.stdin); print(s.get('stage'), s.get('detail'), f\"{s.get('points_done')}/{s.get('planned_points')}\")" 2>/dev/null || echo "starting")
    log "running: $progress"
    continue
  fi
  if (( launches >= MAX )); then
    log "VM gone and $MAX relaunches used up; check logs: gcloud storage ls gs://$BUCKET/logs/"
    exit 1
  fi
  launches=$((launches + 1))
  log "VM gone before the campaign finished (Spot preemption?): relaunch $launches/$MAX"
  "$ROOT/infra/scripts/down.sh" "$ENV" >/dev/null 2>&1 || true
  "$ROOT/infra/scripts/up.sh" "$ENV" "$CONFIG" --yes --relaunch "${UP_ARGS[@]}"
done
