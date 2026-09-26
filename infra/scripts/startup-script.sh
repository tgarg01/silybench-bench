#!/usr/bin/env bash
# Runs as root on the benchmark VM at boot (GCE startup script).
# Pulls the code bundle, runs the configured campaign, uploads results and logs
# to GCS, then deletes the VM. Output: `sudo journalctl -u google-startup-scripts -f`.
set -euo pipefail

MD=http://metadata.google.internal/computeMetadata/v1/instance
md() { curl -sf -H "Metadata-Flavor: Google" "$MD/$1"; }

NAME=$(md name)
ZONE=$(md zone | awk -F/ '{print $NF}')
CODE_URI=$(md attributes/gpubench-code-uri)
CONFIG=$(md attributes/gpubench-config)
BUCKET=$(md attributes/gpubench-bucket)
SELF_DELETE=$(md attributes/gpubench-self-delete || echo true)
RUN_ARGS=$(md attributes/gpubench-run-args || true)  # e.g. "--precision fp8"

WORK=/opt/gpubench
LOG=/var/log/gpubench.log
MARKER=/var/lib/gpubench.started

if [[ -f $MARKER ]]; then
  echo "gpubench already ran on this VM; not re-running on reboot"
  exit 0
fi
touch "$MARKER"
# -p: if the console/log-forwarder pipe breaks, keep writing the log file instead of dying
# (a broken pipe here once killed a whole run mid-benchmark).
exec > >(tee -p -a "$LOG") 2>&1

finish() {
  local code=$?
  echo "=== gpubench finished with exit code $code at $(date -u +%FT%TZ)"
  gcloud storage cp "$LOG" "gs://$BUCKET/logs/$NAME-$(date -u +%Y%m%dT%H%M%SZ).log" || true
  STATE="gs://$BUCKET/campaign-state/${CAMPAIGN:-unknown}"
  if [[ $code -eq 0 ]]; then
    echo "done" | gcloud storage cp - "$STATE/DONE" || true
  elif (( code < 129 || code > 143 )); then
    # A real failure (not a preemption signal): tell watch.sh not to pay for a relaunch loop.
    echo "exit $code" | gcloud storage cp - "$STATE/FAILED" || true
  fi
  if [[ $SELF_DELETE == "true" ]]; then
    echo "deleting VM $NAME"
    gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
  fi
}
trap finish EXIT
# Turn fatal signals into a normal exit so `finish` still uploads logs and deletes the VM.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 141' PIPE
trap 'exit 143' TERM

echo "=== gpubench start $(date -u +%FT%TZ) on $NAME ($ZONE), config=$CONFIG"
nvidia-smi

# Docker + NVIDIA container runtime (present on Deep Learning VM images; install if missing).
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker info 2>/dev/null | grep -q nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q && apt-get install -yq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
fi

# Code bundle uploaded by up.sh.
rm -rf "$WORK" && mkdir -p "$WORK"
gcloud storage cp "$CODE_URI" /tmp/gpubench.tgz
tar -xzf /tmp/gpubench.tgz -C "$WORK"

# Optional HF token (needed for gated datasets such as GPQA).
if HF_TOKEN=$(gcloud secrets versions access latest --secret=hf-token 2>/dev/null); then
  export HF_TOKEN
  echo "HF token loaded from Secret Manager"
else
  echo "no hf-token secret version; continuing without HF auth"
fi

export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$WORK"
uv sync --extra eval --python 3.11

IMAGE=$(uv run python -c "from gpubench.config import load_config; print(load_config('$WORK/$CONFIG').engine.image)")
docker pull "$IMAGE"

mkdir -p /opt/hf-cache /opt/gpubench-results
# Resume after a Spot preemption: pull back every run directory mirrored so far (result.json,
# raw bench JSON, telemetry, logs) plus the campaign's fingerprint; `--resume` then skips all
# finished work. On a first launch there is nothing to pull.
CAMPAIGN=$(uv run python -c "from gpubench.config import load_config; print(load_config('$WORK/$CONFIG').name)")
gcloud storage rsync -r "gs://$BUCKET/runs" /opt/gpubench-results || true
gcloud storage cp "gs://$BUCKET/campaign-state/$CAMPAIGN/fingerprint.json" /opt/gpubench-results/ 2>/dev/null || true
# RUN_ARGS carries --provider gcp --price-per-hour ... (set by up.sh) plus any filters.
# shellcheck disable=SC2086  # RUN_ARGS is intentionally word-split into flags
uv run gpubench run "$WORK/$CONFIG" \
  --runtime docker \
  --resume \
  --bucket "$BUCKET" \
  --out /opt/gpubench-results \
  --hf-cache /opt/hf-cache \
  $RUN_ARGS
