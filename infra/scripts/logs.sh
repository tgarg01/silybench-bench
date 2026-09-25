#!/usr/bin/env bash
# Stream the benchmark log from the VM over IAP SSH (works for Spot and Flex-start VMs).
#   infra/scripts/logs.sh h100-1g
set -euo pipefail
ENV=${1:?usage: logs.sh <env>}
NAME="gpubench-$ENV"
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
[[ -n $ZONE ]] || { echo "no VM named $NAME (not created yet, or already finished and deleted)" >&2; exit 1; }
gcloud compute ssh "$NAME" --zone "$ZONE" --tunnel-through-iap \
  --command "sudo journalctl -u google-startup-scripts -f -o cat"
