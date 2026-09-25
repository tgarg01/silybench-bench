#!/usr/bin/env bash
# Delete the benchmark VM for an env: Terraform-managed (Spot) and/or gcloud-created (Flex-start).
# Safe to run when the VM already self-deleted.
#   infra/scripts/down.sh h100-1g
set -euo pipefail
ENV=${1:?usage: down.sh <env>}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
TF="$ROOT/infra/terraform/envs/$ENV"
NAME="gpubench-$ENV"
terraform -chdir="$TF" destroy -auto-approve \
  -var campaign=x -var code_uri=x -var config_path=x
ZONE=$(gcloud compute instances list --filter="name=$NAME" --format="value(zone.basename())")
if [[ -n $ZONE ]]; then
  gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
fi
