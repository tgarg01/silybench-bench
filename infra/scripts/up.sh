#!/usr/bin/env bash
# Bundle the repo, upload it to GCS and create a benchmark VM that runs CONFIG.
# GCP only (Terraform-managed Spot VM). For any other provider, see AGENTS.md.
#   infra/scripts/up.sh h100-1g configs/smoke.yaml [--keep] [--hours N] [--zone Z] [--yes] [--flex [WAIT]]
#                        [--run-args "--precision fp8 --workload chat-128-128 --skip-accuracy"]
# --keep:  don't delete the VM when the run ends (debugging). Always run down.sh after.
# --zone:  override the env's zone (e.g. when Spot capacity is exhausted in one zone).
# --yes:   skip Terraform's interactive approval.
# --flex:  queue a Flex-start VM (Dynamic Workload Scheduler) instead of Spot, waiting up to
#          WAIT (default 2h, the max) for capacity. Created with gcloud because Terraform can't
#          set the queue duration; find/delete it with logs.sh / down.sh as usual.
set -euo pipefail

ENV=${1:?usage: up.sh <env> <config> [--keep] [--hours N]}
CONFIG=${2:?usage: up.sh <env> <config> [--keep] [--hours N]}
shift 2
SELF_DELETE=true
HOURS=12
ZONE=""
APPROVE=""
FLEX=""
RUN_ARGS=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --keep) SELF_DELETE=false; shift ;;
    --hours) HOURS=$2; shift 2 ;;
    --zone) ZONE=$2; shift 2 ;;
    --yes) APPROVE=-auto-approve; shift ;;
    --run-args) RUN_ARGS=$2; shift 2 ;;
    --flex)
      FLEX=2h
      if [[ ${2:-} =~ ^[0-9]+[hms] ]]; then FLEX=$2; shift; fi
      shift ;;
    *) echo "unknown arg $1" >&2; exit 1 ;;
  esac
done

ROOT=$(cd "$(dirname "$0")/../.." && pwd)  # the silybench-bench checkout
TF="$ROOT/infra/terraform/envs/$ENV"
[[ -d $TF ]] || { echo "no such env: $TF" >&2; exit 1; }
[[ -f $ROOT/$CONFIG ]] || { echo "no such config: $CONFIG" >&2; exit 1; }

BUCKET=$(sed -n 's/^bucket *= *"\(.*\)"/\1/p' "$TF/terraform.tfvars")
PRICE=$(sed -n 's/^price_per_hour *= *\([0-9.]*\).*/\1/p' "$TF/terraform.tfvars")
MACHINE=$(sed -n 's/^ *machine_type *= *"\(.*\)"/\1/p' "$TF/main.tf")
CAMPAIGN=$(basename "$CONFIG" .yaml)
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
CODE_URI="gs://$BUCKET/code/$CAMPAIGN-$STAMP.tgz"

# Validate locally before paying for a GPU.
# (PYTHONPATH: macOS can flag the venv's .pth files "hidden" under ~/Desktop, which Python skips.)
# shellcheck disable=SC2086
(cd "$ROOT" && PYTHONPATH=. uv run python -m gpubench.cli validate "$ROOT/$CONFIG" $RUN_ARGS)

echo "bundling repo -> $CODE_URI"
# The bundle has no .git, so record the commit (+dirty flag) for result provenance.
COMMIT=$(git -C "$ROOT" rev-parse --verify -q HEAD || echo uncommitted)
git -C "$ROOT" diff --quiet HEAD 2>/dev/null || COMMIT="$COMMIT-dirty"
echo "$COMMIT" > "$ROOT/.gpubench-commit"
TARBALL=$(mktemp -t gpubench).tgz
COPYFILE_DISABLE=1 tar -czf "$TARBALL" -C "$ROOT" \
  --exclude .git --exclude .venv --exclude .terraform --exclude results \
  --exclude submission --exclude '*.tfstate*' --exclude terraform.tfvars \
  .
gcloud storage cp "$TARBALL" "$CODE_URI"
rm -f "$TARBALL" "$ROOT/.gpubench-commit"

ZONE=${ZONE:-$(sed -n 's/^zone *= *"\(.*\)"/\1/p' "$TF/terraform.tfvars")}
PROVISIONING=spot
[[ -n $FLEX ]] && PROVISIONING=flex-start
# Hardware provenance for result.json (GPUs themselves are detected on the VM).
RUN_ARGS="--provider gcp --provisioning $PROVISIONING --machine-type $MACHINE --zone $ZONE${PRICE:+ --price-per-hour $PRICE} $RUN_ARGS"

if [[ -n $FLEX ]]; then
  PROJECT=$(sed -n 's/^project_id *= *"\(.*\)"/\1/p' "$TF/terraform.tfvars")
  NAME="gpubench-$ENV"
  echo "queueing Flex-start $MACHINE $NAME in $ZONE (waits up to $FLEX for capacity)"
  gcloud compute instances create "$NAME" \
    --project "$PROJECT" --zone "$ZONE" --machine-type "$MACHINE" \
    --provisioning-model FLEX_START --request-valid-for-duration "$FLEX" \
    --max-run-duration "${HOURS}h" --instance-termination-action DELETE \
    --maintenance-policy TERMINATE --reservation-affinity none \
    --image-family common-cu129-ubuntu-2404-nvidia-580 --image-project deeplearning-platform-release \
    --boot-disk-size 200GB --boot-disk-type pd-balanced \
    --network-interface nic-type=GVNIC,network=default \
    --service-account "gpubench-runner@$PROJECT.iam.gserviceaccount.com" --scopes cloud-platform \
    --tags gpubench --labels "app=gpubench,campaign=$CAMPAIGN,provisioning=flex-start" \
    --metadata-from-file startup-script="$ROOT/infra/scripts/startup-script.sh" \
    --metadata "gpubench-code-uri=$CODE_URI,gpubench-config=$CONFIG,gpubench-bucket=$BUCKET,gpubench-self-delete=$SELF_DELETE,gpubench-run-args=$RUN_ARGS,enable-oslogin=TRUE"
else
  terraform -chdir="$TF" init -input=false -upgrade >/dev/null
  terraform -chdir="$TF" apply $APPROVE \
    -var "zone=$ZONE" \
    -var "campaign=$CAMPAIGN" \
    -var "code_uri=$CODE_URI" \
    -var "config_path=$CONFIG" \
    -var "max_run_hours=$HOURS" \
    -var "self_delete=$SELF_DELETE" \
    -var "run_args=$RUN_ARGS"
fi

echo
echo "VM is starting. Follow progress with: infra/scripts/logs.sh $ENV"
echo "Results will appear in gs://$BUCKET/runs/"
