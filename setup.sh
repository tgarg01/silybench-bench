#!/usr/bin/env bash
# One-time setup on a rented GPU box (RunPod, Vast, Lambda, Hyperbolic, bare metal, ...).
# Safe to re-run. Installs uv, Python deps (incl. lm-eval), gh and tmux; picks the vLLM runtime
# (Docker if it can reach the GPUs, otherwise a pip-installed vLLM); then runs `gpubench doctor`.
#
#   ./setup.sh                          # setup + doctor
#   ./setup.sh configs/qwen3-8b.yaml    # also pre-install that campaign's vLLM (native runtime)
set -euo pipefail
cd "$(dirname "$0")"
CONFIG=${1:-}

SUDO=""
if [[ $(id -u) -ne 0 ]] && command -v sudo >/dev/null; then SUDO=sudo; fi
have() { command -v "$1" >/dev/null 2>&1; }
APT_UPDATED=""
apt_install() {
  if ! have apt-get; then echo "  (no apt-get: install $* yourself)"; return 1; fi
  if [[ -z $APT_UPDATED ]]; then $SUDO apt-get update -qq; APT_UPDATED=1; fi
  DEBIAN_FRONTEND=noninteractive $SUDO apt-get install -yqq "$@" >/dev/null
}

echo "== GPUs"
if ! nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; then
  echo "ERROR: nvidia-smi failed: this machine has no usable NVIDIA GPU" >&2
  exit 1
fi

echo "== Base tools"
for tool in curl git; do have "$tool" || apt_install "$tool"; done
have tmux || apt_install tmux || true

echo "== uv"
if ! have uv; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "== Big-file location (model weights, vLLM, wheel cache)"
CACHE=$(uv run --quiet --python 3.11 --no-project python -c "import sys; sys.path.insert(0, '.'); \
from gpubench.paths import cache_root; print(cache_root())" 2>/dev/null || echo "$HOME/.cache/silybench")
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CACHE/uv}
echo "  $CACHE ($(df -h "$(dirname "$CACHE")" | awk 'NR==2 {print $4}') free)"

echo "== Python deps (gpubench + lm-eval)"
uv sync --extra eval --python 3.11 --quiet

echo "== GitHub CLI (for \`gpubench submit\`)"
if ! have gh; then
  if have apt-get; then
    have gpg || apt_install gnupg
    $SUDO mkdir -p -m 755 /etc/apt/keyrings
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | $SUDO tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null
    $SUDO chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | $SUDO tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    APT_UPDATED=""
    apt_install gh || true
  else
    echo "  no apt-get: install gh from https://cli.github.com (needed only to submit)"
  fi
fi
have gh && gh --version | head -1

echo "== vLLM runtime"
RUNTIME=$(uv run --quiet python -c "from gpubench.server import pick_runtime; print(pick_runtime())")
echo "  $RUNTIME"
if [[ $RUNTIME == native && -n $CONFIG ]]; then
  echo "  installing the campaign's vLLM (a few minutes, several GB)"
  uv run gpubench install-engine "$CONFIG"
elif [[ $RUNTIME == docker && -n $CONFIG ]]; then
  IMAGE=$(uv run --quiet python -c "from gpubench.config import load_config; print(load_config('$CONFIG').engine.image)")
  docker pull -q "$IMAGE"
fi

echo "== Doctor"
uv run gpubench doctor ${CONFIG:+--config "$CONFIG"} || true
echo
echo "Setup done. Next: uv run gpubench plan <config> --price-per-hour <USD/h>"
