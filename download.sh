#!/bin/bash
set -euo pipefail

ASSET_BUCKET_ID="${HF_ASSET_BUCKET_ID:-mobilint/aries-weapon-detection-demo-assets}"
ASSET_BUCKET_URI="hf://buckets/${ASSET_BUCKET_ID}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${HF_DOWNLOAD_VENV_DIR:-$SCRIPT_DIR/.hf_venv}"
LOCAL_DIR="$SCRIPT_DIR/backend_vision/assets"

UV_HOME="${UV_USER_HOME:-}"
if [ -z "$UV_HOME" ] && [ "${SUDO_USER-}" ] && [ "$SUDO_USER" != "root" ]; then
  UV_HOME="$(getent passwd "$SUDO_USER" | cut -d: -f6)"
fi

if [ -n "$UV_HOME" ] && [ -d "$UV_HOME/.local/bin" ]; then
  export PATH="$UV_HOME/.local/bin:$PATH"
fi

if [ -d "$HOME/.local/bin" ]; then
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Please install uv before running this script."
  echo "See: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

echo "Preparing Hugging Face download environment: $VENV_DIR"
if [ ! -d "$VENV_DIR" ]; then
  uv venv "$VENV_DIR"
fi

if [ -x "$VENV_DIR/bin/python" ]; then
  VENV_PYTHON="$VENV_DIR/bin/python"
else
  echo "Cannot find venv python at $VENV_DIR/bin/python"
  exit 1
fi

uv pip install --python "$VENV_PYTHON" huggingface-hub

HF_CLI="$VENV_DIR/bin/hf"
if [ ! -x "$HF_CLI" ]; then
  echo "Cannot find hf CLI at $HF_CLI"
  exit 1
fi

mkdir -p "$LOCAL_DIR"

echo "Downloading vision assets from Hugging Face bucket: $ASSET_BUCKET_URI"
"$HF_CLI" buckets sync "$ASSET_BUCKET_URI" "$LOCAL_DIR" \
  --include "config/*.yaml" \
  --include "layout/*" \
  --include "mxq/*" \
  --include "video/**/*.mp4"

echo "Vision assets downloaded to $LOCAL_DIR"