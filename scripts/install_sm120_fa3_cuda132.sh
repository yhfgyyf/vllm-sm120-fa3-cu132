#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.2}"
CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
NVCC_BIN="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
FLASHINFER_NVCC="${FLASHINFER_NVCC:-$NVCC_BIN}"
FLASH_ATTENTION_REPO="${FLASH_ATTENTION_REPO:-https://github.com/yhfgyyf/flash-attention-sm120-fa3-progress.git}"
FLASH_ATTENTION_REF="${FLASH_ATTENTION_REF:-cee537ecf73d002dfeedc626812eeda100c16b07}"
FLASH_ATTENTION_DIR="${FLASH_ATTENTION_DIR:-$ROOT_DIR/.vendor/flash-attention-sm120-fa3}"
MAX_JOBS="${MAX_JOBS:-8}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

if [[ ! -x "$NVCC_BIN" ]]; then
  echo "nvcc not found at: $NVCC_BIN" >&2
  exit 1
fi

export CUDA_HOME CUDA_PATH CUDACXX="$NVCC_BIN" FLASHINFER_NVCC PATH="$CUDA_HOME/bin:$PATH"

mkdir -p "$(dirname "$FLASH_ATTENTION_DIR")"
if [[ ! -d "$FLASH_ATTENTION_DIR/.git" ]]; then
  git clone "$FLASH_ATTENTION_REPO" "$FLASH_ATTENTION_DIR"
fi

git -C "$FLASH_ATTENTION_DIR" fetch --all --tags
git -C "$FLASH_ATTENTION_DIR" checkout "$FLASH_ATTENTION_REF"

pushd "$FLASH_ATTENTION_DIR/hopper" >/dev/null
uv run --python "$PYTHON_BIN" generate_kernels.py -o instantiations

CUDA_HOME="$CUDA_HOME" \
CUDA_PATH="$CUDA_PATH" \
PYTORCH_NVCC="$NVCC_BIN" \
CUDACXX="$NVCC_BIN" \
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_SM120_ONLY=TRUE \
FLASH_ATTENTION_ALLOW_CUDA_VERSION_MISMATCH=TRUE \
FLASH_ATTENTION_DISABLE_BACKWARD=TRUE \
FLASH_ATTENTION_DISABLE_FP8=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF64=TRUE \
FLASH_ATTENTION_DISABLE_HDIMDIFF192=TRUE \
MAX_JOBS="$MAX_JOBS" \
uv pip install --python "$PYTHON_BIN" --no-build-isolation --force-reinstall .
popd >/dev/null

pushd "$ROOT_DIR" >/dev/null
CUDA_HOME="$CUDA_HOME" \
CUDA_PATH="$CUDA_PATH" \
CUDACXX="$NVCC_BIN" \
uv pip install --python "$PYTHON_BIN" -e . --torch-backend=auto
popd >/dev/null

cat <<EOF
Install complete.

CUDA_HOME=$CUDA_HOME
PYTHON_BIN=$PYTHON_BIN
FLASH_ATTENTION_DIR=$FLASH_ATTENTION_DIR
FLASH_ATTENTION_REF=$FLASH_ATTENTION_REF
EOF
