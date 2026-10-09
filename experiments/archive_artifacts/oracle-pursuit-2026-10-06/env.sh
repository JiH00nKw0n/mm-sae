#!/usr/bin/env bash
set -euo pipefail
export TASK_ROOT=/home/elicer/jihoonkwon/mm_sae_grahtp_20261006
export TMPDIR="$TASK_ROOT/tmp"
export TEMP="$TASK_ROOT/tmp"
export TMP="$TASK_ROOT/tmp"
export XDG_CACHE_HOME="$TASK_ROOT/cache/xdg"
export XDG_CONFIG_HOME="$TASK_ROOT/config/xdg"
export XDG_DATA_HOME="$TASK_ROOT/data/xdg"
export XDG_STATE_HOME="$TASK_ROOT/config/state"
export PYTHONPYCACHEPREFIX="$TASK_ROOT/cache/pycache"
export MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TORCH_HOME="$TASK_ROOT/cache/torch"
export TORCH_EXTENSIONS_DIR="$TASK_ROOT/cache/torch_extensions"
export TRITON_CACHE_DIR="$TASK_ROOT/cache/triton"
export CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export UV_CACHE_DIR="$TASK_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$TASK_ROOT/python"
export PYTHONPATH="$TASK_ROOT/code/src:$TASK_ROOT/code"
export CUDA_VISIBLE_DEVICES=0
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$PYTHONPYCACHEPREFIX" "$MPLCONFIGDIR" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"
cd "$TASK_ROOT/code"
