#!/usr/bin/env bash
set -euo pipefail

export JUPYTER_PORT="${JUPYTER_PORT:-8888}"
export JUPYTER_TOKEN="${JUPYTER_TOKEN:-}"
export JUPYTER_ROOT="${JUPYTER_ROOT:-/workspace/tut}"

mkdir -p \
  "${SPUS_FT_AUX_ROOT}" \
  "${FMOD_DATA_ROOT}" \
  "${FMOD_MODELS_ROOT}" \
  "${FMOD_RUNS_ROOT}" \
  "${FMOD_CACHE_ROOT}"

cd /workspace/tut

exec jupyter lab \
  --ip=0.0.0.0 \
  --port="${JUPYTER_PORT}" \
  --no-browser \
  --allow-root \
  --ServerApp.root_dir="${JUPYTER_ROOT}" \
  --ServerApp.preferred_dir="${JUPYTER_ROOT}" \
  --ServerApp.token="${JUPYTER_TOKEN}"
