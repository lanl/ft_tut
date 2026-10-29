#!/usr/bin/env bash
set -euo pipefail

OUTPUT_NOTEBOOK="${1:-${FMOD_RUNS_ROOT}/vertical_slice-executed.ipynb}"

mkdir -p "$(dirname "${OUTPUT_NOTEBOOK}")"
cd /workspace/tut

/workspace/tut/docker/bootstrap-assets.sh

exec jupyter nbconvert \
  --to notebook \
  --execute notebooks/vertical_slice.ipynb \
  --output "${OUTPUT_NOTEBOOK}" \
  --ExecutePreprocessor.kernel_name=spus-ft \
  --ExecutePreprocessor.timeout=-1
