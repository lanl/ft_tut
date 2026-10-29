#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_SPUS_ASSETS="${BOOTSTRAP_SPUS_ASSETS:-1}"
BOOTSTRAP_SPUS_MODELS="${BOOTSTRAP_SPUS_MODELS:-1}"
BOOTSTRAP_FNS_KF_DATA="${BOOTSTRAP_FNS_KF_DATA:-1}"

if [[ "${BOOTSTRAP_SPUS_ASSETS}" != "1" ]]; then
  echo "[bootstrap-assets] Asset bootstrap disabled; skipping downloads."
  exit 0
fi

cd /workspace/tut

fns_kf_dir="${FMOD_DATA_ROOT}/FNS-KF"
assembled_dataset="${fns_kf_dir}/FNS-KF.nc"
pretrained_model_dir="${FMOD_MODELS_ROOT}/spus-pde-unet-36m-v2"
fine_tuned_model_dir="${FMOD_MODELS_ROOT}/spus-pde-unet-36m-v2-fns-kf-256"

if [[ "${BOOTSTRAP_FNS_KF_DATA}" == "1" ]]; then
  if [[ -f "${assembled_dataset}" ]]; then
    echo "[bootstrap-assets] Found assembled dataset at ${assembled_dataset}."
  else
    if compgen -G "${fns_kf_dir}/solution_*.nc" > /dev/null; then
      echo "[bootstrap-assets] Raw FNS-KF files already present in ${fns_kf_dir}."
      echo "[bootstrap-assets] Note: assembled dataset ${assembled_dataset} is still missing."
    else
      echo "[bootstrap-assets] Downloading raw FNS-KF dataset contents into ${fns_kf_dir}."
      python -m spus_ft download-fns-kf
      if [[ -f "${assembled_dataset}" ]]; then
        echo "[bootstrap-assets] Found assembled dataset at ${assembled_dataset}."
      else
        echo "[bootstrap-assets] Download completed, but assembled dataset ${assembled_dataset} is still missing."
        echo "[bootstrap-assets] The downloaded repo contents include the raw solution files and assemble_data.py."
      fi
    fi
  fi
fi

if [[ "${BOOTSTRAP_SPUS_MODELS}" == "1" ]]; then
  if [[ -f "${pretrained_model_dir}/model.py" && -f "${fine_tuned_model_dir}/model.py" ]]; then
    echo "[bootstrap-assets] Found SPUS model snapshots in ${FMOD_MODELS_ROOT}."
  else
    echo "[bootstrap-assets] Downloading SPUS model snapshots into ${FMOD_MODELS_ROOT}."
    python -m spus_ft download-spus-models
  fi
fi
