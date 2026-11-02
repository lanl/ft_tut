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
fns_kf_assembler="${fns_kf_dir}/assemble_data.py"
pretrained_model_dir="${FMOD_MODELS_ROOT}/spus-pde-unet-36m-v2"
fine_tuned_model_dir="${FMOD_MODELS_ROOT}/spus-pde-unet-36m-v2-fns-kf-256"

assemble_fns_kf_dataset() {
  if [[ -f "${assembled_dataset}" ]]; then
    echo "[bootstrap-assets] Found assembled dataset at ${assembled_dataset}."
    return 0
  fi

  if [[ ! -f "${fns_kf_assembler}" ]]; then
    echo "[bootstrap-assets] Cannot assemble ${assembled_dataset} because ${fns_kf_assembler} is missing."
    return 1
  fi

  if ! compgen -G "${fns_kf_dir}/solution_*.nc" > /dev/null; then
    echo "[bootstrap-assets] Cannot assemble ${assembled_dataset} because no raw solution_*.nc files were found."
    return 1
  fi

  echo "[bootstrap-assets] Assembling FNS-KF.nc from raw solution files."
  python "${fns_kf_assembler}" \
    --input_dir "${fns_kf_dir}" \
    --output_file "${assembled_dataset}"

  if [[ -f "${assembled_dataset}" ]]; then
    echo "[bootstrap-assets] Assembled dataset at ${assembled_dataset}."
    return 0
  fi

  echo "[bootstrap-assets] Assembly finished, but ${assembled_dataset} is still missing."
  return 1
}

if [[ "${BOOTSTRAP_FNS_KF_DATA}" == "1" ]]; then
  if [[ -f "${assembled_dataset}" ]]; then
    echo "[bootstrap-assets] Found assembled dataset at ${assembled_dataset}."
  else
    if compgen -G "${fns_kf_dir}/solution_*.nc" > /dev/null; then
      echo "[bootstrap-assets] Raw FNS-KF files already present in ${fns_kf_dir}."
      assemble_fns_kf_dataset
    else
      echo "[bootstrap-assets] Downloading raw FNS-KF dataset contents into ${fns_kf_dir}."
      python -m spus_ft download-fns-kf
      assemble_fns_kf_dataset
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
