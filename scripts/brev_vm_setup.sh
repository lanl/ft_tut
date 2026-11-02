#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------------------------
# Brev VM Mode setup for the SPUS fine-tuning tutorial
# ------------------------------------------------------------------------------

WORKSPACE_DIR="/home/ubuntu/workspace"
REPO_URL="${REPO_URL:-https://github.com/<your-org>/<your-repo>.git}"
REPO_NAME="${REPO_NAME:-$(basename "${REPO_URL%.git}")}"
REPO_DIR="${REPO_DIR:-$WORKSPACE_DIR/$REPO_NAME}"

UV_VERSION="${UV_VERSION:-0.13.6}"
UV_BIN_DIR="/home/ubuntu/.local/bin"
UV="$UV_BIN_DIR/uv"
export PATH="$UV_BIN_DIR:$PATH"

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
VENV_DIR="$REPO_DIR/.venv"

AUX_ROOT="${AUX_ROOT:-$WORKSPACE_DIR/tut-aux}"
DATA_ROOT="${DATA_ROOT:-$AUX_ROOT/spus-ft-data}"
MODELS_ROOT="${MODELS_ROOT:-$AUX_ROOT/spus-models}"
RUNS_ROOT="${RUNS_ROOT:-$AUX_ROOT/spus-ft-runs}"
CACHE_ROOT="${CACHE_ROOT:-$AUX_ROOT/spus-ft-cache}"

FNS_KF_DIR="$DATA_ROOT/FNS-KF"
ASSEMBLED_DATASET="$FNS_KF_DIR/FNS-KF.nc"
FNS_KF_ASSEMBLER="$FNS_KF_DIR/assemble_data.py"
PRETRAINED_MODEL_DIR="$MODELS_ROOT/spus-pde-unet-36m-v2"
FINETUNED_MODEL_DIR="$MODELS_ROOT/spus-pde-unet-36m-v2-fns-kf-256"

BOOTSTRAP_SPUS_ASSETS="${BOOTSTRAP_SPUS_ASSETS:-1}"
BOOTSTRAP_SPUS_MODELS="${BOOTSTRAP_SPUS_MODELS:-1}"
BOOTSTRAP_FNS_KF_DATA="${BOOTSTRAP_FNS_KF_DATA:-1}"

KERNEL_NAME="${KERNEL_NAME:-spus-ft}"
KERNEL_DISPLAY_NAME="${KERNEL_DISPLAY_NAME:-Python (spus-ft)}"
KERNEL_DIR="/home/ubuntu/.local/share/jupyter/kernels/$KERNEL_NAME"
KERNEL_LAUNCHER="/home/ubuntu/.local/bin/${KERNEL_NAME}-kernel"

mkdir -p "$WORKSPACE_DIR" "$UV_BIN_DIR" "$AUX_ROOT" "$DATA_ROOT" "$MODELS_ROOT" "$RUNS_ROOT" "$CACHE_ROOT"

echo ">>> 1. Installing uv..."
if [ ! -x "$UV" ] || ! "$UV" --version | grep -q "uv $UV_VERSION"; then
    curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_UNMANAGED_INSTALL="$UV_BIN_DIR" sh
fi
echo ">>> Using: $($UV --version)"

echo ">>> 2. Cloning tutorial repository..."
cd "$WORKSPACE_DIR"
if [ ! -d "$REPO_NAME/.git" ]; then
    git clone "$REPO_URL" "$REPO_NAME"
fi

cd "$REPO_DIR"

if [ ! -f "pyproject.toml" ] || [ ! -f "uv.lock" ]; then
    echo "ERROR: pyproject.toml and uv.lock must exist in the repo root."
    exit 1
fi

echo ">>> 3. Installing managed Python $PYTHON_VERSION..."
"$UV" python install "$PYTHON_VERSION"

echo ">>> 4. Creating project-local virtual environment..."
"$UV" venv --python "$PYTHON_VERSION" .venv

echo ">>> 5. Syncing exact locked dependencies..."
"$UV" sync --frozen --extra spus --extra notebook --extra dev

echo ">>> 6. Writing persistent tutorial environment..."
ENV_FILE="/home/ubuntu/.spus-ft-env.sh"
cat > "$ENV_FILE" <<EOF
export PATH="$UV_BIN_DIR:\$PATH"
export PYTHONPATH="$REPO_DIR/src"
export SPUS_FT_AUX_ROOT="$AUX_ROOT"
export FMOD_DATA_ROOT="$DATA_ROOT"
export FMOD_MODELS_ROOT="$MODELS_ROOT"
export FMOD_RUNS_ROOT="$RUNS_ROOT"
export FMOD_CACHE_ROOT="$CACHE_ROOT"
EOF
chmod +x "$ENV_FILE"

if ! grep -q "spus-ft-env.sh" /home/ubuntu/.bashrc; then
    echo "source \"$ENV_FILE\"" >> /home/ubuntu/.bashrc
fi

echo ">>> 7. Registering Jupyter kernel..."
PYTHON_MM=$(printf '%s' "$PYTHON_VERSION" | cut -d. -f1-2)
SITE_PACKAGES="$VENV_DIR/lib/python${PYTHON_MM}/site-packages"

CUDA_LIB_PATH=""
if [ -d "$SITE_PACKAGES/torch/lib" ]; then
    CUDA_LIB_PATH="$SITE_PACKAGES/torch/lib"
fi

if [ -d "$SITE_PACKAGES/nvidia" ]; then
    while IFS= read -r lib_dir; do
        if [ -n "$CUDA_LIB_PATH" ]; then
            CUDA_LIB_PATH="$CUDA_LIB_PATH:$lib_dir"
        else
            CUDA_LIB_PATH="$lib_dir"
        fi
    done < <(find "$SITE_PACKAGES/nvidia" -type d -name lib | sort)
fi

mkdir -p "$KERNEL_DIR" "$(dirname "$KERNEL_LAUNCHER")"

cat > "$KERNEL_LAUNCHER" <<EOF
#!/usr/bin/env bash
source "$ENV_FILE"
export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:\${LD_LIBRARY_PATH:-}"
exec "$VENV_DIR/bin/python" -m ipykernel_launcher "\$@"
EOF
chmod +x "$KERNEL_LAUNCHER"

cat > "$KERNEL_DIR/kernel.json" <<EOF
{
  "argv": [
    "$KERNEL_LAUNCHER",
    "-f",
    "{connection_file}"
  ],
  "display_name": "$KERNEL_DISPLAY_NAME",
  "language": "python"
}
EOF

echo ">>> 8. Bootstrapping SPUS assets..."
source "$ENV_FILE"

if [ -z "${HF_TOKEN:-}" ]; then
    echo ">>> HF_TOKEN is not set; Hugging Face downloads will be unauthenticated and may be slower or rate-limited"
fi

assemble_fns_kf_dataset() {
    if [ -f "$ASSEMBLED_DATASET" ]; then
        echo ">>> Found assembled dataset at $ASSEMBLED_DATASET"
        return 0
    fi

    if [ ! -f "$FNS_KF_ASSEMBLER" ]; then
        echo ">>> Cannot assemble $ASSEMBLED_DATASET because $FNS_KF_ASSEMBLER is missing"
        return 1
    fi

    if ! compgen -G "$FNS_KF_DIR/solution_*.nc" > /dev/null; then
        echo ">>> Cannot assemble $ASSEMBLED_DATASET because no raw solution_*.nc files were found"
        return 1
    fi

    echo ">>> Assembling FNS-KF.nc from raw solution files..."
    "$UV" run python "$FNS_KF_ASSEMBLER" \
        --input_dir "$FNS_KF_DIR" \
        --output_file "$ASSEMBLED_DATASET"

    if [ -f "$ASSEMBLED_DATASET" ]; then
        echo ">>> Assembled dataset at $ASSEMBLED_DATASET"
        return 0
    fi

    echo ">>> Assembly finished but $ASSEMBLED_DATASET is still missing"
    return 1
}

if [ "$BOOTSTRAP_SPUS_ASSETS" = "1" ]; then
    if [ "$BOOTSTRAP_FNS_KF_DATA" = "1" ]; then
        if [ -f "$ASSEMBLED_DATASET" ]; then
            echo ">>> Found assembled dataset at $ASSEMBLED_DATASET"
        else
            if compgen -G "$FNS_KF_DIR/solution_*.nc" > /dev/null; then
                echo ">>> Raw FNS-KF files already present in $FNS_KF_DIR"
                assemble_fns_kf_dataset
            else
                echo ">>> Downloading raw FNS-KF dataset contents..."
                "$UV" run python -m spus_ft download-fns-kf
                assemble_fns_kf_dataset
            fi
        fi
    fi

    if [ "$BOOTSTRAP_SPUS_MODELS" = "1" ]; then
        if [ -f "$PRETRAINED_MODEL_DIR/model.py" ] && [ -f "$FINETUNED_MODEL_DIR/model.py" ]; then
            echo ">>> Found SPUS model snapshots in $MODELS_ROOT"
        else
            echo ">>> Downloading SPUS model snapshots..."
            "$UV" run python -m spus_ft download-spus-models
        fi
    fi
else
    echo ">>> Asset bootstrap disabled"
fi

echo ">>> 9. Fixing permissions..."
sudo chown -R ubuntu:ubuntu \
    "$WORKSPACE_DIR" \
    "$UV_BIN_DIR" \
    "$KERNEL_DIR" \
    "$KERNEL_LAUNCHER" \
    "$ENV_FILE"

echo ">>> SETUP COMPLETE"
echo ">>> In Jupyter, select kernel: $KERNEL_DISPLAY_NAME"
echo ">>> Repo: $REPO_DIR"
echo ">>> Aux data: $AUX_ROOT"
