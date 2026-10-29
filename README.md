# FMOD Tutorial Scaffold

**Foundation Model Fine-Tuning Tutorial&#x20;**&#x68;as been acknowledged by NNSA for open-source release. O# **(O5229)**

This repository contains the Phase 0 and Phase 1 scaffold for a scientist-facing
tutorial on running pretrained foundation models, fine-tuning them, and comparing
their results on PDE-style time series data.

The current scaffold focuses on a synthetic end-to-end workflow so the software
shape is tested before wiring in the real SPUS model and FNS-KF dataset.

## Current scope

- Python package under `src/spus_ft/`
- External path management for data, cache, and run outputs
- Synthetic dataset generator for autoregressive forecasting
- Minimal baseline model wrapper
- Autoregressive inference loop
- MSE metrics and result serialization
- Starter Jupyter notebook that calls Python package functions
- Starter Dockerfile for notebook-based development
- Pytest suite for the scaffold

## Repository layout

- `src/spus_ft/` — tutorial package code
- `tests/` — unit tests for Phase 0 and Phase 1
- `notebooks/` — scientist-facing notebook entrypoint
- `configs/` — small JSON configuration files
- `docker/` — container scaffold

## External directories

The code keeps large artifacts outside git by default under a separate auxiliary
directory outside the repository:

- aux root: `/usr/projects/artimis/<user>/tut-aux` when available
- data root: `<aux-root>/spus-ft-data`
- models root: `<aux-root>/spus-models`
- run root: `<aux-root>/spus-ft-runs`
- cache root: `<aux-root>/spus-ft-cache`

These can be overridden with environment variables:

- `SPUS_FT_AUX_ROOT`
- `FMOD_DATA_ROOT`
- `FMOD_MODELS_ROOT`
- `FMOD_RUNS_ROOT`
- `FMOD_CACHE_ROOT`

## Quickstart

Create a virtual environment, install dev dependencies, and run tests:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

Run the synthetic notebook workflow from Python:

```bash
PYTHONPATH=src python -m spus_ft
```

For the real SPUS model path, install the model runtime dependencies first:

```bash
uv sync --extra spus
```

To add Jupyter notebook capability to the same `uv` environment:

```bash
uv sync --extra notebook
```

For the full tutorial environment on a backend node, including the SPUS runtime
stack and notebook support:

```bash
uv sync --extra spus --extra notebook --extra dev
```

If you want this environment to appear as a Jupyter kernel, register it once:

```bash
uv run python -m ipykernel install --user --name spus-ft --display-name "Python (spus-ft)"
```

Then launch JupyterLab from the repo root with:

```bash
PYTHONPATH=src uv run jupyter lab
```

This `spus` extra also includes `matplotlib`, so the same environment can run
the notebook plotting helpers.

## Container / NVIDIA Brev

The repository now includes a Brev-friendly container scaffold under `docker/`
for running the tutorial notebooks and Python package in an isolated JupyterLab
environment.

The container image:

- installs the locked project environment with `uv sync --frozen`
- includes the `spus`, `notebook`, and `dev` extras
- registers a `Python (spus-ft)` Jupyter kernel
- keeps data, model snapshots, cache, and run outputs under `/workspace/tut-aux`
- exposes JupyterLab on port `8888`

Build the image from the repo root:

```bash
docker build -f docker/Dockerfile -t spus-ft-brev .
```

Use Docker Compose from the repo root as the default launch path:

```bash
docker compose up --build jupyter
```

The included `docker-compose.yml` provides two services:

- `jupyter` — interactive JupyterLab on port `8888`
- `vertical-slice` — batch execution of `notebooks/vertical_slice.ipynb`

Examples:

```bash
docker compose up --build jupyter
docker compose run --rm vertical-slice
```

By default, Compose mounts:

- the repo at `/workspace/tut`
- `../tut-aux` at `/workspace/tut-aux`

You can override the aux mount and token with environment variables:

```bash
export TUT_AUX_DIR=/absolute/path/to/tut-aux
export JUPYTER_TOKEN=brev
docker compose up --build jupyter
```

As a fallback, start JupyterLab locally or on a Brev instance with `docker run`:

```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$PWD":/workspace/tut \
  -v "$PWD/../tut-aux":/workspace/tut-aux \
  -e JUPYTER_TOKEN=brev \
  spus-ft-brev
```

Then open the JupyterLab URL printed in the logs and launch:

- `notebooks/vertical_slice.ipynb`
- `notebooks/01_phase0_phase1_scaffold.ipynb`

The container startup command is `docker/start-jupyter.sh`. It creates the
runtime directories, points JupyterLab at `/workspace/tut`, and uses the
mounted `/workspace/tut-aux` directory for datasets, model snapshots, cache,
and run outputs.

By default, container startup now also runs `docker/bootstrap-assets.sh`, which
downloads missing SPUS model snapshots and missing raw FNS-KF dataset contents
into the mounted aux directory before Jupyter starts. You can disable that with:

```bash
export BOOTSTRAP_SPUS_ASSETS=0
docker compose up --build jupyter
```

The bootstrap step does not currently assemble `FNS-KF.nc` automatically. If
that assembled NetCDF file is missing, startup logs a reminder after downloading
the raw FNS-KF files.

To execute the vertical-slice notebook non-interactively inside the container,
prefer Compose:

```bash
docker compose run --rm vertical-slice
```

As a fallback, use the helper script with `docker run`:

```bash
docker run --rm -it \
  -v "$PWD":/workspace/tut \
  -v "$PWD/../tut-aux":/workspace/tut-aux \
  spus-ft-brev \
  run-vertical-slice.sh
```

That executes `notebooks/vertical_slice.ipynb` with `jupyter nbconvert` and
writes the executed notebook to:

```bash
/workspace/tut-aux/spus-ft-runs/vertical_slice-executed.ipynb
```

If you are using NVIDIA Brev with a GPU-backed instance, you can pass your
usual Docker GPU runtime flags when launching the container, for example
`--gpus all`, and keep the same mounted repo and aux directories. The Compose
file already defaults `DOCKER_GPUS` to `all`.

To download the FNS-KF dataset into the external data directory:

```bash
PYTHONPATH=src uv run python -m spus_ft download-fns-kf
```

This downloads the dataset repo contents needed for the tutorial into:

```bash
/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF
```

including:

- `assemble_data.py`
- `README.md`
- `solution_*.nc`

To download the pretrained SPUS model repo plus the three fine-tuned FNS-KF
model repos into the external models directory:

```bash
PYTHONPATH=src uv run python -m spus_ft download-spus-models
```

This downloads model snapshots into:

```bash
/usr/projects/artimis/cahrens/tut-aux/spus-models
```

including subdirectories for:

- `spus-pde-unet-36m-v2`
- `spus-pde-unet-36m-v2-fns-kf`
- `spus-pde-unet-36m-v2-fns-kf-32`
- `spus-pde-unet-36m-v2-fns-kf-256`

The tutorial expects those Hugging Face model snapshots to live in the external
models directory.

If you want the same four Hugging Face model repos materialized as full git
clones for reproducibility, use:

```bash
PYTHONPATH=src uv run python -m spus_ft clone-spus-models
```

This clones the same four repos under `spus-models`, which can be useful when
you want explicit git metadata and commit hashes in the mounted model storage.

To fine-tune on an assembled NetCDF file from a notebook or Python shell:

Note: `FNS-KF.nc` contains 2 channels per frame. For compatibility with the
pretrained SPUS 36M model, the tutorial pads each frame to 5 channels by
keeping the first 2 data channels unchanged and appending 3 channels filled
with `1.0`. This follows the handling described for downstream adaptation in
the paper *Generalizable PDE Learning with Pretrained Convolutional U-Net
Architecture* by Abu Bucker Siddik, Diane Oyen, Alexander Most, Michal Kucer,
Nathan A. DeBardeleben, Earl Lawrence, and Ayan Biswas.

Also note: the tutorial uses a single assembled NetCDF file, `FNS-KF.nc`, and
constructs train/val/test splits in memory by trajectory index. It does not
expect separate train/val/test NetCDF files on disk.

By default, the tutorial baseline pretrained model is now the Hugging Face
snapshot `siddik-lanl/spus-pde-unet-36m-v2`, stored externally under
`spus-models/spus-pde-unet-36m-v2`. The companion fine-tuned Hugging Face model
snapshots used for comparisons are:

- `spus-pde-unet-36m-v2-fns-kf`
- `spus-pde-unet-36m-v2-fns-kf-32`
- `spus-pde-unet-36m-v2-fns-kf-256`

```python
from pathlib import Path

from spus_ft.pipeline import fine_tune, infer_rollout, evaluate_rollout

dataset_path = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc")

finetune_outcome = fine_tune(
    dataset_path=dataset_path,
    pretrained_source="/usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2",
    num_train_trajectories=16,
    epochs=10,
)

rollout_outcome = infer_rollout(
    dataset_path=dataset_path,
    pretrained_source="/usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2",
    num_train_trajectories=16,
    checkpoint_path=finetune_outcome["checkpoint_path"],
)

evaluation_outcome = evaluate_rollout(
    rollout_outcome=rollout_outcome,
)
```

Inference now supports two distinct fine-tuned model paths for FNS-KF data:

- Tutorial checkpoint inference: loads the base `Unet2D` model and applies a
  saved `best_checkpoint.pt`. Because the base SPUS model expects 5 channels,
  the tutorial pads the 2-channel FNS-KF frames with three extra channels set
  to `1.0`.
- Hugging Face fine-tuned snapshot inference: loads the `SPUSFNSKF` class from
  `spus-pde-unet-36m-v2-fns-kf*` and uses the raw 2-channel FNS-KF frames with
  no padding.

To infer an autoregressive rollout for exactly one trajectory from the test
split with a tutorial-produced checkpoint:

```python
from pathlib import Path

from spus_ft import infer_rollout

dataset_path = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc")

rollout_outcome = infer_rollout(
    dataset_path=dataset_path,
    split_name="test",
    test_trajectory_index=0,
    pretrained_source="/usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2",
    checkpoint_path="/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/your_run/best_checkpoint.pt",
)
```

To infer the same trajectory using one of the downloaded Hugging Face
fine-tuned FNS-KF snapshots instead:

```python
from pathlib import Path

from spus_ft import infer_rollout

dataset_path = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc")

rollout_outcome = infer_rollout(
    dataset_path=dataset_path,
    split_name="test",
    test_trajectory_index=0,
    pretrained_source="/usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2-fns-kf-256",
)
```

These calls select the first trajectory in the test split, predict the full
rollout autoregressively, and write the outputs plus MSE summary into
`spus-ft-runs`.

For pretrained-only baseline inference, point `pretrained_source` at the base
snapshot `spus-pde-unet-36m-v2`. That path uses `Unet2D` and the tutorial’s
5-channel padding behavior for FNS-KF inputs.

The same workflow is also available from the CLI:

```bash
PYTHONPATH=src uv run python -m spus_ft fine-tune \
  --dataset-path /usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc \
  --pretrained-source /usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2 \
  --num-train-trajectories 16 \
  --epochs 10

PYTHONPATH=src uv run python -m spus_ft infer-rollout \
  --dataset-path /usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc \
  --pretrained-source /usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2 \
  --checkpoint-path /usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/your_run/best_checkpoint.pt \
  --split-name test \
  --split-trajectory-index 0

PYTHONPATH=src uv run python -m spus_ft infer-rollout \
  --dataset-path /usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc \
  --pretrained-source /usr/projects/artimis/cahrens/tut-aux/spus-models/spus-pde-unet-36m-v2-fns-kf-256 \
  --split-name test \
  --split-trajectory-index 0

PYTHONPATH=src uv run python -m spus_ft evaluate-rollout \
  --rollout-run-dir /usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/your_rollout_run
```

To make a Fig. 3-style comparison plot for predicted and true frames at rollout
steps 1, 8, and 16:

```python
from pathlib import Path

from spus_ft import plot_rollout_frames

run_dir = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/your_infer_run")

plot_rollout_frames(
    run_dir=run_dir,
    trajectory_index=0,
    timesteps=(1, 8, 16),
    channel_index=0,
    save_path=run_dir / "fig3_like_rollout.png",
)
```

To evaluate rollout quality over multiple trajectories from a split:

```python
from pathlib import Path

from spus_ft import evaluate_rollout

dataset_path = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc")

evaluation_outcome = evaluate_rollout(
    rollout_run_dir="/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/your_infer_run",
)
```

By default, this evaluates **all trajectories in the selected split** and writes
`evaluation.json` containing overall average MSE, per-timestep MSE, and
per-trajectory MSE. It also saves preview
tensors for the first few evaluated trajectories as:

- `preview_predictions.pt`
- `preview_targets.pt`

By default the first 5 evaluated trajectories are saved in those preview files.

To make a Fig. 4-style comparison of average MSE for each rollout step across
multiple fine-tuned models trained on different amounts of FNS-KF data:

```python
from pathlib import Path

from spus_ft import plot_multiple_rollout_mse_curves

evaluation_paths = [
    Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/eval_ft_1/evaluation.json"),
    Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/eval_ft_8/evaluation.json"),
    Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/eval_ft_32/evaluation.json"),
]

plot_multiple_rollout_mse_curves(
    evaluation_paths,
    labels=[
        "pretrained v2 baseline",
        "HF fine-tuned v2",
        "HF fine-tuned v2 (32)",
        "HF fine-tuned v2 (256)",
    ],
    save_path=Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs/fns_kf_mse_compare.png"),
)
```

If you do not pass explicit labels, the plotting helper derives them from the
saved evaluation metadata, including whether the run is pretrained or
fine-tuned and the recorded number of training trajectories.

Fine-tuned checkpoints, rollout tensors, summaries, and metadata are written under:

- `/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs`

## Next phases

The next build stages will add:

- FNS-KF dataset download and loading
- SPUS pretrained model/config loading
- SLURM script generation
- Fine-tuning for 10 epochs
- Saved-model inference and pretrained vs. finetuned MSE comparison

## Git note

This workspace currently contains scaffold files only. If you want this directory
connected to `https://lisdi-git.lanl.gov/artimis/fmod`, the next step is to
initialize or attach a writable git repository and add that remote.
