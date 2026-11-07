# FMOD Tutorial

**Foundation Model Fine-Tuning Tutorial&#x20;**&#x68;as been acknowledged by NNSA for open-source release. O# **(O5229)**

This repository contains a scientist-facing tutorial on running pretrained
foundation models, fine-tuning them, and comparing their results on PDE-style
time series data.

The repository includes the SPUS/FNS-KF tutorial path, including
pretrained-model inference, fine-tuning, rollout evaluation, plotting helpers,
and Brev VM startup scripts.

## Current scope

- Python package under `src/spus_ft/`
- External path management for data, cache, and run outputs
- SPUS v2 pretrained and fine-tuned FNS-KF model support
- Autoregressive inference, evaluation, and plotting helpers
- FNS-KF download and `FNS-KF.nc` assembly helpers
- Jupyter notebooks for the vertical-slice tutorial flow
- Brev VM setup script under `scripts/`
- Dockerfiles and Compose files for Brev single-container experiments
- Pytest/unit-test coverage for the tutorial helpers

## Repository layout

- `src/spus_ft/` — tutorial package code
- `tests/` — unit tests for the tutorial helpers
- `notebooks/` — scientist-facing notebook entrypoint
- `configs/` — small JSON configuration files
- `docker/` — container support files
- `scripts/` — Brev VM startup scripts

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

## Run on Brev VM

The currently tested Brev path is **VM Mode** with the startup script in
`scripts/brev_vm_setup.sh` and **Jupyter on the host**.

Published Brev Launchable Link:

- `https://jupyter-qlbkbtaq1.gobrev.dev`​

How to launch it:

- open the Launchable link in your browser
- sign in to NVIDIA Brev if prompted
- select your Brev organization or workspace if you have more than one
- review the Launchable configuration
- choose an available GPU instance and review its estimated hourly cost
- click the launch / deploy button to provision your VM
- wait for provisioning and for the setup script to complete; this can take around 15-20 minutes
- once the instance is ready, use the Brev notebook entrypoint and SSH steps
  below to connect

This repository has been tested on a Brev instance using that VM startup
script. The script:

- clones the tutorial repo onto the VM
- creates the locked `uv` environment
- registers the `Python (spus-ft)` Jupyter kernel
- downloads SPUS model snapshots
- downloads the raw FNS-KF files and assembles `FNS-KF.nc`

The primary tutorial workflow today assumes you launch a Brev VM from a Mac and
use the Brev CLI locally.

On your Mac:

```bash
curl -fsSL https://raw.githubusercontent.com/brevdev/brev-cli/main/bin/install-latest.sh | sed 's/release_tag="$(get_latest_release_tag)"/release_tag="v0.6.330"/' | bash

export PATH="$HOME/.local/bin:$PATH"

# Log in to Brev; this opens a browser window.
brev login

# Start the Launchable from the Brev Console and note the instance name.
# Wait for the VM to build and for the tutorial startup script to finish.
# The startup script checks out the tutorial code, downloads the data and
# models, and assembles the FNS-KF NetCDF dataset.

# Show running Brev instances.
brev ls

# Refresh local SSH config for Brev-managed instances.
brev refresh

# Confirm your local port 8888 is free before tunneling.
lsof -nP -iTCP:8888 -sTCP:LISTEN

# Forward local port 8888 to the Brev VM.
ssh -F ~/.brev/ssh_config -N -L 8888:localhost:8888 <launchable>
```

From the Brev web page:

- open `Compute -> Launchable -> Start notebook`
- wait if the notebook server is still coming up

If you want a shell on the Launchable in a second terminal on your Mac:

```bash
ssh -F ~/.brev/ssh_config <launchable>
```

Inside the jupyter notebook UI, open:

- `notebooks/vertical_slice.ipynb`

Select the `Python (spus-ft)` kernel.

## Run on your own machine (not recommended for this tutorial)

Create a virtual environment, install dev dependencies, and run tests:

```bash
export PATH="$HOME/.local/bin:$PATH"
uv sync --extra dev
PYTHONPATH=src uv run python -m unittest discover -s tests -v
```

For the real SPUS model path, install the model runtime dependencies first:

```bash
uv sync --extra spus
```

To add Jupyter notebook capability to the same `uv` environment:

```bash
uv sync --extra notebook
```

For the full tutorial environment on your workstation, including the SPUS
runtime stack and notebook support:

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

The notebook-first local workflow is:

- install the locked `uv` environment with the `spus` and `notebook` extras
- download the FNS-KF dataset and SPUS model snapshots into `tut-aux`
- assemble `FNS-KF.nc` if it is not already present
- launch JupyterLab and open `notebooks/vertical_slice.ipynb`

Useful helper commands:

```bash
PYTHONPATH=src uv run python -m spus_ft download-fns-kf
PYTHONPATH=src uv run python -m spus_ft download-spus-models
PYTHONPATH=src uv run jupyter lab
```

The dataset is stored under:

```bash
/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF
```

The model snapshots are stored under:

```bash
/usr/projects/artimis/cahrens/tut-aux/spus-models
```

The tutorial uses a single assembled NetCDF file, `FNS-KF.nc`, and constructs
train/val/test splits in memory by trajectory index. It does not expect
separate train/val/test NetCDF files on disk.

For inference on FNS-KF data:

- pretrained v2 baseline inference uses `spus-pde-unet-36m-v2` with 5-channel
  padded input
- Hugging Face fine-tuned FNS-KF snapshots use `SPUSFNSKF` with raw 2-channel
  input

Outputs such as checkpoints, rollout tensors, plots, summaries, and evaluation
artifacts are written under:

```bash
/usr/projects/artimis/cahrens/tut-aux/spus-ft-runs
```

If you want the same four Hugging Face model repos materialized as full git
clones for reproducibility, use:

```bash
PYTHONPATH=src uv run python -m spus_ft clone-spus-models
```

For the full end-to-end tutorial workflow, use `notebooks/vertical_slice.ipynb`.

## Run on Brev Single Container (Under Development)

The repository includes Dockerfiles, Compose configuration, and Brev
single-container experiments under `docker/`, but this path is still
experimental on Brev. The single-container notebook connection has not yet been
reliable end-to-end, so it is not the recommended tutorial workflow today.

For Docker build/run details, Compose usage, and the current Brev
single-container notes, see `docker/README.md`.

## Status notes

Current state of the tutorial repo:

- Brev VM Mode with `scripts/brev_vm_setup.sh` is the main tested deployment
- Brev single-container Docker support remains experimental
- Brev Single Container remains experimental and is not yet the recommended path

