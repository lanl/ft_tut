# GPU Tutorial Setup Guide (NVIDIA Brev + Docker)

This guide shows a tutorial participant how to launch a GPU instance on NVIDIA
Brev, pull this repository, run the prebuilt container image from Docker Hub,
materialize the required dataset and model snapshots, and open the tutorial
notebooks in JupyterLab.

It assumes the tutorial container image has already been built and pushed to
Docker Hub.

## Prerequisites

Before starting, make sure you have:

- an NVIDIA Brev account with access to a GPU instance type
- Docker available on the Brev instance
- access to the tutorial repository
- the Docker Hub image name for this tutorial

Throughout this guide, replace:

- `<instance-name>` with your Brev instance name
- `<repo-url>` with the git URL for this repository
- `<repo-dir>` with the local checkout directory name
- `<dockerhub-image>` with the published tutorial image, for example `your-org/spus-ft-brev:latest`

## 1. Create and enter the Brev instance

From your local laptop terminal:

```bash
brev login
brev create <instance-name> --gpu "nebius.l40sx1.pcie"
brev shell <instance-name>
```

Once `brev shell` connects, the rest of the commands run on the remote Brev
instance.

## 2. Clone the tutorial repo onto the Brev host

Use a persistent workspace directory on the Brev VM:

```bash
mkdir -p /home/ubuntu/workspace
cd /home/ubuntu/workspace

git clone <repo-url>
cd <repo-dir>
```

## 3. Create persistent host directories for tutorial artifacts

This tutorial keeps code inside the container image or mounted repo checkout,
while large artifacts live in a separate persistent directory on the host.

```bash
mkdir -p /home/ubuntu/workspace/tut-aux/spus-ft-data
mkdir -p /home/ubuntu/workspace/tut-aux/spus-models
mkdir -p /home/ubuntu/workspace/tut-aux/spus-ft-runs
mkdir -p /home/ubuntu/workspace/tut-aux/spus-ft-cache
```

These directories are mounted into the container as:

- `/workspace/tut-aux/spus-ft-data`
- `/workspace/tut-aux/spus-models`
- `/workspace/tut-aux/spus-ft-runs`
- `/workspace/tut-aux/spus-ft-cache`

## 4. Create a Compose file on the Brev host

Use Docker Compose as the default workflow so participants do not need to keep
retyping a long `docker run` command.

Create `docker-compose.yml` in `/home/ubuntu/workspace/<repo-dir>`:

```yaml
services:
  jupyter:
    image: <dockerhub-image>
    ports:
      - "8888:8888"
    environment:
      BOOTSTRAP_SPUS_ASSETS: 1
      BOOTSTRAP_SPUS_MODELS: 1
      BOOTSTRAP_FNS_KF_DATA: 1
      JUPYTER_PORT: 8888
      JUPYTER_TOKEN: brev
      JUPYTER_ROOT: /workspace/tut
      SPUS_FT_AUX_ROOT: /workspace/tut-aux
      FMOD_DATA_ROOT: /workspace/tut-aux/spus-ft-data
      FMOD_MODELS_ROOT: /workspace/tut-aux/spus-models
      FMOD_RUNS_ROOT: /workspace/tut-aux/spus-ft-runs
      FMOD_CACHE_ROOT: /workspace/tut-aux/spus-ft-cache
    volumes:
      - .:/workspace/tut
      - /home/ubuntu/workspace/tut-aux:/workspace/tut-aux
    command: ["start-jupyter.sh"]
    gpus: all

  vertical-slice:
    image: <dockerhub-image>
    profiles: ["batch"]
    environment:
      BOOTSTRAP_SPUS_ASSETS: 1
      BOOTSTRAP_SPUS_MODELS: 1
      BOOTSTRAP_FNS_KF_DATA: 1
      SPUS_FT_AUX_ROOT: /workspace/tut-aux
      FMOD_DATA_ROOT: /workspace/tut-aux/spus-ft-data
      FMOD_MODELS_ROOT: /workspace/tut-aux/spus-models
      FMOD_RUNS_ROOT: /workspace/tut-aux/spus-ft-runs
      FMOD_CACHE_ROOT: /workspace/tut-aux/spus-ft-cache
    volumes:
      - .:/workspace/tut
      - /home/ubuntu/workspace/tut-aux:/workspace/tut-aux
    command: ["run-vertical-slice.sh"]
    gpus: all
```

## 5. Start the published tutorial container with Compose

From `/home/ubuntu/workspace/<repo-dir>`, launch the interactive notebook
service:

```bash
docker compose up -d jupyter
```

As a fallback, you can still launch the same container directly with `docker run`.

Launch the container with GPU access, JupyterLab, and the correct repo and aux
mounts:

```bash
docker run --rm -it \
  --gpus all \
  -p 8888:8888 \
  -v /home/ubuntu/workspace/<repo-dir>:/workspace/tut \
  -v /home/ubuntu/workspace/tut-aux:/workspace/tut-aux \
  -e JUPYTER_TOKEN=brev \
  --name spus-ft-jupyter \
  <dockerhub-image>
```

The container startup path uses these runtime locations:

- `SPUS_FT_AUX_ROOT=/workspace/tut-aux`
- `FMOD_DATA_ROOT=/workspace/tut-aux/spus-ft-data`
- `FMOD_MODELS_ROOT=/workspace/tut-aux/spus-models`
- `FMOD_RUNS_ROOT=/workspace/tut-aux/spus-ft-runs`
- `FMOD_CACHE_ROOT=/workspace/tut-aux/spus-ft-cache`

It also runs `/workspace/tut/docker/bootstrap-assets.sh` by default, which:

- downloads raw FNS-KF dataset contents if they are missing
- downloads SPUS model snapshots if they are missing
- reuses anything already present on the mounted `/workspace/tut-aux` volume

If you want to disable that automatic bootstrap behavior, set:

```bash
BOOTSTRAP_SPUS_ASSETS=0
```

## 6. Download the dataset and model snapshots inside the container

Open a second shell on the Brev host and enter the running container:

```bash
docker compose exec jupyter bash
cd /workspace/tut
```

Fallback with `docker run`:

```bash
docker exec -it spus-ft-jupyter bash
cd /workspace/tut
```

Download the FNS-KF dataset into the mounted data volume:

```bash
python -m spus_ft download-fns-kf
```

Download the pretrained SPUS model snapshot and the fine-tuned FNS-KF model
snapshots into the mounted models volume:

```bash
python -m spus_ft download-spus-models
```

If you prefer full git clones of the Hugging Face repos instead of snapshots,
use:

```bash
python -m spus_ft clone-spus-models
```

## 7. Port-forward Jupyter back to your laptop

From a new terminal on your local laptop:

```bash
brev port-forward <instance-name> --port 8888:8888
```

Then open this URL in your local browser:

```text
http://localhost:8888
```

Use the token you set in `JUPYTER_TOKEN` above, for example:

```text
brev
```

## 8. Open the tutorial notebook

Inside JupyterLab, open:

- `notebooks/vertical_slice.ipynb`
- `notebooks/01_phase0_phase1_scaffold.ipynb`

The notebook code will find its data and model artifacts under the mounted
`/workspace/tut-aux` tree.

## 9. Optional: run the vertical-slice notebook non-interactively

If you want to execute the vertical-slice notebook from the command line rather
than interactively in JupyterLab, use Compose:

```bash
docker compose run --rm vertical-slice
```

Fallback with `docker run`:

```bash
docker run --rm -it \
  --gpus all \
  -v /home/ubuntu/workspace/<repo-dir>:/workspace/tut \
  -v /home/ubuntu/workspace/tut-aux:/workspace/tut-aux \
  <dockerhub-image> \
  run-vertical-slice.sh
```

This writes the executed notebook to:

```text
/workspace/tut-aux/spus-ft-runs/vertical_slice-executed.ipynb
```

## Notes

- Keep data and models on the mounted host directory, not baked into the image.
- Reusing `/home/ubuntu/workspace/tut-aux` lets you stop and restart containers
  without re-downloading the dataset and models.
- If you update the repo checkout on the Brev VM, the mounted `/workspace/tut`
  path inside the container sees those changes immediately.
