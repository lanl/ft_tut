ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:${PATH} \
    PYTHONPATH=/workspace/tut/src \
    SPUS_FT_AUX_ROOT=/workspace/tut-aux \
    FMOD_DATA_ROOT=/workspace/tut-aux/spus-ft-data \
    FMOD_MODELS_ROOT=/workspace/tut-aux/spus-models \
    FMOD_RUNS_ROOT=/workspace/tut-aux/spus-ft-runs \
    FMOD_CACHE_ROOT=/workspace/tut-aux/spus-ft-cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip && python -m pip install uv

WORKDIR /workspace/tut

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY configs ./configs
COPY notebooks ./notebooks
COPY tests ./tests
COPY docker/start-jupyter.sh /usr/local/bin/start-jupyter.sh
COPY docker/run-vertical-slice.sh /usr/local/bin/run-vertical-slice.sh

RUN chmod +x /usr/local/bin/start-jupyter.sh /usr/local/bin/run-vertical-slice.sh
RUN uv sync --frozen --extra spus --extra notebook --extra dev
RUN python -m ipykernel install --prefix /usr/local --name spus-ft --display-name "Python (spus-ft)"
RUN mkdir -p /workspace/tut-aux/spus-ft-data /workspace/tut-aux/spus-models /workspace/tut-aux/spus-ft-runs /workspace/tut-aux/spus-ft-cache

EXPOSE 8888

ENTRYPOINT ["tini", "--"]
CMD ["start-jupyter.sh"]
