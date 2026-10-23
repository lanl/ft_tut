from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Dict, List, Optional, Sequence

from spus_ft.paths import TutorialPaths

Array2D = List[List[float]]
Array3D = List[List[List[float]]]


@dataclass(frozen=True)
class SyntheticDataset:
    inputs: Array2D
    targets: Array3D
    metadata: Dict[str, float]


FNS_KF_REPO_ID = "camlab-ethz/FNS-KF"
FNS_KF_SUBDIR = "FNS-KF"
FNS_KF_PATTERNS = [
    "README.md",
    "assemble_data.py",
    ".gitattributes",
    "solution_*.nc",
]

SPUS_MODEL_REPO_IDS = [
    "siddik-lanl/spus-pde-unet-36m-v2",
    "siddik-lanl/spus-pde-unet-36m-v2-fns-kf",
    "siddik-lanl/spus-pde-unet-36m-v2-fns-kf-32",
    "siddik-lanl/spus-pde-unet-36m-v2-fns-kf-256",
]
SPUS_MODEL_GIT_BASE_URL = "https://huggingface.co"
SPUS_MODEL_PATTERNS = [
    "*.json",
    "*.md",
    "*.py",
    "*.safetensors",
    "*.txt",
]


def generate_synthetic_dataset(
    num_samples: int,
    num_timesteps: int,
    num_features: int,
    decay: float,
    forcing: float,
    seed: int,
) -> SyntheticDataset:
    rng = Random(seed)
    initial_state = [
        [rng.gauss(0.0, 1.0) for _ in range(num_features)]
        for _ in range(num_samples)
    ]
    targets: Array3D = []
    previous = [row[:] for row in initial_state]

    for timestep in range(num_timesteps):
        next_state = [
            [decay * value + forcing * (timestep + 1) for value in sample]
            for sample in previous
        ]
        targets.append(next_state)
        previous = [row[:] for row in next_state]

    targets_by_sample: Array3D = [
        [[targets[timestep][sample][feature] for feature in range(num_features)] for timestep in range(num_timesteps)]
        for sample in range(num_samples)
    ]

    return SyntheticDataset(
        inputs=initial_state,
        targets=targets_by_sample,
        metadata={
            "num_samples": float(num_samples),
            "num_timesteps": float(num_timesteps),
            "num_features": float(num_features),
            "decay": decay,
            "forcing": forcing,
            "seed": float(seed),
        },
    )


def resolve_fns_kf_data_dir(paths: Optional[TutorialPaths] = None) -> Path:
    tutorial_paths = paths or TutorialPaths.discover()
    return tutorial_paths.data_root / FNS_KF_SUBDIR


def resolve_spus_models_dir(paths: Optional[TutorialPaths] = None) -> Path:
    tutorial_paths = paths or TutorialPaths.discover()
    return tutorial_paths.models_root


def resolve_spus_model_dir(repo_id: str, paths: Optional[TutorialPaths] = None) -> Path:
    tutorial_paths = paths or TutorialPaths.discover()
    return tutorial_paths.models_root / repo_id.split("/", 1)[1]


def resolve_spus_model_git_url(repo_id: str) -> str:
    return f"{SPUS_MODEL_GIT_BASE_URL}/{repo_id}"


def download_fns_kf_dataset(
    paths: Optional[TutorialPaths] = None,
    local_dir: Optional[Path] = None,
    repo_id: str = FNS_KF_REPO_ID,
    allow_patterns: Optional[List[str]] = None,
    snapshot_download_fn: Optional[Callable[..., Any]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    target_dir = (local_dir or resolve_fns_kf_data_dir(tutorial_paths)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    download_fn = snapshot_download_fn
    if download_fn is None:
        from huggingface_hub import snapshot_download as imported_snapshot_download

        download_fn = imported_snapshot_download

    used_patterns = allow_patterns or list(FNS_KF_PATTERNS)
    downloaded_path = download_fn(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=used_patterns,
    )

    return {
        "repo_id": repo_id,
        "target_dir": target_dir,
        "downloaded_path": Path(downloaded_path),
        "allow_patterns": used_patterns,
    }


def download_spus_model(
    repo_id: str,
    paths: Optional[TutorialPaths] = None,
    local_dir: Optional[Path] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    snapshot_download_fn: Optional[Callable[..., Any]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    target_dir = (local_dir or resolve_spus_model_dir(repo_id, tutorial_paths)).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    download_fn = snapshot_download_fn
    if download_fn is None:
        from huggingface_hub import snapshot_download as imported_snapshot_download

        download_fn = imported_snapshot_download

    used_patterns = list(allow_patterns or SPUS_MODEL_PATTERNS)
    downloaded_path = download_fn(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
        allow_patterns=used_patterns,
    )

    return {
        "repo_id": repo_id,
        "target_dir": target_dir,
        "downloaded_path": Path(downloaded_path),
        "allow_patterns": used_patterns,
    }


def download_spus_model_collection(
    paths: Optional[TutorialPaths] = None,
    repo_ids: Optional[Sequence[str]] = None,
    allow_patterns: Optional[Sequence[str]] = None,
    snapshot_download_fn: Optional[Callable[..., Any]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    selected_repo_ids = list(repo_ids or SPUS_MODEL_REPO_IDS)
    downloads = [
        download_spus_model(
            repo_id=repo_id,
            paths=tutorial_paths,
            allow_patterns=allow_patterns,
            snapshot_download_fn=snapshot_download_fn,
        )
        for repo_id in selected_repo_ids
    ]
    return {
        "models_root": tutorial_paths.models_root,
        "repo_ids": selected_repo_ids,
        "downloads": downloads,
    }


def clone_spus_model_repo(
    repo_id: str,
    paths: Optional[TutorialPaths] = None,
    local_dir: Optional[Path] = None,
    revision: Optional[str] = None,
    git_runner: Optional[Callable[..., Any]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    target_dir = (local_dir or resolve_spus_model_dir(repo_id, tutorial_paths)).resolve()
    parent_dir = target_dir.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    repo_url = resolve_spus_model_git_url(repo_id)
    runner = git_runner or subprocess.run

    cloned = False
    if not (target_dir / ".git").exists():
        runner(
            ["git", "clone", repo_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        cloned = True

    if revision is not None:
        runner(
            ["git", "-C", str(target_dir), "checkout", revision],
            check=True,
            capture_output=True,
            text=True,
        )

    return {
        "repo_id": repo_id,
        "repo_url": repo_url,
        "target_dir": target_dir,
        "revision": revision,
        "cloned": cloned,
    }


def clone_spus_model_collection(
    paths: Optional[TutorialPaths] = None,
    repo_ids: Optional[Sequence[str]] = None,
    revisions: Optional[Dict[str, str]] = None,
    local_root: Optional[Path] = None,
    git_runner: Optional[Callable[..., Any]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    selected_repo_ids = list(repo_ids or SPUS_MODEL_REPO_IDS)
    revision_map = revisions or {}
    clones = [
        clone_spus_model_repo(
            repo_id=repo_id,
            paths=tutorial_paths,
            local_dir=(local_root / repo_id.split("/", 1)[1]).resolve() if local_root is not None else None,
            revision=revision_map.get(repo_id),
            git_runner=git_runner,
        )
        for repo_id in selected_repo_ids
    ]
    return {
        "models_root": tutorial_paths.models_root,
        "repo_ids": selected_repo_ids,
        "revisions": revision_map,
        "clones": clones,
    }
