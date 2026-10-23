from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _default_aux_root(repo_root: Path) -> Path:
    configured = os.environ.get("SPUS_FT_AUX_ROOT") or os.environ.get("FMOD_AUX_ROOT")
    if configured:
        return Path(configured).expanduser()

    repo_parts = repo_root.resolve().parts
    if len(repo_parts) >= 6 and repo_parts[:4] == ("/", "vast", "projects", "artimis"):
        user_root = Path("/usr/projects/artimis") / repo_parts[4]
        candidate = user_root / "tut-aux"
        if candidate.exists() or os.access(user_root, os.W_OK):
            return candidate

    if len(repo_parts) >= 6 and repo_parts[:4] == ("/", "usr", "projects", "artimis"):
        user_root = Path("/usr/projects/artimis") / repo_parts[4]
        candidate = user_root / "tut-aux"
        if candidate.exists() or os.access(user_root, os.W_OK):
            return candidate

    fallback = repo_root.parent / "tut-aux"
    if os.access(repo_root.parent, os.W_OK):
        return fallback

    return Path(tempfile.gettempdir()) / "tut-aux"


def _default_root(aux_root: Path, env_var: str, dirname: str) -> Path:
    configured = os.environ.get(env_var)
    if configured:
        return Path(configured).expanduser()
    return aux_root / dirname


@dataclass(frozen=True)
class TutorialPaths:
    repo_root: Path
    aux_root: Path
    data_root: Path
    models_root: Path
    runs_root: Path
    cache_root: Path

    @classmethod
    def discover(cls, start: Optional[Path] = None) -> "TutorialPaths":
        if start is not None:
            repo_root = start.resolve()
        else:
            current = Path(__file__).resolve()
            repo_root = current.parent
            for candidate in [current.parent, *current.parents]:
                if (candidate / "pyproject.toml").exists():
                    repo_root = candidate
                    break
        aux_root = _default_aux_root(repo_root)
        return cls(
            repo_root=repo_root,
            aux_root=aux_root,
            data_root=_default_root(aux_root, "FMOD_DATA_ROOT", "spus-ft-data"),
            models_root=_default_root(aux_root, "FMOD_MODELS_ROOT", "spus-models"),
            runs_root=_default_root(aux_root, "FMOD_RUNS_ROOT", "spus-ft-runs"),
            cache_root=_default_root(aux_root, "FMOD_CACHE_ROOT", "spus-ft-cache"),
        )

    def ensure_runtime_dirs(self) -> None:
        self.aux_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def stage_run_dir(self, stage_name: str) -> Path:
        run_dir = self.runs_root / stage_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
