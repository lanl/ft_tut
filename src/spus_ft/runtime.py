from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, Iterable


def check_spus_runtime_dependencies(
    required: Iterable[str] = ("torch", "huggingface_hub", "safetensors")
) -> Dict[str, object]:
    installed: Dict[str, str] = {}
    missing = []

    for package_name in required:
        try:
            import_module(package_name)
            installed[package_name] = version(package_name)
        except (ImportError, PackageNotFoundError):
            missing.append(package_name)

    return {
        "ok": not missing,
        "installed": installed,
        "missing": missing,
    }


def require_spus_runtime_dependencies(
    required: Iterable[str] = ("torch", "huggingface_hub", "safetensors")
) -> Dict[str, object]:
    result = check_spus_runtime_dependencies(required)
    if result["missing"]:
        missing = ", ".join(result["missing"])
        raise RuntimeError(
            "Missing SPUS runtime dependencies: "
            f"{missing}. Install them with `uv sync --extra spus` before running the model demo."
        )
    return result
