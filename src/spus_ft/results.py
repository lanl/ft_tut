from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from spus_ft.metrics import compute_average_mse, compute_per_trajectory_mse, compute_timestep_mse


def _to_serializable_array(value: Any) -> Any:
    converted = value
    if hasattr(converted, 'detach'):
        converted = converted.detach()
    if hasattr(converted, 'cpu'):
        converted = converted.cpu()
    if hasattr(converted, 'tolist'):
        return converted.tolist()
    return converted


def _flatten_numeric_values(value: Any) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numeric_values(item))
        return flattened
    return [float(value)]


def summarize_rollout_prediction_range(predictions: Any) -> dict:
    values = _flatten_numeric_values(_to_serializable_array(predictions))
    return {
        'min': min(values),
        'max': max(values),
    }


def save_rollout(run_dir: Path, predictions: list, targets: list, metadata: dict) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "rollout.json").open("w", encoding="utf-8") as handle:
        json.dump({"predictions": predictions, "targets": targets}, handle, indent=2)
    summary = {
        "average_mse": compute_average_mse(predictions, targets),
        "timestep_mse": compute_timestep_mse(predictions, targets),
        "metadata": metadata,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_rollout(run_dir: Path) -> dict:
    with (run_dir / "rollout.json").open("r", encoding="utf-8") as handle:
        arrays = json.load(handle)
    with (run_dir / "summary.json").open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {
        "predictions": arrays["predictions"],
        "targets": arrays["targets"],
        "summary": summary,
    }


def save_torch_rollout(
    run_dir: Path,
    predictions: Any,
    targets: Any,
    metadata: dict,
    torch_module: Optional[Any] = None,
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch_lib = torch_module
    if torch_lib is None:
        import torch as imported_torch

        torch_lib = imported_torch

    predictions_cpu = predictions.detach().cpu()
    targets_cpu = targets.detach().cpu()
    torch_lib.save(
        {
            'predictions': predictions_cpu,
            'targets': targets_cpu,
            'metadata': metadata,
        },
        run_dir / 'rollout.pt',
    )

    predictions_list = predictions_cpu.tolist()
    targets_list = targets_cpu.tolist()
    summary = {
        'average_mse': compute_average_mse(predictions_list, targets_list),
        'timestep_mse': compute_timestep_mse(predictions_list, targets_list),
        'metadata': metadata,
    }
    with (run_dir / 'summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_torch_rollout(run_dir: Path, torch_module: Optional[Any] = None) -> dict:
    torch_lib = torch_module
    if torch_lib is None:
        import torch as imported_torch

        torch_lib = imported_torch

    payload = torch_lib.load(run_dir / 'rollout.pt', map_location='cpu')
    with (run_dir / 'summary.json').open('r', encoding='utf-8') as handle:
        summary = json.load(handle)
    return {
        'predictions': payload['predictions'],
        'targets': payload['targets'],
        'metadata': payload['metadata'],
        'summary': summary,
    }


def load_evaluation(run_dir: Path) -> dict:
    with (run_dir / 'evaluation.json').open('r', encoding='utf-8') as handle:
        return json.load(handle)


def save_evaluation(run_dir: Path, predictions: Any, targets: Any, metadata: dict) -> dict:
    predictions_list = predictions.detach().cpu().tolist()
    targets_list = targets.detach().cpu().tolist()
    evaluation = {
        'average_mse': compute_average_mse(predictions_list, targets_list),
        'timestep_mse': compute_timestep_mse(predictions_list, targets_list),
        'per_trajectory_mse': compute_per_trajectory_mse(predictions_list, targets_list),
        'metadata': metadata,
    }
    with (run_dir / 'evaluation.json').open('w', encoding='utf-8') as handle:
        json.dump(evaluation, handle, indent=2)
    return evaluation


def save_preview_tensors(
    run_dir: Path,
    predictions: Any,
    targets: Any,
    preview_count: int = 5,
    torch_module: Optional[Any] = None,
) -> dict:
    torch_lib = torch_module
    if torch_lib is None:
        import torch as imported_torch

        torch_lib = imported_torch

    run_dir.mkdir(parents=True, exist_ok=True)
    limited_predictions = predictions[:preview_count].detach().cpu()
    limited_targets = targets[:preview_count].detach().cpu()

    predictions_path = run_dir / 'preview_predictions.pt'
    targets_path = run_dir / 'preview_targets.pt'
    torch_lib.save(limited_predictions, predictions_path)
    torch_lib.save(limited_targets, targets_path)
    return {
        'preview_predictions_path': predictions_path,
        'preview_targets_path': targets_path,
        'preview_count': int(min(preview_count, len(predictions))),
    }


def save_checkpoint(run_dir: Path, checkpoint: dict, metadata: dict, filename: str = 'checkpoint.pt', torch_module: Optional[Any] = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch_lib = torch_module
    if torch_lib is None:
        import torch as imported_torch

        torch_lib = imported_torch

    checkpoint_path = run_dir / filename
    torch_lib.save(checkpoint, checkpoint_path)
    with (run_dir / 'metadata.json').open('w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)
    return checkpoint_path
