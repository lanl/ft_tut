from __future__ import annotations

from dataclasses import dataclass
from typing import List

Array2D = List[List[float]]
Array3D = List[List[List[float]]]


@dataclass(frozen=True)
class RolloutResult:
    predictions: Array3D
    targets: Array3D


def autoregressive_rollout(model, initial_state: Array2D, targets: Array3D) -> RolloutResult:
    num_samples = len(targets)
    num_timesteps = len(targets[0]) if targets else 0
    state = [row[:] for row in initial_state]
    predictions: Array3D = [[[] for _ in range(num_timesteps)] for _ in range(num_samples)]

    for timestep in range(num_timesteps):
        state = model.predict_next(state, timestep)
        for sample_idx in range(num_samples):
            predictions[sample_idx][timestep] = state[sample_idx][:]

    return RolloutResult(predictions=predictions, targets=targets)


def _ensure_batched_state(initial_condition, torch_module):
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(initial_condition, np.ndarray):
        state = torch_module.from_numpy(initial_condition)
    else:
        state = initial_condition

    if state.dim() == 3:
        state = state.unsqueeze(0)

    return state


def _default_torch():
    import torch

    return torch


def _forward_rollout_step(model, state, torch_module):
    if state.dim() == 4:
        expected_channels = getattr(model, 'in_channels', None)
        if expected_channels is not None and state.shape[1] == expected_channels:
            return model(state)
        return model(state.unsqueeze(1))
    return model(state)


def autoregressive_tensor_rollout(
    model,
    initial_condition,
    rollout_steps: int,
    device,
    torch_module=None,
):
    torch_lib = torch_module or _default_torch()

    @torch_lib.inference_mode()
    def _run():
        state = _ensure_batched_state(initial_condition, torch_lib).to(device)
        predictions = []

        for _ in range(rollout_steps):
            state = _forward_rollout_step(model, state, torch_lib)
            predictions.append(state.detach().cpu())

        return torch_lib.stack(predictions, dim=1)

    return _run()
