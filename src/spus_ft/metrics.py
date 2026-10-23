from __future__ import annotations

from typing import List

Array3D = List[List[List[float]]]


def _sum_squared_error(prediction, target) -> tuple[float, int]:
    if isinstance(prediction, list):
        total = 0.0
        count = 0
        for pred_item, target_item in zip(prediction, target):
            item_total, item_count = _sum_squared_error(pred_item, target_item)
            total += item_total
            count += item_count
        return total, count

    delta = prediction - target
    return delta * delta, 1


def compute_timestep_mse(predictions: Array3D, targets: Array3D) -> List[float]:
    num_samples = len(predictions)
    num_timesteps = len(predictions[0]) if predictions else 0
    per_timestep = []

    for timestep in range(num_timesteps):
        squared_sum = 0.0
        value_count = 0
        for sample in range(num_samples):
            item_sum, item_count = _sum_squared_error(
                predictions[sample][timestep],
                targets[sample][timestep],
            )
            squared_sum += item_sum
            value_count += item_count
        per_timestep.append(squared_sum / value_count)

    return per_timestep


def compute_average_mse(predictions: Array3D, targets: Array3D) -> float:
    per_timestep = compute_timestep_mse(predictions, targets)
    return sum(per_timestep) / len(per_timestep)


def compute_per_trajectory_mse(predictions: Array3D, targets: Array3D) -> List[float]:
    per_trajectory = []
    for prediction, target in zip(predictions, targets):
        item_sum, item_count = _sum_squared_error(prediction, target)
        per_trajectory.append(item_sum / item_count)
    return per_trajectory
