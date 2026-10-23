from __future__ import annotations

from dataclasses import dataclass
from typing import List

Array2D = List[List[float]]


@dataclass(frozen=True)
class BaselineStepper:
    alpha: float

    def predict_next(self, state: Array2D, timestep: int) -> Array2D:
        return [[self.alpha * value for value in sample] for sample in state]


def make_baseline_model(alpha: float) -> BaselineStepper:
    return BaselineStepper(alpha=alpha)
