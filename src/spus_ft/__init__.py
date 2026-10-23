"""SPUS fine-tuning tutorial scaffold package."""

from spus_ft.metrics import compute_average_mse
from spus_ft.paths import TutorialPaths
from spus_ft.pipeline import evaluate_rollout, fine_tune, infer_rollout
from spus_ft.plotting import build_evaluation_label, plot_full_rollout_frames, plot_ground_truth_trajectory_frames, plot_multiple_rollout_mse_curves, plot_rollout_frames, plot_rollout_frames_separate_scales, plot_rollout_mse_curve

__all__ = [
    "TutorialPaths",
    "build_evaluation_label",
    "compute_average_mse",
    "evaluate_rollout",
    "fine_tune",
    "infer_rollout",
    "plot_full_rollout_frames",
    "plot_ground_truth_trajectory_frames",
    "plot_multiple_rollout_mse_curves",
    "plot_rollout_frames",
    "plot_rollout_frames_separate_scales",
    "plot_rollout_mse_curve",
]
