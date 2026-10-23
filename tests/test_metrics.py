import unittest

from spus_ft.metrics import compute_average_mse, compute_per_trajectory_mse, compute_timestep_mse


class TestMetrics(unittest.TestCase):
    def test_metrics_match_manual_mse(self) -> None:
        predictions = [[[1.0], [3.0]]]
        targets = [[[2.0], [1.0]]]

        timestep_mse = compute_timestep_mse(predictions, targets)
        average_mse = compute_average_mse(predictions, targets)

        self.assertEqual(timestep_mse, [1.0, 4.0])
        self.assertEqual(average_mse, 2.5)

    def test_per_trajectory_mse_handles_multiple_samples(self) -> None:
        predictions = [
            [[[1.0], [3.0]]],
            [[[2.0], [2.0]]],
        ]
        targets = [
            [[[2.0], [1.0]]],
            [[[1.0], [1.0]]],
        ]

        per_trajectory = compute_per_trajectory_mse(predictions, targets)

        self.assertEqual(per_trajectory, [2.5, 1.0])
