import unittest

from spus_ft.data import generate_synthetic_dataset


class TestData(unittest.TestCase):
    def test_synthetic_dataset_has_expected_shape(self) -> None:
        dataset = generate_synthetic_dataset(
            num_samples=3,
            num_timesteps=4,
            num_features=2,
            decay=0.9,
            forcing=0.1,
            seed=5,
        )

        self.assertEqual(len(dataset.inputs), 3)
        self.assertEqual(len(dataset.inputs[0]), 2)
        self.assertEqual(len(dataset.targets), 3)
        self.assertEqual(len(dataset.targets[0]), 4)
        self.assertEqual(len(dataset.targets[0][0]), 2)
