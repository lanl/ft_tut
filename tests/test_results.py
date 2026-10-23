from pathlib import Path
import unittest

from spus_ft.results import load_rollout, save_rollout, summarize_rollout_prediction_range


class TestResults(unittest.TestCase):
    def test_save_and_load_rollout_round_trip(self) -> None:
        tmp_path = Path("/tmp/src-test-results")
        if tmp_path.exists():
            for child in sorted(tmp_path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            tmp_path.rmdir()
        tmp_path.mkdir(parents=True)
        run_dir = tmp_path / "demo"
        predictions = [[[1.0], [2.0]]]
        targets = [[[1.5], [1.5]]]

        summary = save_rollout(run_dir, predictions, targets, {"stage": 1})
        restored = load_rollout(run_dir)

        self.assertEqual(summary["average_mse"], 0.25)
        self.assertEqual(restored["summary"]["metadata"]["stage"], 1)
        self.assertEqual(restored["predictions"], predictions)
        self.assertEqual(restored["targets"], targets)

    def test_save_evaluation_preview_tensors_is_documented_by_api(self) -> None:
        from spus_ft.results import save_preview_tensors

        class FakeTensor:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, item):
                return FakeTensor(self.value[item])

            def detach(self):
                return self

            def cpu(self):
                return self

            def __len__(self):
                return len(self.value)

        class FakeTorch:
            def save(self, obj, path):
                Path(path).write_text(repr(obj.value), encoding="utf-8")

        tmp_path = Path("/tmp/fmod-test-preview")
        if tmp_path.exists():
            for child in sorted(tmp_path.rglob("*"), reverse=True):
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            tmp_path.rmdir()
        tmp_path.mkdir(parents=True)

        result = save_preview_tensors(
            tmp_path,
            FakeTensor([1, 2, 3, 4, 5, 6]),
            FakeTensor([6, 5, 4, 3, 2, 1]),
            preview_count=2,
            torch_module=FakeTorch(),
        )

        self.assertEqual(result["preview_count"], 2)
        self.assertTrue(result["preview_predictions_path"].is_file())
        self.assertTrue(result["preview_targets_path"].is_file())

    def test_summarize_rollout_prediction_range_reports_min_and_max(self) -> None:
        predictions = [
            [
                [[[1.0, -2.0], [3.5, 0.0]]],
                [[[4.0, 5.5], [-1.0, 2.0]]],
            ]
        ]

        summary = summarize_rollout_prediction_range(predictions)

        self.assertEqual(summary['min'], -2.0)
        self.assertEqual(summary['max'], 5.5)
