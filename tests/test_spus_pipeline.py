from pathlib import Path
from types import SimpleNamespace
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from spus_ft.paths import TutorialPaths
from spus_ft.pipeline import (
    append_run_log,
    evaluate_rollout,
    resolve_fns_kf_nc_path,
    resolve_spus_inference_strategy,
    resolve_spus_model_definition_repo,
    resolve_spus_pretrained_source,
    resolve_spus_36m_repo_path,
    resolve_spus_training_repo_path,
    select_split_trajectory_indices,
    select_test_trajectory_indices,
    run_spus_36m,
)


class FakeTensor:
    def __init__(self, shape, label="tensor"):
        self.shape = shape
        self.label = label


class FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeTorch:
    def randn(self, *shape):
        return FakeTensor(shape, label="randn")

    def no_grad(self):
        return FakeNoGrad()


class FakeHubMixin:
    @classmethod
    def from_pretrained(cls, pretrained_model_id):
        instance = cls()
        instance.pretrained_model_id = pretrained_model_id
        return instance


class TestSpusPipeline(unittest.TestCase):
    def test_append_run_log_writes_message(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-log-") as tmp_dir:
            run_dir = Path(tmp_dir)

            log_path = append_run_log(run_dir, "Starting epoch 1/10")

            self.assertEqual(log_path, run_dir / "training.log")
            self.assertTrue(log_path.is_file())
            self.assertIn("Starting epoch 1/10", log_path.read_text(encoding="utf-8"))

    def test_fns_kf_paper_split_helper_matches_reference_counts(self) -> None:
        try:
            from spus_ft.dataloader import get_fns_kf_paper_split_trajectory_indices
        except ImportError:
            self.skipTest("dataloader scientific dependencies are not installed in this test environment")

        train_indices, val_indices, test_indices = get_fns_kf_paper_split_trajectory_indices(20000)

        self.assertEqual(len(train_indices), 19640)
        self.assertEqual(len(val_indices), 120)
        self.assertEqual(len(test_indices), 240)
        self.assertEqual(train_indices[0], 0)
        self.assertEqual(val_indices[0], 19640)
        self.assertEqual(test_indices[0], 19760)

    def test_resolve_spus_repo_path_defaults_to_models_root_location(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-repo-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_spus_36m_repo_path(paths=paths)

            self.assertEqual(
                resolved,
                (paths.models_root / "spus-pde-unet-36m-v2").resolve(),
            )

    def test_resolve_spus_pretrained_source_defaults_to_local_model_repo(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-source-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_spus_pretrained_source(paths=paths)

            self.assertEqual(
                resolved,
                str((paths.models_root / "spus-pde-unet-36m-v2").resolve()),
            )

    def test_resolve_spus_training_repo_path_defaults_to_v2_model_snapshot_location(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-train-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_spus_training_repo_path(paths=paths)

            self.assertEqual(
                resolved,
                (paths.models_root / "spus-pde-unet-36m-v2").resolve(),
            )

    def test_resolve_spus_model_definition_repo_prefers_local_pretrained_source_with_model_py(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-model-def-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            local_model_repo = paths.models_root / "spus-pde-unet-36m-v2"
            local_model_repo.mkdir(parents=True, exist_ok=True)
            (local_model_repo / "model.py").write_text("class Unet2D: pass\n", encoding="utf-8")

            resolved = resolve_spus_model_definition_repo(
                paths=paths,
                pretrained_source=local_model_repo,
            )

            self.assertEqual(resolved, local_model_repo.resolve())

    def test_resolve_spus_inference_strategy_uses_five_channel_pretrained_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-infer-pretrained-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            model_root = paths.models_root / "spus-pde-unet-36m-v2"
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "model.py").write_text("class Unet2D: pass\n", encoding="utf-8")

            strategy = resolve_spus_inference_strategy(
                paths=paths,
                pretrained_source=model_root,
            )

            self.assertEqual(strategy['loader'], 'pretrained')
            self.assertEqual(strategy['model_class_name'], 'Unet2D')
            self.assertEqual(strategy['target_channels'], 5)
            self.assertFalse(strategy['is_fine_tuned'])

    def test_resolve_spus_inference_strategy_uses_two_channel_fns_kf_model(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-infer-fns-kf-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            model_root = paths.models_root / "spus-pde-unet-36m-v2-fns-kf"
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "model.py").write_text(
                "class Unet2D: pass\nclass SPUSFNSKF: pass\n",
                encoding="utf-8",
            )

            strategy = resolve_spus_inference_strategy(
                paths=paths,
                pretrained_source=model_root,
            )

            self.assertEqual(strategy['loader'], 'fns_kf')
            self.assertEqual(strategy['model_class_name'], 'SPUSFNSKF')
            self.assertIsNone(strategy['target_channels'])
            self.assertTrue(strategy['is_fine_tuned'])

    def test_resolve_spus_inference_strategy_keeps_checkpoint_inference_on_unet2d(self) -> None:
        strategy = resolve_spus_inference_strategy(checkpoint_path='/tmp/checkpoint.pt')

        self.assertEqual(strategy['loader'], 'checkpoint')
        self.assertEqual(strategy['model_class_name'], 'Unet2D')
        self.assertEqual(strategy['target_channels'], 5)
        self.assertTrue(strategy['is_fine_tuned'])

    def test_resolve_fns_kf_nc_path_defaults_under_data_root(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-data-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_fns_kf_nc_path(paths=paths, dataset_path=repo_root / "custom.nc")

            self.assertEqual(resolved, (repo_root / "custom.nc").resolve())

    def test_resolve_fns_kf_nc_path_accepts_string_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-data-str-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)

            resolved = resolve_fns_kf_nc_path(paths=paths, dataset_path=str(repo_root / "custom.nc"))

            self.assertEqual(resolved, (repo_root / "custom.nc").resolve())

    def test_select_test_trajectory_indices_by_relative_test_index(self) -> None:
        selected, actual = select_test_trajectory_indices([8, 9, 10], test_trajectory_index=1)

        self.assertEqual(selected, [9])
        self.assertEqual(actual, 9)

    def test_select_test_trajectory_indices_respects_max_when_no_specific_index(self) -> None:
        selected, actual = select_test_trajectory_indices([8, 9, 10], max_test_trajectories=2)

        self.assertEqual(selected, [8, 9])
        self.assertIsNone(actual)

    def test_select_test_trajectory_indices_raises_for_bad_index(self) -> None:
        with self.assertRaises(IndexError):
            select_test_trajectory_indices([8, 9], test_trajectory_index=2)

    def test_select_split_trajectory_indices_by_relative_index(self) -> None:
        selected, actual = select_split_trajectory_indices([3, 4, 5], split_trajectory_index=2)

        self.assertEqual(selected, [5])
        self.assertEqual(actual, 5)

    def test_select_split_trajectory_indices_respects_max(self) -> None:
        selected, actual = select_split_trajectory_indices([3, 4, 5], max_trajectories=2)

        self.assertEqual(selected, [3, 4])
        self.assertIsNone(actual)

    def test_select_split_trajectory_indices_raises_for_bad_index(self) -> None:
        with self.assertRaises(IndexError):
            select_split_trajectory_indices([3], split_trajectory_index=1)

    def test_run_spus_36m_uses_submodule_model_definition(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-run-") as tmp_dir:
            repo_root = Path(tmp_dir)
            model_root = TutorialPaths.discover(repo_root).models_root / "spus-pde-unet-36m-v2"
            model_root.mkdir(parents=True, exist_ok=True)
            (model_root / "model.py").write_text(
                textwrap.dedent(
                    """
                    class Unet2D:
                        def eval(self):
                            self.was_eval = True

                        def __call__(self, x):
                            return {
                                'received_shape': getattr(x, 'shape', None),
                                'pretrained_model_id': getattr(self, 'pretrained_model_id', None),
                                'was_eval': getattr(self, 'was_eval', False),
                            }
                    """
                ).strip(),
                encoding="utf-8",
            )

            paths = TutorialPaths.discover(repo_root)
            outcome = run_spus_36m(
                paths=paths,
                torch_module=FakeTorch(),
                hub_mixin_class=FakeHubMixin,
                dependency_check=lambda: {"ok": True, "installed": {}, "missing": []},
            )

            self.assertEqual(
                outcome["model_repo_path"],
                model_root.resolve(),
            )
            self.assertEqual(outcome["pretrained_model_id"], "siddik-lanl/spus-pde-unet-36m-v2")
            self.assertEqual(
                outcome["pretrained_source"],
                str(model_root.resolve()),
            )
            self.assertEqual(
                outcome["model"].pretrained_model_id,
                str(model_root.resolve()),
            )
            self.assertEqual(outcome["input"].shape, (1, 1, 5, 128, 128))
            self.assertEqual(outcome["prediction"]["received_shape"], (1, 1, 5, 128, 128))
            self.assertEqual(
                outcome["prediction"]["pretrained_model_id"],
                str(model_root.resolve()),
            )
            self.assertTrue(outcome["prediction"]["was_eval"])

    def test_evaluate_rollout_can_reuse_existing_rollout_outcome(self) -> None:
        fake_rollout_outcome = {
            'run_dir': Path('/tmp/existing-rollout'),
            'predictions_path': Path('/tmp/existing-rollout/rollout.pt'),
            'summary_path': Path('/tmp/existing-rollout/summary.json'),
        }
        fake_rollout_payload = {
            'predictions': [[['pred']]],
            'targets': [[['target']]],
            'metadata': {
                'split_name': 'test',
                'rollout_source': {'checkpoint_path': '/tmp/ckpt.pt'},
            },
        }
        fake_evaluation = {
            'average_mse': 0.125,
            'timestep_mse': [0.125],
            'per_trajectory_mse': [0.125],
            'metadata': {'split_name': 'test'},
        }
        fake_preview_paths = {
            'preview_predictions_path': Path('/tmp/existing-rollout/preview_predictions.pt'),
            'preview_targets_path': Path('/tmp/existing-rollout/preview_targets.pt'),
            'preview_count': 1,
        }

        with patch('spus_ft.pipeline.infer_rollout', side_effect=AssertionError('infer_rollout should not be called')), patch(
            'spus_ft.pipeline.load_torch_rollout', return_value=fake_rollout_payload
        ) as mock_load, patch('spus_ft.pipeline.save_evaluation', return_value=fake_evaluation) as mock_save_eval, patch(
            'spus_ft.pipeline.save_preview_tensors', return_value=fake_preview_paths
        ) as mock_save_preview, patch('spus_ft.pipeline.append_run_log') as mock_log, patch.dict(
            'sys.modules', {'torch': SimpleNamespace()}
        ):
            outcome = evaluate_rollout(rollout_outcome=fake_rollout_outcome)

        mock_load.assert_called_once()
        mock_save_eval.assert_called_once()
        mock_save_preview.assert_called_once()
        mock_log.assert_called_once()
        self.assertEqual(outcome['run_dir'], fake_rollout_outcome['run_dir'])
        self.assertEqual(outcome['evaluation'], fake_evaluation)
        self.assertEqual(outcome['preview_count'], 1)

    def test_evaluate_rollout_can_use_existing_run_directory(self) -> None:
        fake_rollout_payload = {
            'predictions': [[['pred']]],
            'targets': [[['target']]],
            'metadata': {
                'split_name': 'val',
                'rollout_source': {'checkpoint_path': None},
            },
        }
        fake_evaluation = {
            'average_mse': 0.5,
            'timestep_mse': [0.5],
            'per_trajectory_mse': [0.5],
            'metadata': {'split_name': 'val'},
        }
        fake_preview_paths = {
            'preview_predictions_path': Path('/tmp/existing-rollout/preview_predictions.pt'),
            'preview_targets_path': Path('/tmp/existing-rollout/preview_targets.pt'),
            'preview_count': 1,
        }

        with patch('spus_ft.pipeline.load_torch_rollout', return_value=fake_rollout_payload) as mock_load, patch(
            'spus_ft.pipeline.save_evaluation', return_value=fake_evaluation
        ), patch('spus_ft.pipeline.save_preview_tensors', return_value=fake_preview_paths), patch(
            'spus_ft.pipeline.append_run_log'
        ), patch.dict('sys.modules', {'torch': SimpleNamespace()}):
            outcome = evaluate_rollout(rollout_run_dir='/tmp/existing-rollout')

        self.assertEqual(outcome['run_dir'], Path('/tmp/existing-rollout').resolve())
        mock_load.assert_called_once()
