from pathlib import Path
from types import SimpleNamespace
from types import ModuleType
import tempfile
import textwrap
import unittest
from unittest.mock import patch

from spus_ft.paths import TutorialPaths
from spus_ft.pipeline import (
    append_run_log,
    evaluate_rollout,
    fine_tune,
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
    class _Cuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def manual_seed(seed):
            return None

        @staticmethod
        def manual_seed_all(seed):
            return None

    class _Backends:
        class cudnn:
            deterministic = False
            benchmark = True

    class optim:
        class AdamW:
            def __init__(self, params, lr):
                self.params = list(params)
                self.lr = lr

            def state_dict(self):
                return {"lr": self.lr}

    cuda = _Cuda()
    backends = _Backends()

    def randn(self, *shape):
        return FakeTensor(shape, label="randn")

    def no_grad(self):
        return FakeNoGrad()

    def manual_seed(self, seed):
        return None

    def device(self, name):
        return name


class FakeDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def __len__(self):
        if len(self.dataset) == 0:
            return 0
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        return iter(())


class FakeModel:
    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return [object()]

    def state_dict(self):
        return {"weights": 1}

    def train(self):
        return self

    def eval(self):
        return self


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

    def test_append_run_log_does_not_echo_by_default(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-log-") as tmp_dir, patch('builtins.print') as mock_print:
            run_dir = Path(tmp_dir)

            append_run_log(run_dir, "Quiet log message")

            mock_print.assert_not_called()

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

    def test_fine_tune_records_data_usage_and_elapsed_time_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-fine-tune-") as tmp_dir:
            repo_root = Path(tmp_dir)
            paths = TutorialPaths.discover(repo_root)
            fake_train_set = [0] * 12
            fake_val_set = [0] * 4
            fake_test_set = [0] * 6
            saved = {}

            def fake_save_checkpoint(run_dir, checkpoint, metadata, filename='best_checkpoint.pt', torch_module=None):
                saved['metadata'] = metadata
                return Path(run_dir) / filename

            fake_torch = ModuleType('torch')
            fake_torch_impl = FakeTorch()
            fake_torch.manual_seed = fake_torch_impl.manual_seed
            fake_torch.device = fake_torch_impl.device
            fake_torch.cuda = fake_torch_impl.cuda
            fake_torch.backends = fake_torch_impl.backends
            fake_torch.optim = fake_torch_impl.optim

            fake_torch_nn = ModuleType('torch.nn')
            fake_torch_functional = ModuleType('torch.nn.functional')
            fake_torch_nn.functional = fake_torch_functional

            fake_torch_utils = ModuleType('torch.utils')
            fake_torch_utils_data = ModuleType('torch.utils.data')
            fake_torch_utils_data.DataLoader = FakeDataLoader
            fake_torch_utils.data = fake_torch_utils_data

            fake_dataloader = SimpleNamespace(
                infer_variable_name=lambda data_path, variable_name: 'data',
                get_train_val_test_split_from_file=lambda *args, **kwargs: (fake_train_set, fake_val_set, fake_test_set),
                get_train_val_test_trajectory_indices_from_file=lambda *args, **kwargs: (
                    list(range(3)), [3], [4, 5], {'shape': [6, 21, 2, 128, 128]}
                ),
                close_all_datasets=lambda dataset: None,
            )

            with patch('spus_ft.pipeline.require_spus_runtime_dependencies', return_value={'ok': True}), patch(
                'spus_ft.pipeline.resolve_fns_kf_nc_path', return_value=Path('/tmp/FNS-KF.nc')
            ), patch('spus_ft.pipeline._timestamp_label', return_value='20261104-000000'), patch(
                'spus_ft.pipeline.load_spus_36m_training_model',
                return_value={
                    'model': FakeModel(),
                    'pretrained_source': '/tmp/pretrained',
                    'training_repo_path': Path('/tmp/model-repo'),
                },
            ), patch('spus_ft.pipeline.save_checkpoint', side_effect=fake_save_checkpoint), patch(
                'time.perf_counter', side_effect=[100.0, 112.5]
            ), patch.dict(
                'sys.modules',
                {
                    'torch': fake_torch,
                    'torch.nn': fake_torch_nn,
                    'torch.nn.functional': fake_torch_functional,
                    'torch.utils': fake_torch_utils,
                    'torch.utils.data': fake_torch_utils_data,
                    'spus_ft.dataloader': fake_dataloader,
                },
            ):
                outcome = fine_tune(
                    paths=paths,
                    dataset_path=Path('/tmp/FNS-KF.nc'),
                    epochs=0,
                    batch_size=5,
                )

            metrics = outcome['fine_tune_metrics']
            self.assertEqual(metrics['num_train_trajectories'], 3)
            self.assertEqual(metrics['num_train_samples'], 12)
            self.assertEqual(metrics['num_val_samples'], 4)
            self.assertEqual(metrics['num_test_samples'], 6)
            self.assertEqual(metrics['num_train_batches_per_epoch'], 3)
            self.assertEqual(metrics['num_val_batches_per_epoch'], 1)
            self.assertEqual(metrics['fine_tune_duration_seconds'], 12.5)
            self.assertAlmostEqual(metrics['fine_tune_duration_minutes'], 12.5 / 60.0)
            self.assertEqual(outcome['metadata']['fine_tune_metrics'], metrics)
            self.assertEqual(saved['metadata']['fine_tune_metrics'], metrics)
