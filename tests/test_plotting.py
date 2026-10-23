import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


try:
    import matplotlib

    matplotlib.use("Agg")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TestPlotting(unittest.TestCase):
    def test_build_evaluation_label_uses_training_count(self) -> None:
        from spus_ft.plotting import build_evaluation_label

        label = build_evaluation_label(
            {
                'split_name': 'test',
                'num_train_trajectories': 8,
                'rollout_source': {
                    'checkpoint_path': '/tmp/checkpoint.pt',
                },
            }
        )

        self.assertEqual(label, 'fine-tuned (8 train trajectories) [test]')

    def test_build_evaluation_label_marks_hf_fine_tuned_rollout(self) -> None:
        from spus_ft.plotting import build_evaluation_label

        label = build_evaluation_label(
            {
                'split_name': 'test',
                'rollout_source': {
                    'checkpoint_path': None,
                    'model_variant': 'fns_kf',
                },
            }
        )

        self.assertEqual(label, 'fine-tuned [test]')

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_plot_rollout_frames_writes_image(self) -> None:
        import matplotlib.pyplot as pyplot

        from spus_ft.plotting import plot_rollout_frames

        with tempfile.TemporaryDirectory(prefix='spus-ft-plot-frames-') as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_path = tmp_path / 'frames.png'
            payload = {
                'predictions': [
                    [
                        [[[1.0, 1.5], [2.0, 2.5]]],
                        [[[2.0, 2.5], [3.0, 3.5]]],
                        [[[3.0, 3.5], [4.0, 4.5]]],
                    ]
                ],
                'targets': [
                    [
                        [[[1.2, 1.4], [1.8, 2.1]]],
                        [[[2.2, 2.4], [2.8, 3.1]]],
                        [[[3.2, 3.4], [3.8, 4.1]]],
                    ]
                ],
                'metadata': {
                    'split_name': 'test',
                    'num_train_trajectories': 1,
                    'rollout_source': {'checkpoint_path': None},
                },
            }

            result = plot_rollout_frames(
                payload=payload,
                trajectory_index=0,
                timesteps=(1, 2, 3),
                channel_index=0,
                channel_names=['density'],
                save_path=save_path,
            )

            self.assertTrue(save_path.is_file())
            self.assertEqual(result['timesteps'], [1, 2, 3])
            self.assertEqual(result['axes'].shape, (3, 3))
            self.assertAlmostEqual(result['subtitle'].get_position()[1], 0.24)
            self.assertEqual(result['channel_label'], 'Channel 0 (density)')
            self.assertEqual(result['colorbar'].ax.get_xlabel(), 'Channel 0 (density) value — pretrained (1 train trajectories) [test]')
            self.assertEqual(
                result['difference_colorbar'].ax.get_xlabel(),
                'Channel 0 (density) difference (predicted − true) — pretrained (1 train trajectories) [test]',
            )
            self.assertAlmostEqual(result['difference_range']['min'], -0.4)
            self.assertAlmostEqual(result['difference_range']['max'], 0.4)
            pyplot.close(result['figure'])

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_plot_rollout_frames_separate_scales_writes_image(self) -> None:
        import matplotlib.pyplot as pyplot

        from spus_ft.plotting import plot_rollout_frames_separate_scales

        with tempfile.TemporaryDirectory(prefix='spus-ft-plot-frames-separate-') as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_path = tmp_path / 'frames_separate.png'
            payload = {
                'predictions': [
                    [
                        [[[10.0, 15.0], [20.0, 25.0]]],
                        [[[20.0, 25.0], [30.0, 35.0]]],
                        [[[30.0, 35.0], [40.0, 45.0]]],
                    ]
                ],
                'targets': [
                    [
                        [[[1.2, 1.4], [1.8, 2.1]]],
                        [[[2.2, 2.4], [2.8, 3.1]]],
                        [[[3.2, 3.4], [3.8, 4.1]]],
                    ]
                ],
                'metadata': {
                    'split_name': 'test',
                    'num_train_trajectories': 1,
                    'rollout_source': {'checkpoint_path': None},
                },
            }

            result = plot_rollout_frames_separate_scales(
                payload=payload,
                trajectory_index=0,
                timesteps=(1, 2, 3),
                channel_index=0,
                channel_names=['density'],
                save_path=save_path,
            )

            self.assertTrue(save_path.is_file())
            self.assertEqual(result['timesteps'], [1, 2, 3])
            self.assertEqual(result['prediction_range'], {'min': 10.0, 'max': 45.0})
            self.assertEqual(result['target_range'], {'min': 1.2, 'max': 4.1})
            self.assertEqual(
                result['prediction_colorbar'].ax.get_xlabel(),
                'Channel 0 (density) predicted value — pretrained (1 train trajectories) [test]',
            )
            self.assertEqual(
                result['target_colorbar'].ax.get_xlabel(),
                'Channel 0 (density) true value — pretrained (1 train trajectories) [test]',
            )
            pyplot.close(result['figure'])

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_plot_multiple_rollout_mse_curves_writes_image(self) -> None:
        import matplotlib.pyplot as pyplot

        from spus_ft.plotting import plot_multiple_rollout_mse_curves

        with tempfile.TemporaryDirectory(prefix='spus-ft-plot-mse-') as tmp_dir:
            tmp_path = Path(tmp_dir)
            first_run = tmp_path / 'run_a'
            second_run = tmp_path / 'run_b'
            first_run.mkdir()
            second_run.mkdir()

            (first_run / 'evaluation.json').write_text(
                json.dumps(
                    {
                        'average_mse': 0.2,
                        'timestep_mse': [0.1, 0.2, 0.3],
                        'metadata': {
                            'split_name': 'test',
                            'num_train_trajectories': 1,
                            'rollout_source': {'checkpoint_path': '/tmp/ckpt_a.pt'},
                        },
                    }
                ),
                encoding='utf-8',
            )
            (second_run / 'evaluation.json').write_text(
                json.dumps(
                    {
                        'average_mse': 0.1,
                        'timestep_mse': [0.05, 0.1, 0.15],
                        'metadata': {
                            'split_name': 'test',
                            'num_train_trajectories': 8,
                            'rollout_source': {'checkpoint_path': '/tmp/ckpt_b.pt'},
                        },
                    }
                ),
                encoding='utf-8',
            )

            save_path = tmp_path / 'mse_compare.png'
            result = plot_multiple_rollout_mse_curves(
                [first_run, second_run],
                save_path=save_path,
            )

            self.assertTrue(save_path.is_file())
            self.assertEqual(len(result['ax'].lines), 2)
            pyplot.close(result['figure'])

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_plot_full_rollout_frames_writes_image(self) -> None:
        import matplotlib.pyplot as pyplot

        from spus_ft.plotting import plot_full_rollout_frames

        with tempfile.TemporaryDirectory(prefix='spus-ft-plot-full-rollout-') as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_path = tmp_path / 'full_rollout.png'
            payload = {
                'predictions': [
                    [
                        [[[1.0, 1.5], [2.0, 2.5]]],
                        [[[2.0, 2.5], [3.0, 3.5]]],
                        [[[3.0, 3.5], [4.0, 4.5]]],
                        [[[4.0, 4.5], [5.0, 5.5]]],
                    ]
                ],
                'targets': [
                    [
                        [[[1.2, 1.4], [1.8, 2.1]]],
                        [[[2.2, 2.4], [2.8, 3.1]]],
                        [[[3.2, 3.4], [3.8, 4.1]]],
                        [[[4.2, 4.4], [4.8, 5.1]]],
                    ]
                ],
                'metadata': {
                    'split_name': 'test',
                    'num_train_trajectories': 1,
                    'rollout_source': {'checkpoint_path': None},
                },
            }

            result = plot_full_rollout_frames(
                payload=payload,
                trajectory_index=0,
                channel_index=0,
                channel_names=['density'],
                max_columns=3,
                save_path=save_path,
            )

            self.assertTrue(save_path.is_file())
            self.assertEqual(result['timesteps'], [1, 2, 3, 4])
            self.assertEqual(result['trajectory_index'], 0)
            self.assertEqual(result['channel_index'], 0)
            self.assertEqual(result['channel_label'], 'Channel 0 (density)')
            self.assertEqual(result['axes'].shape, (4, 3))
            pyplot.close(result['figure'])

    @unittest.skipUnless(HAS_MATPLOTLIB, 'matplotlib is not installed')
    def test_plot_ground_truth_trajectory_frames_writes_image(self) -> None:
        import matplotlib.pyplot as pyplot

        from spus_ft.plotting import plot_ground_truth_trajectory_frames

        class FakeTrajectoryTensor:
            def __init__(self, value):
                self.value = value

            def __getitem__(self, item):
                selected = self.value[item]
                if isinstance(selected, list):
                    return FakeTrajectoryTensor(selected)
                return selected

            def detach(self):
                return self

            def cpu(self):
                return self

            def tolist(self):
                return self.value

        trajectory = [[[[float(timestep), float(timestep + 1)], [float(timestep + 2), float(timestep + 3)]]]
                      for timestep in range(4)]

        with tempfile.TemporaryDirectory(prefix='spus-ft-plot-ground-truth-') as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_path = tmp_path / 'ground_truth.png'

            with patch('spus_ft.dataloader.infer_variable_name', return_value='solution'), patch(
                'spus_ft.dataloader.get_train_val_test_trajectory_indices_from_file',
                return_value=([0], [1], [2, 3], {'shape': [4, 4, 1, 2, 2]}),
            ), patch(
                'spus_ft.dataloader.load_trajectories_by_indices',
                return_value=(FakeTrajectoryTensor([trajectory]), 'solution'),
            ):
                result = plot_ground_truth_trajectory_frames(
                    dataset_path='/tmp/FNS-KF.nc',
                    split_name='test',
                    split_trajectory_index=1,
                    channel_index=0,
                    channel_names=['density'],
                    max_columns=3,
                    save_path=save_path,
                )

            self.assertTrue(save_path.is_file())
            self.assertEqual(result['split_name'], 'test')
            self.assertEqual(result['selected_trajectory_index'], 3)
            self.assertEqual(result['channel_label'], 'Channel 0 (density)')
            self.assertEqual(result['num_frames'], 4)
            self.assertEqual(result['axes'].shape, (2, 3))
            self.assertIsNone(result['subtitle'])
            self.assertIn('Split trajectory index 1 (global trajectory 3) for Channel 0 (density)', result['figure']._suptitle.get_text())
            self.assertEqual(result['colorbar'].ax.get_xlabel(), 'Ground-truth Channel 0 (density) value')
            pyplot.close(result['figure'])
