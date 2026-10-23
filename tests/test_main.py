import unittest
from pathlib import Path
from unittest.mock import patch

from spus_ft import __main__


class TestMain(unittest.TestCase):
    def test_fine_tune_command_invokes_pipeline(self) -> None:
        with patch("spus_ft.__main__.fine_tune") as mock_fine_tune, patch(
            "sys.argv",
            [
                "spus_ft",
                "fine-tune",
                "--dataset-path",
                "/tmp/FNS-KF.nc",
                "--num-train-trajectories",
                "16",
                "--epochs",
                "3",
                "--batch-size",
                "2",
                "--learning-rate",
                "0.001",
                "--pretrained-source",
                "/tmp/model",
                "--device",
                "cpu",
            ],
        ):
            mock_fine_tune.return_value = {
                "run_dir": "/tmp/run",
                "checkpoint_path": "/tmp/run/best_checkpoint.pt",
                "best_val_loss": 0.25,
            }

            __main__.main()

        mock_fine_tune.assert_called_once_with(
            dataset_path=Path("/tmp/FNS-KF.nc"),
            variable_name=None,
            num_train_trajectories=16,
            epochs=3,
            batch_size=2,
            learning_rate=0.001,
            pretrained_source="/tmp/model",
            model_repo_path=None,
            training_repo_path=None,
            device="cpu",
        )

    def test_infer_rollout_command_invokes_pipeline(self) -> None:
        with patch("spus_ft.__main__.infer_rollout") as mock_infer, patch(
            "sys.argv",
            [
                "spus_ft",
                "infer-rollout",
                "--dataset-path",
                "/tmp/FNS-KF.nc",
                "--num-train-trajectories",
                "8",
                "--checkpoint-path",
                "/tmp/run/best_checkpoint.pt",
                "--split-name",
                "val",
                "--split-trajectory-index",
                "1",
                "--num-trajectories",
                "2",
            ],
        ):
            mock_infer.return_value = {
                "run_dir": "/tmp/run",
                "predictions_path": "/tmp/run/rollout.pt",
                "summary": {"average_mse": 0.125},
            }

            __main__.main()

        mock_infer.assert_called_once_with(
            dataset_path=Path("/tmp/FNS-KF.nc"),
            variable_name=None,
            num_train_trajectories=8,
            pretrained_source=None,
            model_repo_path=None,
            checkpoint_path=Path("/tmp/run/best_checkpoint.pt"),
            training_repo_path=None,
            device=None,
            max_test_trajectories=2,
            test_trajectory_index=1,
            split_name="val",
        )

    def test_evaluate_rollout_command_invokes_pipeline(self) -> None:
        with patch("spus_ft.__main__.evaluate_rollout") as mock_evaluate, patch(
            "sys.argv",
            [
                "spus_ft",
                "evaluate-rollout",
                "--rollout-run-dir",
                "/tmp/existing-rollout",
                "--split-name",
                "test",
                "--preview-trajectory-count",
                "3",
            ],
        ):
            mock_evaluate.return_value = {
                "run_dir": "/tmp/existing-rollout",
                "evaluation_path": "/tmp/existing-rollout/evaluation.json",
                "evaluation": {"average_mse": 0.125},
            }

            __main__.main()

        mock_evaluate.assert_called_once_with(
            dataset_path=None,
            variable_name=None,
            num_train_trajectories=1,
            pretrained_source=None,
            model_repo_path=None,
            checkpoint_path=None,
            training_repo_path=None,
            device=None,
            num_trajectories=None,
            split_trajectory_index=None,
            split_name="test",
            preview_trajectory_count=3,
            rollout_run_dir=Path("/tmp/existing-rollout"),
        )

    def test_download_command_invokes_fns_kf_downloader(self) -> None:
        with patch("spus_ft.__main__.download_fns_kf_dataset") as mock_download, patch(
            "sys.argv", ["spus_ft", "download-fns-kf"]
        ):
            mock_download.return_value = {
                "repo_id": "camlab-ethz/FNS-KF",
                "target_dir": "/tmp/FNS-KF",
                "allow_patterns": ["solution_*.nc"],
            }

            __main__.main()

        mock_download.assert_called_once_with()

    def test_download_spus_models_command_invokes_model_downloader(self) -> None:
        with patch("spus_ft.__main__.download_spus_model_collection") as mock_download, patch(
            "sys.argv", ["spus_ft", "download-spus-models"]
        ):
            mock_download.return_value = {
                "models_root": "/tmp/spus-models",
                "downloads": [
                    {"repo_id": "siddik-lanl/spus-pde-unet-36m-v2", "target_dir": "/tmp/spus-models/spus-pde-unet-36m-v2"},
                ],
            }

            __main__.main()

        mock_download.assert_called_once_with()

    def test_clone_spus_models_command_invokes_model_cloner(self) -> None:
        with patch("spus_ft.__main__.clone_spus_model_collection") as mock_clone, patch(
            "sys.argv", ["spus_ft", "clone-spus-models"]
        ):
            mock_clone.return_value = {
                "models_root": "/tmp/spus-models",
                "clones": [
                    {"repo_id": "siddik-lanl/spus-pde-unet-36m-v2", "target_dir": "/tmp/spus-models/spus-pde-unet-36m-v2"},
                ],
            }

            __main__.main()

        mock_clone.assert_called_once_with()
