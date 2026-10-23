from __future__ import annotations

import argparse
from pathlib import Path

from spus_ft.data import clone_spus_model_collection, download_fns_kf_dataset, download_spus_model_collection
from spus_ft.pipeline import evaluate_rollout, fine_tune, infer_rollout, run_synthetic_demo


def _add_shared_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-path", type=Path, help="Path to the assembled FNS-KF NetCDF file")
    parser.add_argument("--variable-name", help="Variable name to read from the NetCDF file")
    parser.add_argument("--num-train-trajectories", type=int, default=1, help="Number of training trajectories to use")
    parser.add_argument("--pretrained-source", help="Local model snapshot path or Hugging Face model id")
    parser.add_argument("--model-repo-path", type=Path, help="Path to the repo providing `model.py`")
    parser.add_argument("--training-repo-path", type=Path, help="Path to the SPUS training repo checkout")
    parser.add_argument("--device", help="Requested torch device, for example `cpu` or `cuda`")


def _add_shared_rollout_arguments(parser: argparse.ArgumentParser) -> None:
    _add_shared_pipeline_arguments(parser)
    parser.add_argument(
        "--split-name",
        choices=("train", "val", "test"),
        default="test",
        help="Split to evaluate",
    )
    parser.add_argument(
        "--split-trajectory-index",
        type=int,
        help="Relative trajectory index within the selected split",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        help="Maximum number of trajectories to process from the selected split",
    )
    parser.add_argument("--checkpoint-path", type=Path, help="Checkpoint path for a fine-tuned model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SPUS tutorial helpers")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Run the synthetic demo pipeline")
    subparsers.add_parser("download-fns-kf", help="Download the FNS-KF dataset into the data root")
    subparsers.add_parser("download-spus-models", help="Download SPUS pretrained and fine-tuned model snapshots into the models root")
    subparsers.add_parser("clone-spus-models", help="Git clone SPUS pretrained and fine-tuned Hugging Face repos into the models root")

    fine_tune_parser = subparsers.add_parser("fine-tune", help="Fine-tune the SPUS model on the assembled FNS-KF dataset")
    _add_shared_pipeline_arguments(fine_tune_parser)
    fine_tune_parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    fine_tune_parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    fine_tune_parser.add_argument("--learning-rate", type=float, default=1e-4, help="Optimizer learning rate")

    infer_parser = subparsers.add_parser("infer-rollout", help="Run autoregressive inference for one or more trajectories")
    _add_shared_rollout_arguments(infer_parser)

    evaluate_parser = subparsers.add_parser("evaluate-rollout", help="Evaluate an existing or freshly generated rollout")
    _add_shared_rollout_arguments(evaluate_parser)
    evaluate_parser.add_argument("--rollout-run-dir", type=Path, help="Existing rollout run directory to evaluate")
    evaluate_parser.add_argument(
        "--preview-trajectory-count",
        type=int,
        default=5,
        help="Number of trajectories to save into preview tensors",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "demo"):
        outcome = run_synthetic_demo()
        print(f"Saved synthetic demo outputs to {outcome['run_dir']}")
        print(f"Average MSE: {outcome['summary']['average_mse']:.6f}")
        return

    if args.command == "download-fns-kf":
        outcome = download_fns_kf_dataset()
        print(f"Downloaded {outcome['repo_id']} into {outcome['target_dir']}")
        print("Included patterns:")
        for pattern in outcome["allow_patterns"]:
            print(f"- {pattern}")
        return

    if args.command == "download-spus-models":
        outcome = download_spus_model_collection()
        print(f"Downloaded {len(outcome['downloads'])} model snapshots into {outcome['models_root']}")
        for item in outcome["downloads"]:
            print(f"- {item['repo_id']} -> {item['target_dir']}")
        return

    if args.command == "clone-spus-models":
        outcome = clone_spus_model_collection()
        print(f"Cloned {len(outcome['clones'])} model repos into {outcome['models_root']}")
        for item in outcome["clones"]:
            print(f"- {item['repo_id']} -> {item['target_dir']}")
        return

    if args.command == "fine-tune":
        outcome = fine_tune(
            dataset_path=args.dataset_path,
            variable_name=args.variable_name,
            num_train_trajectories=args.num_train_trajectories,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            pretrained_source=args.pretrained_source,
            model_repo_path=args.model_repo_path,
            training_repo_path=args.training_repo_path,
            device=args.device,
        )
        print(f"Saved fine-tuning outputs to {outcome['run_dir']}")
        print(f"Best checkpoint: {outcome['checkpoint_path']}")
        print(f"Best validation loss: {outcome['best_val_loss']:.6f}")
        return

    if args.command == "infer-rollout":
        outcome = infer_rollout(
            dataset_path=args.dataset_path,
            variable_name=args.variable_name,
            num_train_trajectories=args.num_train_trajectories,
            pretrained_source=args.pretrained_source,
            model_repo_path=args.model_repo_path,
            checkpoint_path=args.checkpoint_path,
            training_repo_path=args.training_repo_path,
            device=args.device,
            max_test_trajectories=args.num_trajectories,
            test_trajectory_index=args.split_trajectory_index,
            split_name=args.split_name,
        )
        print(f"Saved rollout outputs to {outcome['run_dir']}")
        print(f"Rollout tensor: {outcome['predictions_path']}")
        print(f"Average MSE: {outcome['summary']['average_mse']:.6f}")
        return

    if args.command == "evaluate-rollout":
        outcome = evaluate_rollout(
            dataset_path=args.dataset_path,
            variable_name=args.variable_name,
            num_train_trajectories=args.num_train_trajectories,
            pretrained_source=args.pretrained_source,
            model_repo_path=args.model_repo_path,
            checkpoint_path=args.checkpoint_path,
            training_repo_path=args.training_repo_path,
            device=args.device,
            num_trajectories=args.num_trajectories,
            split_trajectory_index=args.split_trajectory_index,
            split_name=args.split_name,
            preview_trajectory_count=args.preview_trajectory_count,
            rollout_run_dir=args.rollout_run_dir,
        )
        print(f"Saved evaluation outputs to {outcome['run_dir']}")
        print(f"Evaluation JSON: {outcome['evaluation_path']}")
        print(f"Average MSE: {outcome['evaluation']['average_mse']:.6f}")
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
