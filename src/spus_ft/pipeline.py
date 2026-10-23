from __future__ import annotations

import copy
import importlib
import json
import random
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from spus_ft.data import generate_synthetic_dataset
from spus_ft.inference import autoregressive_rollout, autoregressive_tensor_rollout
from spus_ft.modeling import make_baseline_model
from spus_ft.paths import TutorialPaths
from spus_ft.results import (
    load_torch_rollout,
    save_checkpoint,
    save_evaluation,
    save_preview_tensors,
    save_rollout,
    save_torch_rollout,
)
from spus_ft.runtime import require_spus_runtime_dependencies


DEFAULT_PRETRAINED_MODEL_ID = "siddik-lanl/spus-pde-unet-36m-v2"
DEFAULT_FNS_KF_DATASET_PATH = Path("/usr/projects/artimis/cahrens/tut-aux/spus-ft-data/FNS-KF/FNS-KF.nc")


def _coerce_optional_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def load_demo_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_spus_36m_repo_path(
    paths: Optional[TutorialPaths] = None,
    model_repo_path: Optional[str | Path] = None,
) -> Path:
    if model_repo_path is not None:
        return Path(model_repo_path).expanduser().resolve()

    tutorial_paths = paths or TutorialPaths.discover()
    return tutorial_paths.models_root / DEFAULT_PRETRAINED_MODEL_ID.split("/", 1)[1]


def resolve_spus_training_repo_path(
    paths: Optional[TutorialPaths] = None,
    training_repo_path: Optional[str | Path] = None,
) -> Path:
    if training_repo_path is not None:
        return Path(training_repo_path).expanduser().resolve()

    return resolve_spus_36m_repo_path(paths=paths)


def resolve_spus_pretrained_source(
    paths: Optional[TutorialPaths] = None,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
) -> str:
    if pretrained_source is not None:
        if isinstance(pretrained_source, Path):
            return str(pretrained_source.expanduser().resolve())
        return str(pretrained_source)

    local_repo = resolve_spus_36m_repo_path(paths=paths, model_repo_path=model_repo_path)
    return str(local_repo.resolve())


def resolve_spus_model_definition_repo(
    paths: Optional[TutorialPaths] = None,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[str | Path] = None,
    training_repo_path: Optional[str | Path] = None,
) -> Path:
    if model_repo_path is not None:
        return Path(model_repo_path).expanduser().resolve()

    if pretrained_source is not None:
        candidate = Path(pretrained_source).expanduser()
        if candidate.exists() and candidate.is_dir() and (candidate / "model.py").exists():
            return candidate.resolve()

    return resolve_spus_training_repo_path(paths=paths, training_repo_path=training_repo_path)


def resolve_fns_kf_nc_path(paths: Optional[TutorialPaths] = None, dataset_path: Optional[str | Path] = None) -> Path:
    if dataset_path is not None:
        return Path(dataset_path).expanduser().resolve()
    tutorial_paths = paths or TutorialPaths.discover()
    default_path = tutorial_paths.data_root / "FNS-KF" / "FNS-KF.nc"
    if default_path.exists():
        return default_path
    return DEFAULT_FNS_KF_DATASET_PATH


@contextmanager
def temporary_sys_path(path: Path) -> Iterator[None]:
    sys.path.insert(0, str(path))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(path))
        except ValueError:
            pass


def _load_model_module(repo_path: Path) -> Any:
    if not repo_path.exists():
        raise FileNotFoundError(
            "SPUS model repository not found at "
            f"{repo_path}. Download the model snapshot into the external models directory first."
        )

    with temporary_sys_path(repo_path):
        sys.modules.pop("model", None)
        return importlib.import_module("model")


def _load_named_model_class(repo_path: Path, class_name: str) -> type:
    model_module = _load_model_module(repo_path)
    try:
        return getattr(model_module, class_name)
    except AttributeError as error:
        raise AttributeError(f"{repo_path / 'model.py'} does not define {class_name}.") from error


def _make_hub_compatible_model_class(model_class: type, hub_mixin_class: Any) -> type:
    if hasattr(model_class, "from_pretrained"):
        return model_class

    mixin_class = hub_mixin_class
    if mixin_class is None:
        from huggingface_hub import PyTorchModelHubMixin as imported_mixin

        mixin_class = imported_mixin
    return type(f"SPUS{model_class.__name__}", (model_class, mixin_class), {})


def _repo_defines_model_class(repo_path: Path, class_name: str) -> bool:
    model_module = _load_model_module(repo_path)
    return hasattr(model_module, class_name)


def _load_unet2d_class(repo_path: Path) -> type:
    return _load_named_model_class(repo_path, "Unet2D")


def _load_spus_unet2d(repo_path: Path, hub_mixin_class: Any) -> type:
    unet_class = _load_unet2d_class(repo_path)
    return _make_hub_compatible_model_class(unet_class, hub_mixin_class)


def _load_spus_fns_kf_model_class(repo_path: Path, hub_mixin_class: Any) -> type:
    fns_kf_class = _load_named_model_class(repo_path, "SPUSFNSKF")
    return _make_hub_compatible_model_class(fns_kf_class, hub_mixin_class)


def _timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def append_run_log(run_dir: Path, message: str) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "training.log"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {message}"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(formatted + "\n")
    print(formatted)
    return log_path


def _set_random_seed(seed: int, torch_module: Any) -> None:
    random.seed(seed)
    torch_module.manual_seed(seed)
    if hasattr(torch_module, 'cuda') and torch_module.cuda.is_available():
        torch_module.cuda.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, 'cudnn'):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


def _resolve_device(torch_module: Any, requested_device: Optional[str] = None):
    if requested_device is not None:
        return torch_module.device(requested_device)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def select_test_trajectory_indices(
    test_indices: Sequence[int],
    test_trajectory_index: Optional[int] = None,
    max_test_trajectories: Optional[int] = None,
) -> tuple[list[int], Optional[int]]:
    selected_indices = list(test_indices)
    selected_test_trajectory_index = None

    if test_trajectory_index is not None:
        if test_trajectory_index < 0 or test_trajectory_index >= len(selected_indices):
            raise IndexError(
                f"test_trajectory_index={test_trajectory_index} is out of range for "
                f"{len(selected_indices)} available test trajectories"
            )
        selected_test_trajectory_index = selected_indices[test_trajectory_index]
        selected_indices = [selected_test_trajectory_index]

    if max_test_trajectories is not None:
        selected_indices = selected_indices[:max_test_trajectories]

    return selected_indices, selected_test_trajectory_index


def select_split_trajectory_indices(
    split_indices: Sequence[int],
    split_trajectory_index: Optional[int] = None,
    max_trajectories: Optional[int] = None,
) -> tuple[list[int], Optional[int]]:
    selected_indices = list(split_indices)
    selected_split_trajectory_index = None

    if split_trajectory_index is not None:
        if split_trajectory_index < 0 or split_trajectory_index >= len(selected_indices):
            raise IndexError(
                f"split_trajectory_index={split_trajectory_index} is out of range for "
                f"{len(selected_indices)} available trajectories in the selected split"
            )
        selected_split_trajectory_index = selected_indices[split_trajectory_index]
        selected_indices = [selected_split_trajectory_index]

    if max_trajectories is not None:
        selected_indices = selected_indices[:max_trajectories]

    return selected_indices, selected_split_trajectory_index


def load_spus_36m_model(
    paths: Optional[TutorialPaths] = None,
    model_repo_path: Optional[Path] = None,
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    hub_mixin_class: Any = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    spus_repo_path = resolve_spus_36m_repo_path(tutorial_paths, model_repo_path)
    resolved_pretrained_source = resolve_spus_pretrained_source(
        paths=tutorial_paths,
        pretrained_source=pretrained_source,
        model_repo_path=spus_repo_path,
    )

    mixin_class = hub_mixin_class
    if mixin_class is None:
        from huggingface_hub import PyTorchModelHubMixin as imported_mixin

        mixin_class = imported_mixin

    spus_model_class = _load_spus_unet2d(spus_repo_path, mixin_class)
    model = spus_model_class.from_pretrained(resolved_pretrained_source)
    model.eval()

    return {
        "paths": tutorial_paths,
        "model_repo_path": spus_repo_path,
        "pretrained_model_id": pretrained_model_id,
        "pretrained_source": resolved_pretrained_source,
        "model_class": spus_model_class,
        "model": model,
    }


def load_spus_36m_training_model(
    paths: Optional[TutorialPaths] = None,
    training_repo_path: Optional[Path] = None,
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    hub_mixin_class: Any = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    training_repo = resolve_spus_training_repo_path(tutorial_paths, training_repo_path)
    resolved_pretrained_source = resolve_spus_pretrained_source(
        paths=tutorial_paths,
        pretrained_source=pretrained_source,
        model_repo_path=model_repo_path,
    )
    model_definition_repo = resolve_spus_model_definition_repo(
        paths=tutorial_paths,
        pretrained_source=resolved_pretrained_source,
        model_repo_path=model_repo_path,
        training_repo_path=training_repo,
    )

    mixin_class = hub_mixin_class
    if mixin_class is None:
        from huggingface_hub import PyTorchModelHubMixin as imported_mixin

        mixin_class = imported_mixin

    spus_model_class = _load_spus_unet2d(model_definition_repo, mixin_class)
    model = spus_model_class.from_pretrained(resolved_pretrained_source)
    model.eval()
    return {
        'paths': tutorial_paths,
        'training_repo_path': training_repo,
        'model_definition_repo': str(model_definition_repo),
        'pretrained_model_id': pretrained_model_id,
        'pretrained_source': resolved_pretrained_source,
        'model_class': spus_model_class,
        'model': model,
    }


def load_spus_fns_kf_model(
    paths: Optional[TutorialPaths] = None,
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    training_repo_path: Optional[Path] = None,
    hub_mixin_class: Any = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    resolved_pretrained_source = resolve_spus_pretrained_source(
        paths=tutorial_paths,
        pretrained_source=pretrained_source,
        model_repo_path=model_repo_path,
    )
    model_definition_repo = resolve_spus_model_definition_repo(
        paths=tutorial_paths,
        pretrained_source=resolved_pretrained_source,
        model_repo_path=model_repo_path,
        training_repo_path=training_repo_path,
    )

    spus_model_class = _load_spus_fns_kf_model_class(model_definition_repo, hub_mixin_class)
    model = spus_model_class.from_pretrained(resolved_pretrained_source)
    model.eval()
    return {
        'paths': tutorial_paths,
        'training_repo_path': resolve_spus_training_repo_path(tutorial_paths, training_repo_path),
        'model_definition_repo': str(model_definition_repo),
        'pretrained_model_id': pretrained_model_id,
        'pretrained_source': resolved_pretrained_source,
        'model_class': spus_model_class,
        'model': model,
    }


def resolve_spus_inference_strategy(
    paths: Optional[TutorialPaths] = None,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    training_repo_path: Optional[Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    target_channels: Optional[int] = 5,
    pad_value: float = 1.0,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    if checkpoint_path is not None:
        return {
            'loader': 'checkpoint',
            'target_channels': target_channels,
            'pad_value': pad_value,
            'is_fine_tuned': True,
            'model_class_name': 'Unet2D',
        }

    resolved_pretrained_source = resolve_spus_pretrained_source(
        paths=tutorial_paths,
        pretrained_source=pretrained_source,
        model_repo_path=model_repo_path,
    )
    model_definition_repo = resolve_spus_model_definition_repo(
        paths=tutorial_paths,
        pretrained_source=resolved_pretrained_source,
        model_repo_path=model_repo_path,
        training_repo_path=training_repo_path,
    )

    if _repo_defines_model_class(model_definition_repo, 'SPUSFNSKF'):
        return {
            'loader': 'fns_kf',
            'target_channels': None,
            'pad_value': pad_value,
            'is_fine_tuned': True,
            'model_class_name': 'SPUSFNSKF',
        }

    return {
        'loader': 'pretrained',
        'target_channels': target_channels,
        'pad_value': pad_value,
        'is_fine_tuned': False,
        'model_class_name': 'Unet2D',
    }


def run_spus_36m(
    paths: Optional[TutorialPaths] = None,
    model_repo_path: Optional[Path] = None,
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    input_tensor: Any = None,
    torch_module: Any = None,
    hub_mixin_class: Any = None,
    dependency_check: Optional[Callable[[], dict]] = None,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    dependency_status = (dependency_check or require_spus_runtime_dependencies)()

    torch_lib = torch_module
    if torch_lib is None:
        import torch as imported_torch

        torch_lib = imported_torch

    model_details = load_spus_36m_model(
        paths=tutorial_paths,
        model_repo_path=model_repo_path,
        pretrained_model_id=pretrained_model_id,
        pretrained_source=pretrained_source,
        hub_mixin_class=hub_mixin_class,
    )
    model = model_details["model"]

    example_input = input_tensor
    if example_input is None:
        example_input = torch_lib.randn(1, 1, 5, 128, 128)

    with torch_lib.no_grad():
        prediction = model(example_input)

    return {
        **model_details,
        "dependency_status": dependency_status,
        "input": example_input,
        "prediction": prediction,
    }


def fine_tune(
    paths: Optional[TutorialPaths] = None,
    dataset_path: Optional[Path] = None,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = 1,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    split_strategy: str = 'fns_kf_paper',
    epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    training_repo_path: Optional[Path] = None,
    device: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
    target_channels: int = 5,
    pad_value: float = 1.0,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    dependency_status = require_spus_runtime_dependencies(("torch", "huggingface_hub", "safetensors", "netCDF4"))

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from spus_ft.dataloader import (
        close_all_datasets,
        get_train_val_test_split_from_file,
        get_train_val_test_trajectory_indices_from_file,
        infer_variable_name,
    )

    _set_random_seed(seed, torch)
    target_dataset_path = resolve_fns_kf_nc_path(tutorial_paths, dataset_path)
    resolved_variable_name = infer_variable_name(str(target_dataset_path), variable_name)
    train_set, val_set, test_set = get_train_val_test_split_from_file(
        str(target_dataset_path),
        variable_name=resolved_variable_name,
        num_train_trajectories=num_train_trajectories,
        splits=splits,
        target_channels=target_channels,
        pad_value=pad_value,
        split_strategy=split_strategy,
    )
    train_indices, val_indices, test_indices, split_info = get_train_val_test_trajectory_indices_from_file(
        str(target_dataset_path),
        variable_name=resolved_variable_name,
        num_train_trajectories=num_train_trajectories,
        splits=splits,
        split_strategy=split_strategy,
    )

    run_dir = tutorial_paths.runs_root / f"fine_tune_spus_36m_{_timestamp_label()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = append_run_log(
        run_dir,
        (
            f"Constructed in-memory train/val/test trajectory splits from assembled NetCDF file {target_dataset_path}: "
            f"variable={resolved_variable_name}, "
            f"shape={split_info['shape']}, split_strategy={split_strategy}, train_trajectories={len(train_indices)}, "
            f"val_trajectories={len(val_indices)}, test_trajectories={len(test_indices)}, "
            f"target_channels={target_channels}, pad_value={pad_value}"
        ),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    append_run_log(
        run_dir,
        (
            f"Created data loaders: train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
            f"batch_size={batch_size}, num_workers={num_workers}"
        ),
    )
    runtime_device = _resolve_device(torch, device)
    model_details = load_spus_36m_training_model(
        paths=tutorial_paths,
        training_repo_path=training_repo_path,
        pretrained_model_id=pretrained_model_id,
        pretrained_source=pretrained_source,
        model_repo_path=model_repo_path,
    )
    model = model_details['model'].to(runtime_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    append_run_log(
        run_dir,
        (
            f"Starting fine-tune: dataset={target_dataset_path}, pretrained_source={model_details['pretrained_source']}, "
            f"epochs={epochs}, batch_size={batch_size}, learning_rate={learning_rate}, "
            f"train_trajectories={len(train_indices)}, val_trajectories={len(val_indices)}, "
            f"test_trajectories={len(test_indices)}, device={runtime_device}"
        ),
    )

    history = []
    best_val_loss = float('inf')
    best_checkpoint = None

    try:
        for epoch in range(epochs):
            append_run_log(run_dir, f"Starting epoch {epoch + 1}/{epochs}")
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for x_current, x_next in train_loader:
                optimizer.zero_grad()
                x_current = x_current.to(runtime_device).unsqueeze(1)
                x_next = x_next.to(runtime_device)
                prediction = model(x_current)
                loss = F.mse_loss(prediction, x_next)
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item())
                train_batches += 1

            average_train_loss = train_loss_sum / max(train_batches, 1)
            model.eval()
            val_loss_sum = 0.0
            val_batches = 0
            with torch.no_grad():
                for x_current, x_next in val_loader:
                    x_current = x_current.to(runtime_device).unsqueeze(1)
                    x_next = x_next.to(runtime_device)
                    prediction = model(x_current)
                    val_loss = F.mse_loss(prediction, x_next)
                    val_loss_sum += float(val_loss.item())
                    val_batches += 1
            average_val_loss = val_loss_sum / max(val_batches, 1)
            history.append(
                {
                    'epoch': epoch + 1,
                    'train_loss': average_train_loss,
                    'val_loss': average_val_loss,
                }
            )
            append_run_log(
                run_dir,
                f"Completed epoch {epoch + 1}/{epochs}: train_loss={average_train_loss:.6f}, val_loss={average_val_loss:.6f}",
            )
            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                best_checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': copy.deepcopy(model.state_dict()),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }
                append_run_log(
                    run_dir,
                    f"New best checkpoint at epoch {epoch + 1}: val_loss={best_val_loss:.6f}",
                )
    finally:
        close_all_datasets(train_set)
        close_all_datasets(val_set)
        close_all_datasets(test_set)

    metadata = {
        'pretrained_model_id': pretrained_model_id,
        'dataset_path': str(target_dataset_path),
        'variable_name': resolved_variable_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'seed': seed,
        'device': str(runtime_device),
        'target_channels': target_channels,
        'pad_value': pad_value,
        'split_strategy': split_strategy,
        'num_train_trajectories': len(train_indices),
        'num_val_trajectories': len(val_indices),
        'num_test_trajectories': len(test_indices),
        'split_info': split_info,
        'history': history,
        'training_repo_path': str(model_details['training_repo_path']),
        'log_path': str(log_path),
        'dependency_status': dependency_status,
    }
    checkpoint = best_checkpoint or {
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    checkpoint_path = save_checkpoint(run_dir, checkpoint, metadata, filename='best_checkpoint.pt', torch_module=torch)
    append_run_log(run_dir, f"Finished fine-tune. Saved checkpoint to {checkpoint_path}")
    return {
        'run_dir': run_dir,
        'log_path': log_path,
        'checkpoint_path': checkpoint_path,
        'metadata': metadata,
        'best_val_loss': best_val_loss,
        'history': history,
    }


def infer_rollout(
    paths: Optional[TutorialPaths] = None,
    dataset_path: Optional[Path] = None,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = 1,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    split_strategy: str = 'fns_kf_paper',
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    checkpoint_path: Optional[Path] = None,
    training_repo_path: Optional[Path] = None,
    device: Optional[str] = None,
    max_test_trajectories: Optional[int] = None,
    test_trajectory_index: Optional[int] = None,
    split_name: str = 'test',
    target_channels: int = 5,
    pad_value: float = 1.0,
) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    dependency_status = require_spus_runtime_dependencies(("torch", "huggingface_hub", "safetensors", "netCDF4"))

    import torch

    from spus_ft.dataloader import (
        get_train_val_test_trajectory_indices_from_file,
        infer_variable_name,
        load_trajectories_by_indices,
    )

    target_dataset_path = resolve_fns_kf_nc_path(tutorial_paths, dataset_path)
    resolved_variable_name = infer_variable_name(str(target_dataset_path), variable_name)
    train_indices, val_indices, test_indices, split_info = get_train_val_test_trajectory_indices_from_file(
        str(target_dataset_path),
        variable_name=resolved_variable_name,
        num_train_trajectories=num_train_trajectories,
        splits=splits,
        split_strategy=split_strategy,
    )
    split_map = {
        'train': train_indices,
        'val': val_indices,
        'test': test_indices,
    }
    normalized_split_name = split_name.lower()
    if normalized_split_name not in split_map:
        raise ValueError("split_name must be one of: 'train', 'val', 'test'")

    inference_strategy = resolve_spus_inference_strategy(
        paths=tutorial_paths,
        pretrained_source=pretrained_source,
        model_repo_path=model_repo_path,
        training_repo_path=training_repo_path,
        checkpoint_path=checkpoint_path,
        target_channels=target_channels,
        pad_value=pad_value,
    )

    selected_indices, selected_split_trajectory_index = select_split_trajectory_indices(
        split_map[normalized_split_name],
        split_trajectory_index=test_trajectory_index,
        max_trajectories=max_test_trajectories,
    )

    trajectories, resolved_variable_name = load_trajectories_by_indices(
        str(target_dataset_path),
        selected_indices,
        variable_name=resolved_variable_name,
        target_channels=inference_strategy['target_channels'],
        pad_value=inference_strategy['pad_value'],
    )

    runtime_device = _resolve_device(torch, device)
    if inference_strategy['loader'] == 'pretrained':
        model_details = load_spus_36m_model(
            paths=tutorial_paths,
            pretrained_model_id=pretrained_model_id,
            pretrained_source=pretrained_source,
            model_repo_path=model_repo_path,
        )
    elif inference_strategy['loader'] == 'fns_kf':
        model_details = load_spus_fns_kf_model(
            paths=tutorial_paths,
            pretrained_model_id=pretrained_model_id,
            pretrained_source=pretrained_source,
            model_repo_path=model_repo_path,
            training_repo_path=training_repo_path,
        )
    else:
        model_details = load_spus_36m_training_model(
            paths=tutorial_paths,
            training_repo_path=training_repo_path,
            pretrained_model_id=pretrained_model_id,
            pretrained_source=pretrained_source,
            model_repo_path=model_repo_path,
        )
    model = model_details['model'].to(runtime_device)
    rollout_source = {
        'pretrained_model_id': pretrained_model_id,
        'pretrained_source': model_details['pretrained_source'],
        'checkpoint_path': None,
        'model_variant': inference_strategy['loader'],
        'model_class_name': inference_strategy['model_class_name'],
    }
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=runtime_device)
        model.load_state_dict(checkpoint['model_state_dict'])
        rollout_source['checkpoint_path'] = str(Path(checkpoint_path).resolve())
    model.eval()

    trajectories = trajectories.to(runtime_device)
    predictions = autoregressive_tensor_rollout(
        model=model,
        initial_condition=trajectories[:, 0],
        rollout_steps=trajectories.size(1) - 1,
        device=runtime_device,
        torch_module=torch,
    )
    targets = trajectories[:, 1:].detach().cpu()
    mode_label = 'finetuned' if inference_strategy['is_fine_tuned'] else 'pretrained'
    run_dir = tutorial_paths.runs_root / f"infer_rollout_spus_36m_{mode_label}_{_timestamp_label()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'dataset_path': str(target_dataset_path),
        'variable_name': resolved_variable_name,
        'split_name': normalized_split_name,
        'split_strategy': split_strategy,
        'num_train_trajectories': num_train_trajectories,
        'requested_test_trajectory_index': test_trajectory_index,
        'selected_test_trajectory_index': selected_split_trajectory_index,
        'test_trajectory_indices': list(selected_indices),
        'num_test_trajectories': len(selected_indices),
        'split_info': split_info,
        'rollout_source': rollout_source,
        'training_repo_path': str(model_details.get('training_repo_path')) if model_details.get('training_repo_path') is not None else None,
        'dependency_status': dependency_status,
        'device': str(runtime_device),
        'target_channels': inference_strategy['target_channels'],
        'pad_value': inference_strategy['pad_value'],
    }
    summary = save_torch_rollout(run_dir, predictions, targets, metadata, torch_module=torch)
    return {
        'run_dir': run_dir,
        'predictions_path': run_dir / 'rollout.pt',
        'summary_path': run_dir / 'summary.json',
        'summary': summary,
        'metadata': metadata,
    }


def evaluate_rollout(
    paths: Optional[TutorialPaths] = None,
    dataset_path: Optional[str | Path] = None,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = 1,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    split_strategy: str = 'fns_kf_paper',
    pretrained_model_id: str = DEFAULT_PRETRAINED_MODEL_ID,
    pretrained_source: Optional[str | Path] = None,
    model_repo_path: Optional[Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    training_repo_path: Optional[str | Path] = None,
    device: Optional[str] = None,
    num_trajectories: Optional[int] = None,
    split_trajectory_index: Optional[int] = None,
    split_name: str = 'test',
    target_channels: int = 5,
    pad_value: float = 1.0,
    preview_trajectory_count: int = 5,
    rollout_outcome: Optional[dict] = None,
    rollout_run_dir: Optional[str | Path] = None,
) -> dict:
    if rollout_outcome is not None and rollout_run_dir is not None:
        raise ValueError("Provide at most one of rollout_outcome or rollout_run_dir.")

    if rollout_outcome is None and rollout_run_dir is None:
        rollout_outcome = infer_rollout(
            paths=paths,
            dataset_path=dataset_path,
            variable_name=variable_name,
            num_train_trajectories=num_train_trajectories,
            splits=splits,
            split_strategy=split_strategy,
            pretrained_model_id=pretrained_model_id,
            pretrained_source=pretrained_source,
            model_repo_path=model_repo_path,
            checkpoint_path=checkpoint_path,
            training_repo_path=training_repo_path,
            device=device,
            max_test_trajectories=num_trajectories,
            test_trajectory_index=split_trajectory_index,
            split_name=split_name,
            target_channels=target_channels,
            pad_value=pad_value,
        )
    elif rollout_outcome is None:
        rollout_outcome = {
            'run_dir': _coerce_optional_path(rollout_run_dir),
            'predictions_path': _coerce_optional_path(rollout_run_dir) / 'rollout.pt',
            'summary_path': _coerce_optional_path(rollout_run_dir) / 'summary.json',
        }

    import torch

    rollout_payload = load_torch_rollout(rollout_outcome['run_dir'], torch_module=torch)
    evaluation_metadata = {
        **rollout_payload['metadata'],
        'evaluation_num_trajectories': len(rollout_payload['predictions']),
    }
    evaluation = save_evaluation(
        rollout_outcome['run_dir'],
        rollout_payload['predictions'],
        rollout_payload['targets'],
        evaluation_metadata,
    )
    preview_paths = save_preview_tensors(
        rollout_outcome['run_dir'],
        rollout_payload['predictions'],
        rollout_payload['targets'],
        preview_count=preview_trajectory_count,
        torch_module=torch,
    )
    append_run_log(
        rollout_outcome['run_dir'],
        (
            f"Evaluation complete: split={split_name}, num_trajectories={len(rollout_payload['predictions'])}, "
            f"average_mse={evaluation['average_mse']:.6f}"
        ),
    )
    return {
        **rollout_outcome,
        'evaluation_path': rollout_outcome['run_dir'] / 'evaluation.json',
        'evaluation': evaluation,
        **preview_paths,
    }


def run_synthetic_demo(config_path: Optional[Path] = None, paths: Optional[TutorialPaths] = None) -> dict:
    tutorial_paths = paths or TutorialPaths.discover()
    tutorial_paths.ensure_runtime_dirs()
    config = load_demo_config(config_path or tutorial_paths.repo_root / "configs" / "synthetic_demo.json")

    dataset = generate_synthetic_dataset(
        num_samples=config["num_samples"],
        num_timesteps=config["num_timesteps"],
        num_features=config["num_features"],
        decay=config["decay"],
        forcing=config["forcing"],
        seed=config["seed"],
    )
    model = make_baseline_model(alpha=config["baseline_alpha"])
    rollout = autoregressive_rollout(model, dataset.inputs, dataset.targets)
    run_dir = tutorial_paths.stage_run_dir("phase1_synthetic_demo")
    summary = save_rollout(run_dir, rollout.predictions, rollout.targets, dataset.metadata)
    return {
        "paths": tutorial_paths,
        "config": config,
        "run_dir": run_dir,
        "summary": summary,
    }
