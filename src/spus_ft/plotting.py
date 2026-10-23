from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence

from spus_ft.results import load_evaluation, load_torch_rollout


def _trajectory_grid_shape(frame_count: int, max_columns: int) -> tuple[int, int]:
    columns = min(max_columns, frame_count)
    rows = (frame_count + columns - 1) // columns
    return rows, columns


def _resolve_channel_label(channel_index: int, channel_names: Optional[Sequence[str]] = None) -> str:
    if channel_names is not None and 0 <= channel_index < len(channel_names):
        return f'Channel {channel_index} ({channel_names[channel_index]})'
    return f'Channel {channel_index}'


def _import_pyplot():
    import matplotlib.pyplot as pyplot

    return pyplot


def _to_nested_lists(value: Any) -> Any:
    converted = value
    if hasattr(converted, 'detach'):
        converted = converted.detach()
    if hasattr(converted, 'cpu'):
        converted = converted.cpu()
    if hasattr(converted, 'tolist'):
        return converted.tolist()
    return converted


def _load_rollout_payload(
    run_dir: Optional[Path | str] = None,
    payload: Optional[dict] = None,
    torch_module: Optional[Any] = None,
) -> dict:
    if payload is not None:
        return payload
    if run_dir is None:
        raise ValueError('Provide either run_dir or payload.')
    return load_torch_rollout(Path(run_dir), torch_module=torch_module)


def _load_evaluation_payload(source: Path | str | dict) -> dict:
    if isinstance(source, dict):
        return source

    path = Path(source)
    if path.is_dir():
        return load_evaluation(path)

    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _collect_scalar_values(node: Any) -> list[float]:
    if isinstance(node, list):
        values: list[float] = []
        for item in node:
            values.extend(_collect_scalar_values(item))
        return values
    return [float(node)]


def _subtract_nested(prediction: Any, target: Any) -> Any:
    if isinstance(prediction, list):
        return [_subtract_nested(pred_item, target_item) for pred_item, target_item in zip(prediction, target)]
    return float(prediction) - float(target)


def _resolve_rollout_steps(timesteps: Sequence[int], available_steps: int) -> list[tuple[int, int]]:
    resolved: list[tuple[int, int]] = []
    for step in timesteps:
        if step < 1 or step > available_steps:
            raise IndexError(f'Requested rollout step {step}, but only {available_steps} steps are available.')
        resolved.append((step, step - 1))
    return resolved


def build_evaluation_label(metadata: Optional[dict], fallback: Optional[str] = None) -> str:
    details = metadata or {}
    rollout_source = details.get('rollout_source', {})
    checkpoint_path = rollout_source.get('checkpoint_path')
    model_variant = rollout_source.get('model_variant')
    num_train_trajectories = details.get('num_train_trajectories')

    label = 'fine-tuned' if checkpoint_path or model_variant == 'fns_kf' else 'pretrained'
    if num_train_trajectories is not None:
        label = f'{label} ({num_train_trajectories} train trajectories)'
    if details.get('split_name'):
        label = f"{label} [{details['split_name']}]"
    return label or (fallback or 'run')


def plot_rollout_frames(
    run_dir: Optional[Path | str] = None,
    payload: Optional[dict] = None,
    trajectory_index: int = 0,
    timesteps: Sequence[int] = (1, 8, 16),
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    torch_module: Optional[Any] = None,
) -> dict:
    pyplot = _import_pyplot()
    rollout_payload = _load_rollout_payload(run_dir=run_dir, payload=payload, torch_module=torch_module)
    predictions = _to_nested_lists(rollout_payload['predictions'])
    targets = _to_nested_lists(rollout_payload['targets'])
    metadata = rollout_payload.get('metadata', {})

    if trajectory_index < 0 or trajectory_index >= len(predictions):
        raise IndexError(f'trajectory_index {trajectory_index} is out of range for {len(predictions)} trajectories.')

    available_steps = len(predictions[trajectory_index])
    resolved_steps = _resolve_rollout_steps(timesteps, available_steps)

    frame_values: list[float] = []
    difference_values: list[float] = []
    for _, step_index in resolved_steps:
        prediction_frame = predictions[trajectory_index][step_index][channel_index]
        target_frame = targets[trajectory_index][step_index][channel_index]
        difference_frame = _subtract_nested(prediction_frame, target_frame)

        frame_values.extend(_collect_scalar_values(prediction_frame))
        frame_values.extend(_collect_scalar_values(target_frame))
        difference_values.extend(_collect_scalar_values(difference_frame))
    vmin = min(frame_values)
    vmax = max(frame_values)
    difference_abs_max = max(abs(min(difference_values)), abs(max(difference_values)))

    figure, axes = pyplot.subplots(3, len(resolved_steps), figsize=(4 * len(resolved_steps), 10), squeeze=False)
    mappable = None
    difference_mappable = None
    for column_index, (step_label, step_index) in enumerate(resolved_steps):
        prediction_frame = predictions[trajectory_index][step_index][channel_index]
        target_frame = targets[trajectory_index][step_index][channel_index]
        difference_frame = _subtract_nested(prediction_frame, target_frame)

        mappable = axes[0][column_index].imshow(prediction_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0][column_index].set_title(f'Predicted step {step_label}')
        axes[0][column_index].set_axis_off()

        axes[1][column_index].imshow(target_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1][column_index].set_title(f'True step {step_label}')
        axes[1][column_index].set_axis_off()

        difference_mappable = axes[2][column_index].imshow(
            difference_frame,
            cmap='coolwarm',
            vmin=-difference_abs_max,
            vmax=difference_abs_max,
        )
        axes[2][column_index].set_title(f'Difference step {step_label}')
        axes[2][column_index].set_axis_off()

    figure.suptitle(title or 'SPUS rollout frames', fontsize=14)
    figure.tight_layout(rect=(0.0, 0.30, 1.0, 0.93))
    subtitle = figure.text(
        0.5,
        0.24,
        'Top row: predicted rollout. Middle row: ground truth. Bottom row: prediction minus truth.',
        ha='center',
    )

    label = build_evaluation_label(metadata)
    channel_label = _resolve_channel_label(channel_index, channel_names)
    colorbar_axis = figure.add_axes([0.15, 0.12, 0.70, 0.025])
    colorbar = figure.colorbar(
        mappable,
        cax=colorbar_axis,
        orientation='horizontal',
    )
    colorbar.set_label(f'{channel_label} value — {label}')
    difference_colorbar_axis = figure.add_axes([0.15, 0.03, 0.70, 0.025])
    difference_colorbar = figure.colorbar(
        difference_mappable,
        cax=difference_colorbar_axis,
        orientation='horizontal',
    )
    difference_colorbar.set_label(f'{channel_label} difference (predicted − true) — {label}')

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'axes': axes,
        'subtitle': subtitle,
        'colorbar': colorbar,
        'difference_colorbar': difference_colorbar,
        'metadata': metadata,
        'channel_label': channel_label,
        'timesteps': [step for step, _ in resolved_steps],
        'difference_range': {'min': -difference_abs_max, 'max': difference_abs_max},
    }


def plot_rollout_frames_separate_scales(
    run_dir: Optional[Path | str] = None,
    payload: Optional[dict] = None,
    trajectory_index: int = 0,
    timesteps: Sequence[int] = (1, 8, 16),
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    torch_module: Optional[Any] = None,
) -> dict:
    pyplot = _import_pyplot()
    rollout_payload = _load_rollout_payload(run_dir=run_dir, payload=payload, torch_module=torch_module)
    predictions = _to_nested_lists(rollout_payload['predictions'])
    targets = _to_nested_lists(rollout_payload['targets'])
    metadata = rollout_payload.get('metadata', {})

    if trajectory_index < 0 or trajectory_index >= len(predictions):
        raise IndexError(f'trajectory_index {trajectory_index} is out of range for {len(predictions)} trajectories.')

    available_steps = len(predictions[trajectory_index])
    resolved_steps = _resolve_rollout_steps(timesteps, available_steps)

    prediction_values: list[float] = []
    target_values: list[float] = []
    for _, step_index in resolved_steps:
        prediction_values.extend(_collect_scalar_values(predictions[trajectory_index][step_index][channel_index]))
        target_values.extend(_collect_scalar_values(targets[trajectory_index][step_index][channel_index]))
    prediction_vmin = min(prediction_values)
    prediction_vmax = max(prediction_values)
    target_vmin = min(target_values)
    target_vmax = max(target_values)

    figure, axes = pyplot.subplots(2, len(resolved_steps), figsize=(4 * len(resolved_steps), 7.5), squeeze=False)
    prediction_mappable = None
    target_mappable = None
    for column_index, (step_label, step_index) in enumerate(resolved_steps):
        prediction_frame = predictions[trajectory_index][step_index][channel_index]
        target_frame = targets[trajectory_index][step_index][channel_index]

        prediction_mappable = axes[0][column_index].imshow(
            prediction_frame,
            cmap=cmap,
            vmin=prediction_vmin,
            vmax=prediction_vmax,
        )
        axes[0][column_index].set_title(f'Predicted step {step_label}')
        axes[0][column_index].set_axis_off()

        target_mappable = axes[1][column_index].imshow(
            target_frame,
            cmap=cmap,
            vmin=target_vmin,
            vmax=target_vmax,
        )
        axes[1][column_index].set_title(f'True step {step_label}')
        axes[1][column_index].set_axis_off()

    figure.suptitle(title or 'SPUS rollout frames (separate color scales)', fontsize=14)
    subtitle = figure.text(
        0.5,
        0.16,
        'Top row uses prediction range. Bottom row uses ground-truth range.',
        ha='center',
    )

    label = build_evaluation_label(metadata)
    channel_label = _resolve_channel_label(channel_index, channel_names)
    prediction_colorbar = figure.colorbar(
        prediction_mappable,
        ax=axes[0].tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.16,
    )
    prediction_colorbar.set_label(f'{channel_label} predicted value — {label}')
    target_colorbar = figure.colorbar(
        target_mappable,
        ax=axes[1].tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.22,
    )
    target_colorbar.set_label(f'{channel_label} true value — {label}')
    figure.tight_layout(rect=(0.0, 0.20, 1.0, 0.93))

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'axes': axes,
        'subtitle': subtitle,
        'prediction_colorbar': prediction_colorbar,
        'target_colorbar': target_colorbar,
        'metadata': metadata,
        'channel_label': channel_label,
        'timesteps': [step for step, _ in resolved_steps],
        'prediction_range': {'min': prediction_vmin, 'max': prediction_vmax},
        'target_range': {'min': target_vmin, 'max': target_vmax},
    }


def plot_full_rollout_frames(
    run_dir: Optional[Path | str] = None,
    payload: Optional[dict] = None,
    trajectory_index: int = 0,
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    torch_module: Optional[Any] = None,
    max_columns: int = 5,
) -> dict:
    pyplot = _import_pyplot()
    rollout_payload = _load_rollout_payload(run_dir=run_dir, payload=payload, torch_module=torch_module)
    predictions = _to_nested_lists(rollout_payload['predictions'])
    targets = _to_nested_lists(rollout_payload['targets'])
    metadata = rollout_payload.get('metadata', {})

    if trajectory_index < 0 or trajectory_index >= len(predictions):
        raise IndexError(f'trajectory_index {trajectory_index} is out of range for {len(predictions)} trajectories.')

    available_steps = len(predictions[trajectory_index])
    timesteps = tuple(range(1, available_steps + 1))
    resolved_steps = _resolve_rollout_steps(timesteps, available_steps)

    frame_values: list[float] = []
    for _, step_index in resolved_steps:
        frame_values.extend(_collect_scalar_values(predictions[trajectory_index][step_index][channel_index]))
        frame_values.extend(_collect_scalar_values(targets[trajectory_index][step_index][channel_index]))
    vmin = min(frame_values)
    vmax = max(frame_values)

    columns = min(max_columns, len(resolved_steps))
    row_groups = (len(resolved_steps) + columns - 1) // columns
    total_rows = row_groups * 2
    figure, axes = pyplot.subplots(total_rows, columns, figsize=(4 * columns, 3.4 * total_rows), squeeze=False)
    for row in axes:
        for axis in row:
            axis.set_axis_off()

    mappable = None
    for frame_index, (step_label, step_index) in enumerate(resolved_steps):
        column_index = frame_index % columns
        group_index = frame_index // columns
        prediction_row = group_index * 2
        target_row = prediction_row + 1

        prediction_frame = predictions[trajectory_index][step_index][channel_index]
        target_frame = targets[trajectory_index][step_index][channel_index]

        mappable = axes[prediction_row][column_index].imshow(prediction_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[prediction_row][column_index].set_title(f'Predicted step {step_label}')
        axes[prediction_row][column_index].set_axis_off()

        axes[target_row][column_index].imshow(target_frame, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[target_row][column_index].set_title(f'True step {step_label}')
        axes[target_row][column_index].set_axis_off()

    figure.suptitle(title or 'SPUS full rollout frames', fontsize=14)
    subtitle = figure.text(
        0.5,
        0.10,
        'Each pair of rows shows predicted frames above true frames for all rollout steps.',
        ha='center',
    )

    label = build_evaluation_label(metadata)
    channel_label = _resolve_channel_label(channel_index, channel_names)
    colorbar = figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.03,
        pad=0.10,
    )
    colorbar.set_label(f'{channel_label} value — {label}')
    figure.tight_layout(rect=(0.0, 0.13, 1.0, 0.95))

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'axes': axes,
        'subtitle': subtitle,
        'colorbar': colorbar,
        'metadata': metadata,
        'channel_label': channel_label,
        'timesteps': [step for step, _ in resolved_steps],
        'trajectory_index': trajectory_index,
        'channel_index': channel_index,
    }


def plot_ground_truth_trajectory_frames(
    dataset_path: Path | str,
    split_name: str = 'test',
    split_trajectory_index: int = 0,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = 1,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    split_strategy: str = 'fns_kf_paper',
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    target_channels: int = 5,
    pad_value: float = 1.0,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    max_columns: int = 5,
) -> dict:
    pyplot = _import_pyplot()

    from spus_ft.dataloader import (
        get_train_val_test_trajectory_indices_from_file,
        infer_variable_name,
        load_trajectories_by_indices,
    )

    target_dataset_path = Path(dataset_path).expanduser().resolve()
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

    available_indices = split_map[normalized_split_name]
    if split_trajectory_index < 0 or split_trajectory_index >= len(available_indices):
        raise IndexError(
            f'split_trajectory_index {split_trajectory_index} is out of range for {len(available_indices)} trajectories in split {normalized_split_name}.'
        )

    selected_trajectory_index = available_indices[split_trajectory_index]
    trajectories, resolved_variable_name = load_trajectories_by_indices(
        str(target_dataset_path),
        [selected_trajectory_index],
        variable_name=resolved_variable_name,
        target_channels=target_channels,
        pad_value=pad_value,
    )

    trajectory = _to_nested_lists(trajectories[0])
    frame_values: list[float] = []
    for frame in trajectory:
        frame_values.extend(_collect_scalar_values(frame[channel_index]))
    vmin = min(frame_values)
    vmax = max(frame_values)

    rows, columns = _trajectory_grid_shape(len(trajectory), max_columns)
    figure, axes = pyplot.subplots(rows, columns, figsize=(4 * columns, 3.4 * rows), squeeze=False)
    for row_axes in axes:
        for axis in row_axes:
            axis.set_axis_off()

    mappable = None
    for frame_index, frame in enumerate(trajectory):
        row_index = frame_index // columns
        column_index = frame_index % columns
        mappable = axes[row_index][column_index].imshow(frame[channel_index], cmap=cmap, vmin=vmin, vmax=vmax)
        axes[row_index][column_index].set_title(f'Time {frame_index}')
        axes[row_index][column_index].set_axis_off()

    channel_label = _resolve_channel_label(channel_index, channel_names)
    title_lines = [title or f'Ground-truth trajectory frames ({normalized_split_name} split)']
    title_lines.append(
        f'Split trajectory index {split_trajectory_index} (global trajectory {selected_trajectory_index}) for {channel_label}'
    )
    figure.suptitle('\n'.join(title_lines), fontsize=14)

    colorbar = figure.colorbar(
        mappable,
        ax=axes.ravel().tolist(),
        orientation='horizontal',
        fraction=0.03,
        pad=0.18,
    )
    colorbar.set_label(f'Ground-truth {channel_label} value')
    figure.tight_layout(rect=(0.0, 0.12, 1.0, 0.88))

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'axes': axes,
        'subtitle': None,
        'colorbar': colorbar,
        'dataset_path': target_dataset_path,
        'variable_name': resolved_variable_name,
        'split_name': normalized_split_name,
        'split_info': split_info,
        'split_trajectory_index': split_trajectory_index,
        'selected_trajectory_index': selected_trajectory_index,
        'channel_label': channel_label,
        'channel_index': channel_index,
        'num_frames': len(trajectory),
    }


def plot_rollout_mse_curve(
    evaluation_source: Path | str | dict,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    title: str = 'Average MSE by rollout step',
    linewidth: float = 2.0,
    save_path: Optional[Path | str] = None,
) -> dict:
    pyplot = _import_pyplot()
    created_figure = ax is None
    if created_figure:
        figure, ax = pyplot.subplots(figsize=(8, 4.5))
    else:
        figure = ax.figure

    evaluation = _load_evaluation_payload(evaluation_source)
    timestep_mse = evaluation['timestep_mse']
    metadata = evaluation.get('metadata', {})
    resolved_label = label or build_evaluation_label(metadata)
    steps = list(range(1, len(timestep_mse) + 1))

    ax.plot(steps, timestep_mse, marker='o', linewidth=linewidth, label=resolved_label)
    ax.set_xlabel('Rollout step')
    ax.set_ylabel('Average MSE')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if resolved_label:
        ax.legend()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'ax': ax,
        'evaluation': evaluation,
    }


def plot_multiple_rollout_mse_curves(
    evaluation_sources: Sequence[Path | str | dict],
    labels: Optional[Sequence[str]] = None,
    title: str = 'Average MSE by rollout step for FNS-KF',
    linewidth: float = 2.0,
    save_path: Optional[Path | str] = None,
) -> dict:
    if labels is not None and len(labels) != len(evaluation_sources):
        raise ValueError('labels must match evaluation_sources in length')

    pyplot = _import_pyplot()
    figure, ax = pyplot.subplots(figsize=(8.5, 5.0))

    plotted_evaluations = []
    for index, source in enumerate(evaluation_sources):
        evaluation = _load_evaluation_payload(source)
        resolved_label = labels[index] if labels is not None else None
        plot_rollout_mse_curve(
            evaluation,
            label=resolved_label,
            ax=ax,
            title=title,
            linewidth=linewidth,
        )
        plotted_evaluations.append(evaluation)

    ax.legend()
    figure.tight_layout()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'ax': ax,
        'evaluations': plotted_evaluations,
    }
