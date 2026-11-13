from __future__ import annotations

import json
import textwrap
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


def _prepare_image_frame(frame: Any) -> Any:
    if hasattr(frame, 'detach'):
        frame = frame.detach()
    if hasattr(frame, 'cpu'):
        frame = frame.cpu()
    if hasattr(frame, 'tolist'):
        frame = frame.tolist()
    if not isinstance(frame, list) or not frame:
        return frame
    return [list(row) for row in zip(*frame)]


def _set_wrapped_figure_title(title_text: Any, base_title: str, step_label: str, width: int = 64) -> None:
    wrapped_base_title = textwrap.fill(base_title, width=width)
    title_text.set_text(f'{wrapped_base_title}\n{step_label}')


def _resolve_split_trajectory_indices(
    available_indices: Sequence[int],
    split_trajectory_indices: Sequence[int],
    split_name: str,
) -> list[int]:
    resolved_indices: list[int] = []
    for split_trajectory_index in split_trajectory_indices:
        if split_trajectory_index < 0 or split_trajectory_index >= len(available_indices):
            raise IndexError(
                f'split_trajectory_index {split_trajectory_index} is out of range for '
                f'{len(available_indices)} trajectories in split {split_name}.'
            )
        resolved_indices.append(available_indices[split_trajectory_index])
    return resolved_indices


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

        mappable = axes[0][column_index].imshow(_prepare_image_frame(prediction_frame), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0][column_index].set_title(f'Predicted step {step_label}')
        axes[0][column_index].set_axis_off()

        axes[1][column_index].imshow(_prepare_image_frame(target_frame), cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1][column_index].set_title(f'True step {step_label}')
        axes[1][column_index].set_axis_off()

        difference_mappable = axes[2][column_index].imshow(
            _prepare_image_frame(difference_frame),
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

    figure, axes = pyplot.subplots(
        2,
        len(resolved_steps),
        figsize=(4 * len(resolved_steps), 7.8),
        squeeze=False,
        gridspec_kw={'hspace': 0.28},
    )
    prediction_mappable = None
    target_mappable = None
    for column_index, (step_label, step_index) in enumerate(resolved_steps):
        prediction_frame = predictions[trajectory_index][step_index][channel_index]
        target_frame = targets[trajectory_index][step_index][channel_index]

        prediction_mappable = axes[0][column_index].imshow(
            _prepare_image_frame(prediction_frame),
            cmap=cmap,
            vmin=prediction_vmin,
            vmax=prediction_vmax,
        )
        axes[0][column_index].set_title(f'Predicted step {step_label}')
        axes[0][column_index].set_axis_off()

        target_mappable = axes[1][column_index].imshow(
            _prepare_image_frame(target_frame),
            cmap=cmap,
            vmin=target_vmin,
            vmax=target_vmax,
        )
        axes[1][column_index].set_title(f'True step {step_label}')
        axes[1][column_index].set_axis_off()

    figure.suptitle(title or 'SPUS rollout frames (separate color scales)', fontsize=14)
    subtitle = figure.text(
        0.5,
        0.15,
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
        pad=0.20,
    )
    prediction_colorbar.set_label(f'{channel_label} predicted value — {label}')
    target_colorbar = figure.colorbar(
        target_mappable,
        ax=axes[1].tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.20,
    )
    target_colorbar.set_label(f'{channel_label} true value — {label}')
    figure.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.88,
        bottom=0.20,
        wspace=0.08,
        hspace=0.42,
    )

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

        mappable = axes[prediction_row][column_index].imshow(
            _prepare_image_frame(prediction_frame),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[prediction_row][column_index].set_title(f'Predicted step {step_label}')
        axes[prediction_row][column_index].set_axis_off()

        axes[target_row][column_index].imshow(_prepare_image_frame(target_frame), cmap=cmap, vmin=vmin, vmax=vmax)
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
    selected_trajectory_index = _resolve_split_trajectory_indices(
        available_indices,
        [split_trajectory_index],
        normalized_split_name,
    )[0]
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
        mappable = axes[row_index][column_index].imshow(
            _prepare_image_frame(frame[channel_index]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
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
    figure.subplots_adjust(
        left=0.04,
        right=0.98,
        top=0.84,
        bottom=0.14,
        wspace=0.08,
        hspace=0.32,
    )

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


def animate_ground_truth_trajectories_row(
    dataset_path: Path | str,
    split_trajectory_indices: Sequence[int],
    split_name: str = 'test',
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
    interval: int = 250,
    repeat: bool = True,
) -> dict:
    if not split_trajectory_indices:
        raise ValueError('split_trajectory_indices must be a non-empty sequence.')

    pyplot = _import_pyplot()
    from matplotlib.animation import FuncAnimation

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
    selected_trajectory_indices = _resolve_split_trajectory_indices(
        available_indices,
        list(split_trajectory_indices),
        normalized_split_name,
    )
    trajectories, resolved_variable_name = load_trajectories_by_indices(
        str(target_dataset_path),
        selected_trajectory_indices,
        variable_name=resolved_variable_name,
        target_channels=target_channels,
        pad_value=pad_value,
    )

    trajectories_list = _to_nested_lists(trajectories)
    first_trajectory = trajectories_list[0]
    num_frames = len(first_trajectory)
    for trajectory in trajectories_list:
        if len(trajectory) != num_frames:
            raise ValueError('All selected trajectories must have the same number of frames.')

    frame_values: list[float] = []
    for trajectory in trajectories_list:
        for frame in trajectory:
            frame_values.extend(_collect_scalar_values(frame[channel_index]))
    vmin = min(frame_values)
    vmax = max(frame_values)

    columns = len(trajectories_list)
    figure, axes = pyplot.subplots(1, columns, figsize=(4 * columns, 4.8), squeeze=False)
    axes_row = axes[0]
    image_artists = []
    channel_label = _resolve_channel_label(channel_index, channel_names)

    for axis, trajectory, split_index, selected_index in zip(
        axes_row,
        trajectories_list,
        split_trajectory_indices,
        selected_trajectory_indices,
    ):
        image = axis.imshow(
            _prepare_image_frame(trajectory[0][channel_index]),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        axis.set_title(f'Trajectory {split_index}\n(global {selected_index})')
        axis.set_axis_off()
        image_artists.append(image)

    title_text = figure.suptitle('', fontsize=14, y=0.97)
    label = f'Ground-truth {channel_label}'
    subtitle = figure.text(0.5, 0.08, label, ha='center')
    colorbar = figure.colorbar(
        image_artists[0],
        ax=axes_row.tolist(),
        orientation='horizontal',
        fraction=0.05,
        pad=0.12,
    )
    colorbar.set_label(f'{channel_label} value')
    figure.subplots_adjust(left=0.04, right=0.98, top=0.78, bottom=0.18, wspace=0.08)

    base_title = title or f'Ground-truth trajectories ({normalized_split_name} split)'

    def _update(frame_index: int):
        for image_artist, trajectory in zip(image_artists, trajectories_list):
            image_artist.set_array(_prepare_image_frame(trajectory[frame_index][channel_index]))
        _set_wrapped_figure_title(
            title_text,
            base_title,
            f'Timestep {frame_index + 1} / {num_frames}',
        )
        return [title_text, *image_artists]

    animation = FuncAnimation(
        figure,
        _update,
        frames=num_frames,
        interval=interval,
        repeat=repeat,
        blit=False,
    )
    _update(0)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path)

    return {
        'figure': figure,
        'axes': axes,
        'animation': animation,
        'title': title_text,
        'subtitle': subtitle,
        'colorbar': colorbar,
        'dataset_path': target_dataset_path,
        'variable_name': resolved_variable_name,
        'split_name': normalized_split_name,
        'split_info': split_info,
        'split_trajectory_indices': list(split_trajectory_indices),
        'selected_trajectory_indices': selected_trajectory_indices,
        'channel_label': channel_label,
        'channel_index': channel_index,
        'num_frames': num_frames,
        'interval': interval,
    }


def animate_rollout_trajectories_comparison(
    run_dir: Optional[Path | str] = None,
    payload: Optional[dict] = None,
    trajectory_indices: Sequence[int] = (0,),
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    interval: int = 250,
    repeat: bool = True,
    torch_module: Optional[Any] = None,
) -> dict:
    if not trajectory_indices:
        raise ValueError('trajectory_indices must be a non-empty sequence.')

    pyplot = _import_pyplot()
    from matplotlib.animation import FuncAnimation

    rollout_payload = _load_rollout_payload(run_dir=run_dir, payload=payload, torch_module=torch_module)
    predictions = _to_nested_lists(rollout_payload['predictions'])
    targets = _to_nested_lists(rollout_payload['targets'])
    metadata = rollout_payload.get('metadata', {})

    selected_trajectories = list(trajectory_indices)
    for trajectory_index in selected_trajectories:
        if trajectory_index < 0 or trajectory_index >= len(predictions):
            raise IndexError(f'trajectory_index {trajectory_index} is out of range for {len(predictions)} trajectories.')

    num_frames = len(predictions[selected_trajectories[0]])
    for trajectory_index in selected_trajectories:
        if len(predictions[trajectory_index]) != num_frames or len(targets[trajectory_index]) != num_frames:
            raise ValueError('All selected trajectories must have the same number of rollout frames.')

    frame_values: list[float] = []
    difference_values: list[float] = []
    for trajectory_index in selected_trajectories:
        for frame_index in range(num_frames):
            prediction_frame = predictions[trajectory_index][frame_index][channel_index]
            target_frame = targets[trajectory_index][frame_index][channel_index]
            difference_frame = _subtract_nested(prediction_frame, target_frame)
            frame_values.extend(_collect_scalar_values(prediction_frame))
            frame_values.extend(_collect_scalar_values(target_frame))
            difference_values.extend(_collect_scalar_values(difference_frame))
    vmin = min(frame_values)
    vmax = max(frame_values)
    difference_abs_max = max(abs(min(difference_values)), abs(max(difference_values)))

    columns = len(selected_trajectories)
    figure, axes = pyplot.subplots(3, columns, figsize=(4 * columns, 10.0), squeeze=False)
    prediction_artists = []
    target_artists = []
    difference_artists = []
    channel_label = _resolve_channel_label(channel_index, channel_names)

    for column_index, trajectory_index in enumerate(selected_trajectories):
        prediction_frame = predictions[trajectory_index][0][channel_index]
        target_frame = targets[trajectory_index][0][channel_index]
        difference_frame = _subtract_nested(prediction_frame, target_frame)

        prediction_artist = axes[0][column_index].imshow(
            _prepare_image_frame(prediction_frame),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        axes[0][column_index].set_title(f'Trajectory {trajectory_index}')
        axes[0][column_index].set_axis_off()
        prediction_artists.append(prediction_artist)

        target_artist = axes[1][column_index].imshow(
            _prepare_image_frame(target_frame),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        axes[1][column_index].set_axis_off()
        target_artists.append(target_artist)

        difference_artist = axes[2][column_index].imshow(
            _prepare_image_frame(difference_frame),
            cmap='coolwarm',
            vmin=-difference_abs_max,
            vmax=difference_abs_max,
            animated=True,
        )
        axes[2][column_index].set_axis_off()
        difference_artists.append(difference_artist)

    figure.text(0.02, 0.77, 'Predicted', rotation=90, va='center', ha='center')
    figure.text(0.02, 0.50, 'Ground truth', rotation=90, va='center', ha='center')
    figure.text(0.02, 0.23, 'Difference', rotation=90, va='center', ha='center')
    title_text = figure.suptitle('', fontsize=14, y=0.97)
    subtitle = figure.text(
        0.5,
        0.14,
        'Top row: predicted. Middle row: ground truth. Bottom row: predicted minus true.',
        ha='center',
    )
    label = build_evaluation_label(metadata)
    colorbar_axis = figure.add_axes([0.16, 0.08, 0.68, 0.022])
    colorbar = figure.colorbar(prediction_artists[0], cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label(f'{channel_label} value — {label}')
    difference_colorbar_axis = figure.add_axes([0.16, 0.03, 0.68, 0.022])
    difference_colorbar = figure.colorbar(
        difference_artists[0],
        cax=difference_colorbar_axis,
        orientation='horizontal',
    )
    difference_colorbar.set_label(f'{channel_label} difference (predicted − true) — {label}')
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.08, hspace=0.18)

    base_title = title or 'SPUS rollout trajectory comparison'

    def _update(frame_index: int):
        for column_index, trajectory_index in enumerate(selected_trajectories):
            prediction_frame = predictions[trajectory_index][frame_index][channel_index]
            target_frame = targets[trajectory_index][frame_index][channel_index]
            difference_frame = _subtract_nested(prediction_frame, target_frame)
            prediction_artists[column_index].set_array(_prepare_image_frame(prediction_frame))
            target_artists[column_index].set_array(_prepare_image_frame(target_frame))
            difference_artists[column_index].set_array(_prepare_image_frame(difference_frame))
        _set_wrapped_figure_title(
            title_text,
            base_title,
            f'Timestep {frame_index + 1} / {num_frames}',
        )
        return [title_text, *prediction_artists, *target_artists, *difference_artists]

    animation = FuncAnimation(
        figure,
        _update,
        frames=num_frames,
        interval=interval,
        repeat=repeat,
        blit=False,
    )
    _update(0)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path)

    return {
        'figure': figure,
        'axes': axes,
        'animation': animation,
        'title': title_text,
        'subtitle': subtitle,
        'colorbar': colorbar,
        'difference_colorbar': difference_colorbar,
        'metadata': metadata,
        'trajectory_indices': selected_trajectories,
        'channel_label': channel_label,
        'channel_index': channel_index,
        'num_frames': num_frames,
        'interval': interval,
        'difference_range': {'min': -difference_abs_max, 'max': difference_abs_max},
    }


def animate_multi_model_rollout_comparison(
    run_dirs: Optional[Sequence[Path | str]] = None,
    payloads: Optional[Sequence[dict]] = None,
    trajectory_index: int = 0,
    channel_index: int = 0,
    channel_names: Optional[Sequence[str]] = None,
    model_labels: Optional[Sequence[str]] = None,
    cmap: str = 'viridis',
    title: Optional[str] = None,
    save_path: Optional[Path | str] = None,
    interval: int = 250,
    repeat: bool = True,
    use_target_scale: bool = True,
    torch_module: Optional[Any] = None,
) -> dict:
    if run_dirs is None and payloads is None:
        raise ValueError('Provide run_dirs or payloads.')
    if run_dirs is not None and payloads is not None:
        raise ValueError('Provide only one of run_dirs or payloads, not both.')

    sources = list(payloads if payloads is not None else run_dirs or [])
    if not sources:
        raise ValueError('At least one rollout source is required.')
    if model_labels is not None and len(model_labels) != len(sources):
        raise ValueError('model_labels must match the number of rollout sources.')

    pyplot = _import_pyplot()
    from matplotlib.animation import FuncAnimation

    rollout_payloads = []
    for source in sources:
        if payloads is not None:
            rollout_payloads.append(source)
        else:
            rollout_payloads.append(_load_rollout_payload(run_dir=source, torch_module=torch_module))

    first_payload = rollout_payloads[0]
    first_predictions = _to_nested_lists(first_payload['predictions'])
    if trajectory_index < 0 or trajectory_index >= len(first_predictions):
        raise IndexError(f'trajectory_index {trajectory_index} is out of range for {len(first_predictions)} trajectories.')

    num_frames = len(first_predictions[trajectory_index])
    reference_targets = _to_nested_lists(first_payload['targets'])[trajectory_index]

    prepared_payloads = []
    prediction_values: list[float] = []
    target_values: list[float] = []
    difference_values: list[float] = []
    for model_index, rollout_payload in enumerate(rollout_payloads):
        predictions = _to_nested_lists(rollout_payload['predictions'])
        targets = _to_nested_lists(rollout_payload['targets'])
        metadata = rollout_payload.get('metadata', {})

        if trajectory_index < 0 or trajectory_index >= len(predictions):
            raise IndexError(
                f'trajectory_index {trajectory_index} is out of range for model source {model_index} '
                f'with {len(predictions)} trajectories.'
            )
        if len(predictions[trajectory_index]) != num_frames or len(targets[trajectory_index]) != num_frames:
            raise ValueError('All selected model rollouts must have the same number of frames.')

        model_predictions = predictions[trajectory_index]
        model_targets = targets[trajectory_index]
        for frame_index in range(num_frames):
            prediction_frame = model_predictions[frame_index][channel_index]
            target_frame = model_targets[frame_index][channel_index]
            difference_frame = _subtract_nested(prediction_frame, target_frame)
            prediction_values.extend(_collect_scalar_values(prediction_frame))
            target_values.extend(_collect_scalar_values(target_frame))
            difference_values.extend(_collect_scalar_values(difference_frame))

        prepared_payloads.append(
            {
                'predictions': model_predictions,
                'targets': model_targets,
                'metadata': metadata,
                'label': model_labels[model_index] if model_labels is not None else build_evaluation_label(metadata),
            }
        )

    if use_target_scale:
        vmin = min(target_values)
        vmax = max(target_values)
    else:
        vmin = min(prediction_values + target_values)
        vmax = max(prediction_values + target_values)
    difference_abs_max = max(abs(min(difference_values)), abs(max(difference_values)))

    columns = len(prepared_payloads)
    figure, axes = pyplot.subplots(3, columns, figsize=(4 * columns, 10.0), squeeze=False)
    prediction_artists = []
    target_artists = []
    difference_artists = []
    channel_label = _resolve_channel_label(channel_index, channel_names)

    for column_index, prepared in enumerate(prepared_payloads):
        prediction_frame = prepared['predictions'][0][channel_index]
        target_frame = prepared['targets'][0][channel_index]
        difference_frame = _subtract_nested(prediction_frame, target_frame)

        prediction_artist = axes[0][column_index].imshow(
            _prepare_image_frame(prediction_frame),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        axes[0][column_index].set_title(prepared['label'])
        axes[0][column_index].set_axis_off()
        prediction_artists.append(prediction_artist)

        target_artist = axes[1][column_index].imshow(
            _prepare_image_frame(target_frame),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            animated=True,
        )
        axes[1][column_index].set_axis_off()
        target_artists.append(target_artist)

        difference_artist = axes[2][column_index].imshow(
            _prepare_image_frame(difference_frame),
            cmap='coolwarm',
            vmin=-difference_abs_max,
            vmax=difference_abs_max,
            animated=True,
        )
        axes[2][column_index].set_axis_off()
        difference_artists.append(difference_artist)

    figure.text(0.02, 0.77, 'Predicted', rotation=90, va='center', ha='center')
    figure.text(0.02, 0.50, 'Ground truth', rotation=90, va='center', ha='center')
    figure.text(0.02, 0.23, 'Difference', rotation=90, va='center', ha='center')
    title_text = figure.suptitle('', fontsize=14, y=0.97)
    subtitle = figure.text(
        0.5,
        0.14,
        f'All columns show the same trajectory index {trajectory_index} across different models.',
        ha='center',
    )
    label = build_evaluation_label(first_payload.get('metadata', {}))
    colorbar_axis = figure.add_axes([0.16, 0.08, 0.68, 0.022])
    colorbar = figure.colorbar(prediction_artists[0], cax=colorbar_axis, orientation='horizontal')
    colorbar.set_label(f'{channel_label} value — {label}')
    difference_colorbar_axis = figure.add_axes([0.16, 0.03, 0.68, 0.022])
    difference_colorbar = figure.colorbar(difference_artists[0], cax=difference_colorbar_axis, orientation='horizontal')
    difference_colorbar.set_label(f'{channel_label} difference (predicted − true)')
    figure.subplots_adjust(left=0.07, right=0.98, top=0.80, bottom=0.20, wspace=0.12, hspace=0.18)

    base_title = title or 'SPUS multi-model rollout comparison'

    def _update(frame_index: int):
        for column_index, prepared in enumerate(prepared_payloads):
            prediction_frame = prepared['predictions'][frame_index][channel_index]
            target_frame = prepared['targets'][frame_index][channel_index]
            difference_frame = _subtract_nested(prediction_frame, target_frame)
            prediction_artists[column_index].set_array(_prepare_image_frame(prediction_frame))
            target_artists[column_index].set_array(_prepare_image_frame(target_frame))
            difference_artists[column_index].set_array(_prepare_image_frame(difference_frame))
        _set_wrapped_figure_title(title_text, base_title, f'Timestep {frame_index + 1} / {num_frames}')
        return [title_text, *prediction_artists, *target_artists, *difference_artists]

    animation = FuncAnimation(figure, _update, frames=num_frames, interval=interval, repeat=repeat, blit=False)
    _update(0)

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        animation.save(output_path)

    return {
        'figure': figure,
        'axes': axes,
        'animation': animation,
        'title': title_text,
        'subtitle': subtitle,
        'colorbar': colorbar,
        'difference_colorbar': difference_colorbar,
        'trajectory_index': trajectory_index,
        'channel_label': channel_label,
        'channel_index': channel_index,
        'num_frames': num_frames,
        'interval': interval,
        'model_labels': [prepared['label'] for prepared in prepared_payloads],
        'value_range': {'min': vmin, 'max': vmax},
        'difference_range': {'min': -difference_abs_max, 'max': difference_abs_max},
        'reference_target': reference_targets,
        'use_target_scale': use_target_scale,
    }


def plot_rollout_mse_curve(
    evaluation_source: Path | str | dict,
    label: Optional[str] = None,
    ax: Optional[Any] = None,
    title: str = 'Average MSE by rollout step',
    linewidth: float = 2.0,
    marker: str = 'o',
    linestyle: str = '-',
    markevery: Optional[Any] = None,
    alpha: float = 1.0,
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

    ax.plot(
        steps,
        timestep_mse,
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        markevery=markevery,
        alpha=alpha,
        label=resolved_label,
    )
    ax.set_xlabel('Rollout step')
    ax.set_ylabel('Average MSE (log scale)')
    ax.set_yscale('log')
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


def plot_fine_tune_history(
    fine_tune_outcome: dict,
    ax: Optional[Any] = None,
    title: str = 'Fine-tuning train and validation MSE by epoch',
    train_label: str = 'Train MSE',
    val_label: str = 'Validation MSE',
    linewidth: float = 2.0,
    save_path: Optional[Path | str] = None,
) -> dict:
    history = fine_tune_outcome.get('history')
    if not history:
        raise ValueError('fine_tune_outcome must include a non-empty history list.')

    pyplot = _import_pyplot()
    created_figure = ax is None
    if created_figure:
        figure, ax = pyplot.subplots(figsize=(8, 4.5))
    else:
        figure = ax.figure

    epochs = [entry['epoch'] for entry in history]
    train_loss = [entry['train_loss'] for entry in history]
    val_loss = [entry['val_loss'] for entry in history]

    ax.plot(epochs, train_loss, marker='o', linewidth=linewidth, label=train_label)
    ax.plot(epochs, val_loss, marker='s', linewidth=linewidth, label=val_label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE (log scale)')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, bbox_inches='tight')

    return {
        'figure': figure,
        'ax': ax,
        'history': history,
        'epochs': epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
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
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    plotted_evaluations = []
    for index, source in enumerate(evaluation_sources):
        evaluation = _load_evaluation_payload(source)
        resolved_label = labels[index] if labels is not None else None
        marker = markers[index % len(markers)]
        linestyle = linestyles[index % len(linestyles)]
        timestep_count = len(evaluation['timestep_mse'])
        markevery = None
        if timestep_count > 1:
            marker_offset = index % min(timestep_count, 4)
            markevery = (marker_offset, max(1, min(timestep_count // 4, 4)))
        plot_rollout_mse_curve(
            evaluation,
            label=resolved_label,
            ax=ax,
            title=title,
            linewidth=linewidth,
            marker=marker,
            linestyle=linestyle,
            markevery=markevery,
            alpha=0.95,
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
