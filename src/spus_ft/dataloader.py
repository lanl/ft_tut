from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import netCDF4 as nc
import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


def pad_channels(array, target_channels: Optional[int] = None, pad_value: float = 1.0):
    if target_channels is None:
        return array

    current_channels = array.shape[0]
    if current_channels == target_channels:
        return array
    if current_channels > target_channels:
        raise ValueError(
            f"Cannot pad from {current_channels} channels down to target_channels={target_channels}"
        )

    pad_shape = (target_channels - current_channels, *array.shape[1:])
    pad_block = np.full(pad_shape, pad_value, dtype=array.dtype)
    return np.concatenate([array, pad_block], axis=0)


class NetCDFTrajectoryDataset(Dataset):
    def __init__(self, data_path, variable_name='data', transform=None, target_channels: Optional[int] = None, pad_value: float = 1.0):
        """
        Note: Recommended to explicitly call close() to close the NetCDF files.
        Args:
            data_path (str): Path to a .nc file (or directory with .nc files) containing the data.
                             Expected shape is either (N_traj, 21, 5, 128, 128) or (N_traj, 21, 128, 128, 5).
            variable_name (str): Name of the variable to extract.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_path = data_path
        self.variable_name = variable_name
        self.transform = transform
        self.target_channels = target_channels
        self.pad_value = pad_value

        with nc.Dataset(data_path, mode='r') as ds:
            self.N, self.T, self.C, self.H, self.W = ds[variable_name].shape

        self.n_samples_per_traj = self.T - 1
        self.n_samples = self.N * self.n_samples_per_traj
        self.ds = None

    def _ensure_open(self):
        if self.ds is None:
            self.ds = nc.Dataset(self.data_path, "r")

    def __del__(self):
        self.close()

    def close(self):
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._ensure_open()

        traj_idx = idx // self.n_samples_per_traj
        time_idx = idx % self.n_samples_per_traj

        x = self.ds[self.variable_name][traj_idx, time_idx]
        y = self.ds[self.variable_name][traj_idx, time_idx + 1]

        if os.path.basename(self.data_path) == 'CE-RM.nc':
            x[-1] = 0.5 * x[0] * np.sum(x[1:3] ** 2, axis=0) + x[3] / (1.4 - 1.0)
            y[-1] = 0.5 * y[0] * np.sum(y[1:3] ** 2, axis=0) + y[3] / (1.4 - 1.0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        x = pad_channels(x, target_channels=self.target_channels, pad_value=self.pad_value)
        y = pad_channels(y, target_channels=self.target_channels, pad_value=self.pad_value)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class NetCDFPairsInTrajectoryDataset(Dataset):
    def __init__(self, data_path, variable_name='data', transform=None, target_channels: Optional[int] = None, pad_value: float = 1.0):
        self.data_path = data_path
        self.variable_name = variable_name
        self.transform = transform
        self.target_channels = target_channels
        self.pad_value = pad_value

        with nc.Dataset(data_path, mode='r') as ds:
            self.N, self.T, self.C, self.H, self.W = ds[variable_name].shape

        self.n_samples_per_traj = self.T * self.T
        self.n_samples = self.N * self.n_samples_per_traj
        self.ds = None

    def _ensure_open(self):
        if self.ds is None:
            self.ds = nc.Dataset(self.data_path, "r")

    def __del__(self):
        self.close()

    def close(self):
        if self.ds is not None:
            self.ds.close()
            self.ds = None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        self._ensure_open()

        traj_idx = idx // self.n_samples_per_traj
        pair_idx = idx % self.n_samples_per_traj
        time_idx1 = pair_idx // self.T
        time_idx2 = pair_idx % self.T

        x = self.ds[self.variable_name][traj_idx, time_idx1]
        y = self.ds[self.variable_name][traj_idx, time_idx2]

        if os.path.basename(self.data_path) == 'CE-RM.nc':
            x[-1] = 0.5 * x[0] * np.sum(x[1:3] ** 2, axis=0) + x[3] / (1.4 - 1.0)
            y[-1] = 0.5 * y[0] * np.sum(y[1:3] ** 2, axis=0) + y[3] / (1.4 - 1.0)

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        x = pad_channels(x, target_channels=self.target_channels, pad_value=self.pad_value)
        y = pad_channels(y, target_channels=self.target_channels, pad_value=self.pad_value)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), time_idx1, time_idx2


def infer_variable_name(data_path: str, variable_name: Optional[str] = None) -> str:
    if variable_name is not None:
        return variable_name

    with nc.Dataset(data_path, mode='r') as ds:
        if 'data' in ds.variables:
            return 'data'
        if 'solution' in ds.variables:
            return 'solution'
        for name, variable in ds.variables.items():
            if len(variable.shape) == 5:
                return name

    raise ValueError(f"Could not infer a 5D variable name from {data_path}")


def get_trajectory_dataset_info(data_path: str, variable_name: Optional[str] = None) -> dict:
    resolved_variable_name = infer_variable_name(data_path, variable_name)
    with nc.Dataset(data_path, mode='r') as ds:
        shape = tuple(int(dim) for dim in ds[resolved_variable_name].shape)
    return {
        'data_path': str(data_path),
        'variable_name': resolved_variable_name,
        'shape': shape,
        'num_trajectories': shape[0],
        'num_timesteps': shape[1],
        'num_channels': shape[2],
        'height': shape[3],
        'width': shape[4],
    }


def _compute_split_counts(
    total_trajectories: int,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    num_train_trajectories: Optional[int] = None,
) -> Tuple[int, int, int]:
    if total_trajectories < 3:
        train_count = max(1, total_trajectories - 1)
        val_count = 0 if total_trajectories < 2 else 1
        test_count = total_trajectories - train_count - val_count
        return train_count, val_count, test_count

    default_train = int(total_trajectories * splits[0])
    default_val = int(total_trajectories * splits[1])
    train_count = num_train_trajectories if num_train_trajectories is not None else default_train
    train_count = max(1, min(train_count, total_trajectories - 2))

    remaining = total_trajectories - train_count
    val_count = int(total_trajectories * splits[1])
    val_count = max(1, min(val_count, remaining - 1))
    test_count = total_trajectories - train_count - val_count
    return train_count, val_count, test_count


def get_fns_kf_paper_split_trajectory_indices(total_trajectories: int) -> Tuple[list[int], list[int], list[int]]:
    if total_trajectories < 20000:
        raise ValueError(
            f"FNS-KF paper split expects 20000 trajectories, but got {total_trajectories}"
        )

    train_indices = list(range(0, 19640))
    val_indices = list(range(19640, 19760))
    test_indices = list(range(19760, 20000))
    return train_indices, val_indices, test_indices


def get_train_val_test_trajectory_indices_from_file(
    data_path: str,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = None,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    split_strategy: str = 'fractional',
) -> Tuple[list[int], list[int], list[int], dict]:
    dataset_info = get_trajectory_dataset_info(data_path, variable_name)
    if split_strategy == 'fns_kf_paper':
        train_indices, val_indices, test_indices = get_fns_kf_paper_split_trajectory_indices(
            dataset_info['num_trajectories']
        )
        if num_train_trajectories is not None:
            train_indices = train_indices[:num_train_trajectories]
    elif split_strategy == 'fractional':
        train_count, val_count, test_count = _compute_split_counts(
            dataset_info['num_trajectories'],
            splits=splits,
            num_train_trajectories=num_train_trajectories,
        )
        train_indices = list(range(0, train_count))
        val_indices = list(range(train_count, train_count + val_count))
        test_indices = list(range(train_count + val_count, train_count + val_count + test_count))
    else:
        raise ValueError("split_strategy must be one of: 'fractional', 'fns_kf_paper'")
    split_info = {
        **dataset_info,
        'train_trajectories': len(train_indices),
        'val_trajectories': len(val_indices),
        'test_trajectories': len(test_indices),
        'splits': list(splits),
        'split_strategy': split_strategy,
    }
    return train_indices, val_indices, test_indices, split_info


def get_train_val_test_split_from_file(
    data_path: str,
    ds_type=NetCDFTrajectoryDataset,
    variable_name: Optional[str] = None,
    num_train_trajectories: Optional[int] = None,
    splits: Sequence[float] = (0.7, 0.2, 0.1),
    target_channels: Optional[int] = None,
    pad_value: float = 1.0,
    split_strategy: str = 'fractional',
):
    resolved_variable_name = infer_variable_name(data_path, variable_name)
    dataset = ds_type(
        data_path,
        resolved_variable_name,
        target_channels=target_channels,
        pad_value=pad_value,
    )
    train_indices, val_indices, test_indices, _ = get_train_val_test_trajectory_indices_from_file(
        data_path,
        variable_name=resolved_variable_name,
        num_train_trajectories=num_train_trajectories,
        splits=splits,
        split_strategy=split_strategy,
    )
    n_samples_per_traj = dataset.n_samples_per_traj

    def expand(indices: Iterable[int]) -> list[int]:
        expanded = []
        for traj_idx in indices:
            start = traj_idx * n_samples_per_traj
            expanded.extend(range(start, start + n_samples_per_traj))
        return expanded

    train_set = Subset(dataset, expand(train_indices))
    val_set = Subset(dataset, expand(val_indices))
    test_set = Subset(dataset, expand(test_indices))
    return train_set, val_set, test_set


def load_trajectories_by_indices(
    data_path: str,
    trajectory_indices: Sequence[int],
    variable_name: Optional[str] = None,
    target_channels: Optional[int] = None,
    pad_value: float = 1.0,
):
    resolved_variable_name = infer_variable_name(data_path, variable_name)
    with nc.Dataset(data_path, mode='r') as ds:
        data = ds[resolved_variable_name][list(trajectory_indices)]
    if target_channels is not None:
        padded = []
        for trajectory in data:
            padded.append([pad_channels(frame, target_channels=target_channels, pad_value=pad_value) for frame in trajectory])
        data = np.asarray(padded, dtype=data.dtype)
    return torch.tensor(data, dtype=torch.float32), resolved_variable_name


def get_train_val_test_split_from_dir(data_dir, ds_type=NetCDFTrajectoryDataset,
                                      n_trajs_per_file=None, splits=[0.7, 0.2, 0.1],
                                      file_names = None):
    if file_names == None:
        file_names = ["CE-CRP.nc", "CE-Gauss.nc", "CE-KH.nc", "CE-RP.nc"]
    train_ds, val_ds, test_ds = [], [], []

    for fname in file_names:
        file_path = os.path.join(data_dir, fname)
        if "CE-RM.nc" in fname:
            var_name = "solution"
        else:
            var_name = "data"

        ds = ds_type(file_path, var_name)
        n_samples_per_traj = ds.n_samples_per_traj
        if n_trajs_per_file is not None:
            ds = Subset(ds, indices=range(n_trajs_per_file * n_samples_per_traj))
        else:
            n_trajs_per_file = ds.N

        n_train = int(splits[0] * n_trajs_per_file) * n_samples_per_traj
        n_val = int(splits[1] * n_trajs_per_file) * n_samples_per_traj

        train_set = Subset(ds, range(n_train))
        val_set = Subset(ds, range(n_train, n_train + n_val))
        test_set = Subset(ds, range(n_train + n_val, len(ds)))

        train_ds.append(train_set)
        val_ds.append(val_set)
        test_ds.append(test_set)

    train_ds = ConcatDataset(datasets=train_ds)
    val_ds = ConcatDataset(datasets=val_ds)
    test_ds = ConcatDataset(datasets=test_ds)

    return train_ds, val_ds, test_ds


def get_train_val_test_from_dir(data_dir, ds_type=NetCDFTrajectoryDataset,
                                      file_names = None):
    if file_names == None:
        file_names = ["CE-CRP", "CE-Gauss", "CE-KH", "CE-RP"]
    train_ds, val_ds, test_ds = [], [], []
    for eval_type in ['train','test','val']:
        for fname in file_names:
            file_path = os.path.join(data_dir, fname+f'_{eval_type}.nc')
            if "CE-RM.nc" in fname:
                var_name = "solution"
            else:
                var_name = "data"

            ds = ds_type(file_path, var_name)
            if eval_type == 'train':
                train_ds.append(ds)
            elif eval_type == 'test':
                test_ds.append(ds)
            elif eval_type == 'val':
                val_ds.append(ds)
    train_ds = ConcatDataset(datasets=train_ds)
    val_ds = ConcatDataset(datasets=val_ds)
    test_ds = ConcatDataset(datasets=test_ds)

    return train_ds, val_ds, test_ds


def close_all_datasets(dataset):
    if isinstance(dataset, torch.utils.data.ConcatDataset):
        for d in dataset.datasets:
            close_all_datasets(d)
    elif isinstance(dataset, torch.utils.data.Subset):
        close_all_datasets(dataset.dataset)
    elif hasattr(dataset, "close"):
        dataset.close()


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    data_path = '/projects/artimis/PDEgym/CompressibleEuler/downstream/fulldata/CE-RM.nc'
    ds = NetCDFTrajectoryDataset(data_path, variable_name='solution')

    print(ds[40][0].size(),len(ds))
    close_all_datasets(ds)
