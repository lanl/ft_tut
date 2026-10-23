from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import netCDF4 as nc
    from spus_ft.dataloader import load_trajectories_by_indices, pad_channels
    DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    np = None
    nc = None
    load_trajectories_by_indices = None
    pad_channels = None
    DEPS_AVAILABLE = False


@unittest.skipUnless(DEPS_AVAILABLE, "numpy/netCDF4 are not installed in this test environment")
class TestDataloaderPadding(unittest.TestCase):
    def test_pad_channels_adds_ones_to_target_channels(self) -> None:
        array = np.zeros((2, 3, 3), dtype=np.float32)

        padded = pad_channels(array, target_channels=5, pad_value=1.0)

        self.assertEqual(padded.shape, (5, 3, 3))
        self.assertTrue(np.allclose(padded[:2], 0.0))
        self.assertTrue(np.allclose(padded[2:], 1.0))

    def test_load_trajectories_by_indices_pads_channel_dimension(self) -> None:
        with tempfile.TemporaryDirectory(prefix="spus-ft-nc-") as tmp_dir:
            data_path = Path(tmp_dir) / "mini.nc"
            with nc.Dataset(data_path, 'w') as ds:
                ds.createDimension('sample', 2)
                ds.createDimension('time', 3)
                ds.createDimension('channel', 2)
                ds.createDimension('x', 4)
                ds.createDimension('y', 4)
                var = ds.createVariable('solution', 'f4', ('sample', 'time', 'channel', 'x', 'y'))
                var[:] = np.zeros((2, 3, 2, 4, 4), dtype=np.float32)

            trajectories, variable_name = load_trajectories_by_indices(
                str(data_path),
                [0],
                target_channels=5,
                pad_value=1.0,
            )

            self.assertEqual(variable_name, 'solution')
            self.assertEqual(tuple(trajectories.shape), (1, 3, 5, 4, 4))
            self.assertTrue(np.allclose(trajectories.numpy()[:, :, :2], 0.0))
            self.assertTrue(np.allclose(trajectories.numpy()[:, :, 2:], 1.0))
