"""Tests for data loaders."""

import h5py
import numpy as np
import pytest

from nano_arpes_browser.core.io.loaders import DataLoader


def _write_antares_file(path):
    with h5py.File(path, "w") as f:
        f.create_dataset(DataLoader.ANTARES_PATHS["data"], data=np.ones((2, 3, 4, 5)))
        f.create_dataset(DataLoader.ANTARES_PATHS["x_spatial"], data=np.array([[0.0, 1.0, 2.0]]))
        f.create_dataset(DataLoader.ANTARES_PATHS["y_spatial"], data=np.array([0.0, 1.0]))
        f.create_dataset(DataLoader.ANTARES_PATHS["energy_offset"], data=np.array([[10.0]]))
        f.create_dataset(DataLoader.ANTARES_PATHS["energy_step"], data=np.array([[0.5]]))
        f.create_dataset(DataLoader.ANTARES_PATHS["angle_offset"], data=np.array([[-2.0]]))
        f.create_dataset(DataLoader.ANTARES_PATHS["angle_step"], data=np.array([[1.0]]))


def test_load_rejects_unknown_hdf5_schema_with_missing_paths(tmp_path):
    path = tmp_path / "unknown.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("some_other_schema/data", data=np.ones((2, 2)))

    with pytest.raises(ValueError) as exc_info:
        DataLoader.load(path)

    message = str(exc_info.value)
    assert "Unsupported HDF5/NeXus schema" in message
    assert "ANTARES/SOLEIL nano-ARPES" in message
    assert DataLoader.ANTARES_PATHS["data"] in message
    assert "HDF5 tree" in message


def test_load_reads_antares_schema(tmp_path):
    path = tmp_path / "antares.nxs"
    _write_antares_file(path)

    dataset = DataLoader.load(path)

    assert dataset.shape == (2, 3, 4, 5)
    assert dataset.x_axis.size == 3
    assert dataset.y_axis.size == 2
    assert dataset.angle_axis.size == 4
    assert dataset.energy_axis.size == 5
