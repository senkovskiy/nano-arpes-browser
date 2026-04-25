"""Tests for data exporters."""

import numpy as np

from nano_arpes_browser.core.io.export_models import RegionExportData
from nano_arpes_browser.core.io.exporters import DataExporter


def test_save_region_itx_uses_supplied_spatial_map(tmp_path, monkeypatch):
    """Region export should not recompute display orientation internally."""
    data = np.arange(2 * 3 * 2 * 2, dtype=float).reshape(2, 3, 2, 2)
    spatial_map = np.array([[101.0, 102.0], [201.0, 202.0], [301.0, 302.0]])
    x_axis = np.array([0.0, 1.0, 2.0])
    y_axis = np.array([10.0, 20.0])
    angle_axis = np.array([-1.0, 1.0])
    energy_axis = np.array([90.0, 91.0])

    written_2d_waves = {}
    original_write_2d_wave = DataExporter._write_2d_wave

    def capture_2d_wave(f, name, wave_data, *args, **kwargs):
        written_2d_waves[name] = np.array(wave_data, copy=True)
        return original_write_2d_wave(f, name, wave_data, *args, **kwargs)

    monkeypatch.setattr(DataExporter, "_write_2d_wave", staticmethod(capture_2d_wave))

    region = RegionExportData(
        data=data,
        spatial_map=spatial_map,
        x_axis=x_axis,
        y_axis=y_axis,
        angle_axis=angle_axis,
        energy_axis=energy_axis,
    )
    DataExporter.save_region_itx(region, tmp_path / "region.itx")

    np.testing.assert_array_equal(written_2d_waves["region_spatial"], spatial_map)
