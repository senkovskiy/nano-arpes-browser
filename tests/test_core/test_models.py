"""Tests for data models."""

import numpy as np
import pytest
from src.core.models import (
    ARPESDataset,
    AxisInfo,
    IntegrationParams,
    SpatialPosition,
)


@pytest.fixture
def sample_dataset() -> ARPESDataset:
    """Create a sample dataset for testing."""
    n_y, n_x, n_angle, n_energy = 10, 10, 50, 100

    intensity = np.random.rand(n_y, n_x, n_angle, n_energy).astype(np.float32)

    x_axis = AxisInfo(
        values=np.linspace(0, 100, n_x),
        unit="µm",
        label="X Position",
    )
    y_axis = AxisInfo(
        values=np.linspace(0, 100, n_y),
        unit="µm",
        label="Y Position",
    )
    angle_axis = AxisInfo(
        values=np.linspace(-15, 15, n_angle),
        unit="°",
        label="Emission Angle",
    )
    energy_axis = AxisInfo(
        values=np.linspace(80, 120, n_energy),
        unit="eV",
        label="Kinetic Energy",
    )

    return ARPESDataset(
        intensity=intensity,
        x_axis=x_axis,
        y_axis=y_axis,
        angle_axis=angle_axis,
        energy_axis=energy_axis,
    )


class TestAxisInfo:
    """Tests for AxisInfo model."""

    def test_axis_properties(self):
        """Test computed properties."""
        axis = AxisInfo(
            values=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            unit="eV",
            label="Energy",
        )

        assert axis.min == 0.0
        assert axis.max == 4.0
        assert axis.step == 1.0
        assert axis.size == 5

    def test_find_nearest_index(self):
        """Test finding nearest index."""
        axis = AxisInfo(
            values=np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
            unit="eV",
            label="Energy",
        )

        assert axis.find_nearest_index(0.3) == 0
        assert axis.find_nearest_index(0.6) == 1
        assert axis.find_nearest_index(2.5) == 2  # or 3, depending on rounding


class TestARPESDataset:
    """Tests for ARPESDataset model."""

    def test_shape_property(self, sample_dataset):
        """Test shape property."""
        assert sample_dataset.shape == (10, 10, 50, 100)

    def test_integrated_image(self, sample_dataset):
        """Test integrated image computation."""
        image = sample_dataset.integrated_image
        assert image.shape == (10, 10)

    def test_get_spectrum_at(self, sample_dataset):
        """Test spectrum extraction."""
        position = SpatialPosition(
            x_index=5,
            y_index=5,
            x_coord=50.0,
            y_coord=50.0,
        )

        spectrum = sample_dataset.get_spectrum_at(position)
        assert spectrum.shape == (50, 100)

    def test_get_spectrum_with_integration(self, sample_dataset):
        """Test spectrum extraction with integration."""
        position = SpatialPosition(
            x_index=5,
            y_index=5,
            x_coord=50.0,
            y_coord=50.0,
        )
        integration = IntegrationParams(enabled=True, x_pixels=2, y_pixels=2)

        spectrum = sample_dataset.get_spectrum_at(position, integration)
        assert spectrum.shape == (50, 100)

    def test_position_from_coords(self, sample_dataset):
        """Test position creation from coordinates."""
        position = sample_dataset.position_from_coords(50.0, 50.0)

        assert isinstance(position, SpatialPosition)
        assert position.x_index >= 0
        assert position.y_index >= 0
