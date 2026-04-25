"""Tests for k-space conversion and basic GUI k-space behavior."""

from collections.abc import Generator

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from nano_arpes_browser.core.models import (
    ARPESDataset,
    AxisInfo,
    ExperimentalParameters,
)
from nano_arpes_browser.core.processing.kspace import (
    HBAR_SQRT2M,
    KSpaceConverter,
    binding_to_kinetic,
    kinetic_to_binding,
)
from nano_arpes_browser.gui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp() -> Generator[QApplication, None, None]:
    """Ensure a QApplication exists for GUI tests."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestKSpaceConverter:
    """Tests for KSpaceConverter class."""

    @pytest.fixture
    def converter(self):
        return KSpaceConverter()

    def test_angle_to_k_at_normal_emission(self, converter):
        """k should be 0 at normal emission (θ=0)."""
        k = converter.angle_to_k(0.0, 100.0)
        assert k == pytest.approx(0.0, abs=1e-10)

    def test_angle_to_k_physical_constant(self, converter):
        """Verify the physical constant is correct."""
        # At 1 eV and 90°, k = 0.5124 Å⁻¹
        k = converter.angle_to_k(90.0, 1.0)
        assert k == pytest.approx(HBAR_SQRT2M, rel=1e-6)

    def test_angle_to_k_energy_scaling(self, converter):
        """k should scale as sqrt(E)."""
        k1 = converter.angle_to_k(30.0, 100.0)
        k4 = converter.angle_to_k(30.0, 400.0)
        assert k4 == pytest.approx(2.0 * k1, rel=1e-6)

    def test_angle_to_k_symmetry(self, converter):
        """Positive and negative angles should give opposite k."""
        k_pos = converter.angle_to_k(30.0, 100.0)
        k_neg = converter.angle_to_k(-30.0, 100.0)
        assert k_neg == pytest.approx(-k_pos, rel=1e-10)

    def test_k_to_angle_inverse(self, converter):
        """k_to_angle should be inverse of angle_to_k."""
        angles = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
        energy = 100.0

        k_values = converter.angle_to_k(angles, energy)
        angles_back = converter.k_to_angle(k_values, energy)

        np.testing.assert_allclose(angles_back, angles, rtol=1e-10)

    def test_max_k_at_energy(self, converter):
        """Test maximum k calculation."""
        # At 100 eV, k_max = 0.5124 * sqrt(100) = 5.124 Å⁻¹
        k_max = converter.max_k_at_energy(100.0)
        assert k_max == pytest.approx(HBAR_SQRT2M * 10.0, rel=1e-6)

    def test_angle_to_k_negative_energy_raises(self, converter):
        """Should raise error for negative energy."""
        with pytest.raises(ValueError, match="positive"):
            converter.angle_to_k(0.0, -10.0)

    def test_convert_spectrum_preserves_shape(self, converter):
        """Converted spectrum should have correct shape."""
        n_angles, n_energies = 100, 50
        rng = np.random.default_rng(0)
        spectrum = rng.random((n_angles, n_energies))
        energy_axis = np.linspace(80, 120, n_energies)
        angle_axis = np.linspace(-15, 15, n_angles)

        result = converter.convert_spectrum(spectrum, energy_axis, angle_axis)

        assert result.spectrum.shape == (n_angles, n_energies)
        assert len(result.k_axis) == n_angles
        assert len(result.energy_axis) == n_energies


class TestEnergyConversion:
    """Tests for energy conversion functions."""

    def test_binding_to_kinetic(self):
        """Test binding to kinetic energy conversion."""
        # E_B = 0 should give E_kin = hv - phi
        e_kin = binding_to_kinetic(np.array([0.0]), photon_energy=100.0, work_function=4.5)
        assert e_kin[0] == pytest.approx(95.5)

    def test_kinetic_to_binding(self):
        """Test kinetic to binding energy conversion."""
        e_binding = kinetic_to_binding(np.array([95.5]), photon_energy=100.0, work_function=4.5)
        assert e_binding[0] == pytest.approx(0.0)

    def test_energy_conversion_roundtrip(self):
        """Converting back and forth should give original value."""
        e_binding_orig = np.array([0.0, 1.0, 2.0, 5.0])
        hv = 100.0
        wf = 4.5

        e_kinetic = binding_to_kinetic(e_binding_orig, hv, wf)
        e_binding_back = kinetic_to_binding(e_kinetic, hv, wf)

        np.testing.assert_allclose(e_binding_back, e_binding_orig)


class TestMainWindowKSpaceBehavior:
    """Basic GUI-level tests for k-space-related interactions."""

    @pytest.fixture
    def main_window(self, qapp: QApplication) -> MainWindow:
        """Create a MainWindow with a small synthetic dataset."""
        # Small synthetic 4D dataset: (y, x, angle, energy)
        ny, nx, na, ne = 4, 3, 5, 6
        intensity = np.arange(ny * nx * na * ne, dtype=float).reshape(ny, nx, na, ne)

        x_axis = AxisInfo(values=np.linspace(-1.0, 1.0, nx), unit="µm", label="X")
        y_axis = AxisInfo(values=np.linspace(-1.0, 1.0, ny), unit="µm", label="Y")
        angle_axis = AxisInfo(values=np.linspace(-15.0, 15.0, na), unit="°", label="Angle")
        energy_axis = AxisInfo(values=np.linspace(80.0, 120.0, ne), unit="eV", label="Energy")

        dataset = ARPESDataset(
            intensity=intensity,
            x_axis=x_axis,
            y_axis=y_axis,
            angle_axis=angle_axis,
            energy_axis=energy_axis,
            experiment=ExperimentalParameters(),
        )

        window = MainWindow()
        window.dataset = dataset
        window._initialize_display()
        return window

    def test_select_position_updates_spectrum(self, main_window: MainWindow) -> None:
        """Selecting a spatial position should update ARPES viewer without errors."""
        x_center = main_window.dataset.x_axis.values[main_window.dataset.x_axis.size // 2]
        y_center = main_window.dataset.y_axis.values[main_window.dataset.y_axis.size // 2]

        main_window._on_spatial_position_changed(x_center, y_center)

        spectrum = main_window.arpes_viewer.get_current_data()
        x_axis, energy_axis = main_window.arpes_viewer.get_current_axes()

        assert main_window.current_position is not None
        assert spectrum is not None
        assert x_axis is not None
        assert energy_axis is not None
        # Spectrum should be (angle or k, energy)
        assert spectrum.shape == (
            main_window.dataset.angle_axis.size,
            main_window.dataset.energy_axis.size,
        )

    def test_toggle_kspace_refreshes_spectrum(self, main_window: MainWindow) -> None:
        """Toggling k-space should refresh spectrum and x-axis without raising."""
        x_center = main_window.dataset.x_axis.values[main_window.dataset.x_axis.size // 2]
        y_center = main_window.dataset.y_axis.values[main_window.dataset.y_axis.size // 2]
        main_window._on_spatial_position_changed(x_center, y_center)

        # Initial (angle-space) axes
        _, energy_axis_before = main_window.arpes_viewer.get_current_axes()

        # Enable k-space via control panel and trigger handler
        main_window.control_panel.set_zero_checkbox.setChecked(True)
        main_window.control_panel.k_space_checkbox.setChecked(True)
        main_window._on_kspace_changed()

        x_axis_k, energy_axis_after = main_window.arpes_viewer.get_current_axes()

        assert x_axis_k is not None
        assert energy_axis_after is not None
        # Energy axis should stay the same; x-axis should now be k-like
        np.testing.assert_allclose(energy_axis_after, energy_axis_before)
        angle_axis = main_window.dataset.angle_axis.values
        assert not np.allclose(x_axis_k, angle_axis)

    def test_changing_integration_refreshes_spectrum(self, main_window: MainWindow) -> None:
        """Changing integration settings should refresh spectrum without errors."""
        x_center = main_window.dataset.x_axis.values[main_window.dataset.x_axis.size // 2]
        y_center = main_window.dataset.y_axis.values[main_window.dataset.y_axis.size // 2]
        main_window._on_spatial_position_changed(x_center, y_center)

        spectrum_before = main_window.arpes_viewer.get_current_data()

        # Enable integration with a small region
        main_window.control_panel.integrate_checkbox.setChecked(True)
        main_window.control_panel.x_spinbox.setValue(1)
        main_window.control_panel.y_spinbox.setValue(1)
        main_window._on_integration_changed()

        spectrum_after = main_window.arpes_viewer.get_current_data()

        assert spectrum_before is not None
        assert spectrum_after is not None
        # Shapes should be unchanged, but data should generally differ
        assert spectrum_after.shape == spectrum_before.shape
        assert not np.allclose(spectrum_after, spectrum_before)
