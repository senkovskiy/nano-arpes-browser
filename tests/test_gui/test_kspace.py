"""Tests for k-space conversion."""

import numpy as np
import pytest
from src.core.processing.kspace import (
    HBAR_SQRT2M,
    KSpaceConverter,
    binding_to_kinetic,
    kinetic_to_binding,
)


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
        spectrum = np.random.rand(n_angles, n_energies)
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
        # E_B = 0 should give E_kin = hν - φ
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
