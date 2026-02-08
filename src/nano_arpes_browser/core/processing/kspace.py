"""
Angle to k-space conversion utilities.

Physics:
    The parallel component of electron momentum is:

    k_∥ = (1/ℏ) * sqrt(2 * m_e * E_kin) * sin(θ)

    Where:
    - E_kin is the kinetic energy of the photoelectron
    - θ is the emission angle relative to surface normal
    - m_e is the electron mass
    - ℏ is the reduced Planck constant

    Numerically:
    k_∥ [Å⁻¹] = 0.5124 * sqrt(E_kin [eV]) * sin(θ [rad])
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d


# Physical constant: sqrt(2 * m_e) / hbar in units of Å⁻¹ / sqrt(eV)
HBAR_SQRT2M = 0.512316722


@dataclass
class KSpaceResult:
    """Result of k-space conversion."""

    spectrum: np.ndarray  # Converted spectrum (k, energy)
    k_axis: np.ndarray  # k values in Å⁻¹
    energy_axis: np.ndarray  # Energy values (unchanged)
    k_min: float
    k_max: float


class KSpaceConverter:
    """
    Convert ARPES data between angle and k-space.

    Handles:
    - Angle to k conversion with energy-dependent k-range
    - Proper interpolation avoiding artifacts
    - Both kinetic and binding energy scales
    """

    def __init__(self, constant: float = HBAR_SQRT2M):
        """
        Initialize converter.

        Args:
            constant: sqrt(2*m_e)/hbar in Å⁻¹ eV^(-1/2)
                     Default is the physical constant.
                     Can be adjusted for effective mass: constant * sqrt(m*/m_e)
        """
        self.constant = constant

    def angle_to_k(
        self,
        angle_deg: float | np.ndarray,
        kinetic_energy_eV: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Convert emission angle to parallel momentum.

        Args:
            angle_deg: Emission angle(s) in degrees
            kinetic_energy_eV: Kinetic energy in eV (must be positive)

        Returns:
            k_parallel in Å⁻¹

        Raises:
            ValueError: If energy is not positive
        """
        energy = np.asarray(kinetic_energy_eV)
        if np.any(energy <= 0):
            raise ValueError("Kinetic energy must be positive for k conversion")

        angle_rad = np.radians(angle_deg)
        return self.constant * np.sqrt(energy) * np.sin(angle_rad)

    def k_to_angle(
        self,
        k: float | np.ndarray,
        kinetic_energy_eV: float | np.ndarray,
    ) -> float | np.ndarray:
        """
        Convert parallel momentum to emission angle.

        Args:
            k: k_parallel in Å⁻¹
            kinetic_energy_eV: Kinetic energy in eV

        Returns:
            Emission angle in degrees

        Raises:
            ValueError: If |k| exceeds maximum possible value at given energy
        """
        energy = np.asarray(kinetic_energy_eV)
        k = np.asarray(k)

        if np.any(energy <= 0):
            raise ValueError("Kinetic energy must be positive for k conversion")

        # Maximum possible k at this energy
        k_max = self.constant * np.sqrt(energy)

        # Check if k is physically possible
        ratio = k / k_max

        # Clip to valid range with small tolerance for numerical errors
        ratio = np.clip(ratio, -1.0, 1.0)

        return np.degrees(np.arcsin(ratio))

    def max_k_at_energy(self, kinetic_energy_eV: float) -> float:
        """
        Get maximum possible k at given kinetic energy.

        This corresponds to emission at 90° (grazing emission).
        """
        return self.constant * np.sqrt(kinetic_energy_eV)

    def convert_spectrum(
        self,
        spectrum: np.ndarray,          # shape (n_angles, n_energies)
        energy_axis: np.ndarray,       # shape (n_energies,)
        angle_axis: np.ndarray,        # shape (n_angles,)
        zero_angle: float = 0.0,
        n_k_points: int | None = None,
    interpolation_kind: str = "linear",
    ) -> KSpaceResult:
        """
        Convert ARPES spectrum from angle to k-space.

        The conversion accounts for the energy-dependent relationship between
        angle and k. At each energy, the accessible k-range is different:
        k_max(E) = 0.5124 * sqrt(E).

        Args:
            spectrum: 2D array with shape (n_angles, n_energies)
            energy_axis: Kinetic energy values in eV
            angle_axis: Angle values in degrees
            zero_angle: Angle corresponding to k=0 (for alignment)
            n_k_points: Number of k points (default: same as n_angles)
            interpolation_kind: Interpolation method ('linear', 'cubic')

        Returns:
            KSpaceResult with converted spectrum and k-axis
        """
        n_angles, n_energies = spectrum.shape
        if n_k_points is None:
            n_k_points = n_angles

        angles_shifted = angle_axis - zero_angle
        e_max = float(np.max(energy_axis))

        k_min = float(self.angle_to_k(np.min(angles_shifted), e_max))
        k_max = float(self.angle_to_k(np.max(angles_shifted), e_max))
        k_axis = np.linspace(k_min, k_max, n_k_points)

        out = np.zeros((n_k_points, n_energies), dtype=float)

        for i_e, e in enumerate(energy_axis):
            if e <= 0:
                continue

            interp = interp1d(
                angles_shifted,
                spectrum[:, i_e],
                kind=interpolation_kind,
                bounds_error=False,
                fill_value=0.0,
                assume_sorted=False,
            )

            k_max_at_e = self.max_k_at_energy(float(e))
            valid = np.abs(k_axis) <= k_max_at_e
            if not np.any(valid):
                continue

            angles = self.k_to_angle(k_axis[valid], float(e))
            out[valid, i_e] = interp(angles)

        return KSpaceResult(
            spectrum=out,
            k_axis=k_axis,
            energy_axis=energy_axis,
            k_min=k_min,
            k_max=k_max,
        )

    def convert_spectrum_fast(
        self,
        spectrum: np.ndarray,
        energy_axis: np.ndarray,
        angle_axis: np.ndarray,
        zero_angle: float = 0.0,
        n_k_points: int | None = None,
    ) -> KSpaceResult:
        """
        Fast k-space conversion using 2D interpolation.

        This is faster than convert_spectrum() but may have slightly
        different interpolation behavior at the edges.

        Args:
            spectrum: 2D array with shape (n_angles, n_energies)
            energy_axis: Kinetic energy values in eV
            angle_axis: Angle values in degrees
            zero_angle: Angle corresponding to k=0
            n_k_points: Number of k points (default: same as n_angles)

        Returns:
            KSpaceResult with converted spectrum and k-axis
        """
        n_angles, n_energies = spectrum.shape

        if n_k_points is None:
            n_k_points = n_angles

        angles_shifted = angle_axis - zero_angle

        # Create 2D interpolation function
        interp2d_func = interpolate.interp2d(
            angles_shifted,
            energy_axis,
            spectrum.T,  # Transpose to (energy, angle)
            kind="linear",
            fill_value=0.0,
        )

        # Determine k range
        e_max = energy_axis.max()
        k_min = self.angle_to_k(angles_shifted.min(), e_max)
        k_max = self.angle_to_k(angles_shifted.max(), e_max)
        k_axis = np.linspace(k_min, k_max, n_k_points)

        # Allocate result
        result = np.zeros((n_k_points, n_energies))

        # Process each energy
        for i_e, energy in enumerate(energy_axis):
            if energy <= 0:
                continue

            k_max_at_e = self.max_k_at_energy(energy)

            # Find valid k range
            valid_k_mask = np.abs(k_axis) <= k_max_at_e
            valid_k = k_axis[valid_k_mask]

            if len(valid_k) == 0:
                continue

            # Convert k to angles
            angles_for_k = self.k_to_angle(valid_k, energy)

            # Interpolate
            result[valid_k_mask, i_e] = interp2d_func(angles_for_k, energy).flatten()

        return KSpaceResult(
            spectrum=result,
            k_axis=k_axis,
            energy_axis=energy_axis,
            k_min=float(k_min),
            k_max=float(k_max),
        )


def binding_to_kinetic(
    binding_energy: np.ndarray,
    photon_energy: float,
    work_function: float = 4.5,
) -> np.ndarray:
    """
    Convert binding energy to kinetic energy.

    E_kin = hν - φ - E_B

    Args:
        binding_energy: Binding energy in eV (positive values = below Fermi)
        photon_energy: Photon energy in eV
        work_function: Analyzer work function in eV

    Returns:
        Kinetic energy in eV
    """
    return photon_energy - work_function - binding_energy


def kinetic_to_binding(
    kinetic_energy: np.ndarray,
    photon_energy: float,
    work_function: float = 4.5,
) -> np.ndarray:
    """
    Convert kinetic energy to binding energy.

    E_B = hν - φ - E_kin

    Args:
        kinetic_energy: Kinetic energy in eV
        photon_energy: Photon energy in eV
        work_function: Analyzer work function in eV

    Returns:
        Binding energy in eV
    """
    return photon_energy - work_function - kinetic_energy
