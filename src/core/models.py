"""Data models for ARPES datasets using Pydantic."""

from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class EnergyType(str, Enum):
    """Type of energy axis."""

    KINETIC = "kinetic"
    BINDING = "binding"


class AxisInfo(BaseModel):
    """Information about a data axis with physical units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: np.ndarray
    unit: str
    label: str

    @computed_field
    @property
    def min(self) -> float:
        """Minimum value of axis."""
        return float(self.values[0])

    @computed_field
    @property
    def max(self) -> float:
        """Maximum value of axis."""
        return float(self.values[-1])

    @computed_field
    @property
    def step(self) -> float:
        """Step size of axis."""
        if len(self.values) < 2:
            return 0.0
        return float(self.values[1] - self.values[0])

    @computed_field
    @property
    def size(self) -> int:
        """Number of points in axis."""
        return len(self.values)

    def find_nearest_index(self, value: float) -> int:
        """Find index of nearest value in axis."""
        return int(np.argmin(np.abs(self.values - value)))

    def find_nearest_value(self, value: float) -> float:
        """Find nearest value in axis."""
        return float(self.values[self.find_nearest_index(value)])


class SpatialPosition(BaseModel):
    """Current position in spatial map."""

    x_index: int = Field(ge=0, description="X index in data array")
    y_index: int = Field(ge=0, description="Y index in data array")
    x_coord: float = Field(description="X coordinate in physical units")
    y_coord: float = Field(description="Y coordinate in physical units")


class EnergyAngleROI(BaseModel):
    """Region of interest in energy-angle space."""

    angle_start_idx: int = Field(ge=0)
    angle_end_idx: int = Field(ge=0)
    energy_start_idx: int = Field(ge=0)
    energy_end_idx: int = Field(ge=0)

    # Physical values
    angle_start: float | None = None
    angle_end: float | None = None
    energy_start: float | None = None
    energy_end: float | None = None

    @model_validator(mode="after")
    def validate_ranges(self) -> "EnergyAngleROI":
        """Ensure start <= end for all ranges."""
        if self.angle_start_idx > self.angle_end_idx:
            self.angle_start_idx, self.angle_end_idx = (
                self.angle_end_idx,
                self.angle_start_idx,
            )
        if self.energy_start_idx > self.energy_end_idx:
            self.energy_start_idx, self.energy_end_idx = (
                self.energy_end_idx,
                self.energy_start_idx,
            )
        return self


class ExperimentalParameters(BaseModel):
    """Experimental parameters for ARPES measurement."""

    photon_energy: float | None = Field(default=None, gt=0, description="Photon energy in eV")
    work_function: float = Field(
        default=4.5, gt=0, lt=10, description="Analyzer work function in eV"
    )
    temperature: float | None = Field(default=None, ge=0, description="Sample temperature in K")
    polarization: str | None = Field(
        default=None, description="Light polarization (LH, LV, CR, CL)"
    )
    energy_type: EnergyType = Field(default=EnergyType.KINETIC, description="Type of energy axis")

    @computed_field
    @property
    def fermi_energy(self) -> float | None:
        """Fermi energy (kinetic) if photon energy is known."""
        if self.photon_energy is not None:
            return self.photon_energy - self.work_function
        return None


class IntegrationParams(BaseModel):
    """Parameters for spatial integration."""

    enabled: bool = False
    x_pixels: int = Field(default=0, ge=0, le=100)
    y_pixels: int = Field(default=0, ge=0, le=100)


class KSpaceParams(BaseModel):
    """Parameters for k-space conversion."""

    enabled: bool = False
    zero_angle: float = Field(default=0.0, ge=-90, le=90)
    angle_offset: float = Field(default=0.0, description="Additional angle offset for alignment")


class ARPESDataset(BaseModel):
    """
    Container for micro-ARPES dataset.

    Data shape convention (internal raw data):
        intensity.shape == (n_y, n_x, n_angle, n_energy)

    Display conventions:
        - Spatial images are shown as rot90(sum over angle & energy) for viewer alignment.
        - X display index is reversed relative to internal storage (beamline convention).

    Physical conventions:
        - Positive angles = emission towards analyzer
        - Kinetic energy increases with index
        - Binding energy = photon_energy - work_function - kinetic_energy
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
    # Raw intensity data
    intensity: np.ndarray = Field(description="4D intensity array (y, x, angle, energy)")

    x_axis: AxisInfo
    y_axis: AxisInfo
    angle_axis: AxisInfo
    energy_axis: AxisInfo

    experiment: ExperimentalParameters = Field(default_factory=ExperimentalParameters)

    filepath: Path | None = None
    metadata: dict = Field(default_factory=dict)

    # Cache
    _integrated_image: np.ndarray | None = None

    # ---------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------

    @field_validator("intensity")
    @classmethod
    def validate_intensity_shape(cls, v: np.ndarray) -> np.ndarray:
        """Validate intensity array is 4D."""
        if v.ndim != 4:
            raise ValueError(f"Intensity must be 4D array, got {v.ndim}D")
        return v

    @model_validator(mode="after")
    def validate_axis_dimensions(self) -> "ARPESDataset":
        """Validate axis dimensions match intensity shape."""
        ny, nx, na, ne = self.intensity.shape
        expected = {
            "y_axis": ny,
            "x_axis": nx,
            "angle_axis": na,
            "energy_axis": ne,
        }
        for axis_name, expected_size in expected.items():
            axis = getattr(self, axis_name)
            if axis.size != expected_size:
                raise ValueError(
                    f"{axis_name} size {axis.size} doesn't match intensity dimension {expected_size}"
                )
        return self

    # ---------------------------------------------------------------------
    # Basic properties
    # ---------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return (n_y, n_x, n_angle, n_energy)."""
        return tuple(self.intensity.shape)

    @computed_field
    @property
    def n_y(self) -> int:
        return int(self.intensity.shape[0])

    @computed_field
    @property
    def n_x(self) -> int:
        return int(self.intensity.shape[1])

    @computed_field
    @property
    def n_angle(self) -> int:
        return int(self.intensity.shape[2])

    @computed_field
    @property
    def n_energy(self) -> int:
        return int(self.intensity.shape[3])

    # ---------------------------------------------------------------------
    # Display mapping helpers (single source of truth)
    # ---------------------------------------------------------------------

    def data_x_index(self, display_x_index: int) -> int:
        """
        Convert display X index to internal data X index.

        Your current convention: X is stored reversed in the raw data format.
        Keep this mapping HERE so GUI/export code never re-implements it.
        """
        return self.x_axis.size - 1 - int(display_x_index)

    def spatial_image_for_selection(self, selection: np.ndarray) -> np.ndarray:
        """
        Convert a (y, x, angle, energy) selection into the displayed 2D spatial image.

        Display convention used across app: rotate 90° after integrating.
        Keep this mapping HERE so GUI/export code never re-implements it.
        """
        # selection expected shape: (ny, nx, na, ne)
        return np.rot90(np.sum(selection, axis=(2, 3)))

    def spectrum_at_display_index(
        self,
        y_index: int,
        x_index: int,
        integration: IntegrationParams | None = None,
    ) -> np.ndarray:
        """
        Convenience accessor for GUI: provide display indices, get (angle, energy) spectrum.

        This shields GUI from the internal X reversal.
        """
        pos = SpatialPosition(
            x_index=int(x_index),
            y_index=int(y_index),
            x_coord=float(self.x_axis.values[int(x_index)]),
            y_coord=float(self.y_axis.values[int(y_index)]),
        )
        return self.get_spectrum_at(pos, integration)

    # ---------------------------------------------------------------------
    # Cache + derived data
    # ---------------------------------------------------------------------

    @property
    def integrated_image(self) -> np.ndarray:
        """Spatial image integrated over all angles and energies (display-mapped)."""
        if self._integrated_image is None:
            self._integrated_image = self.spatial_image_for_selection(self.intensity)
        return self._integrated_image

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._integrated_image = None

    # ---------------------------------------------------------------------
    # Core accessors
    # ---------------------------------------------------------------------

    def get_spectrum_at(
        self,
        position: SpatialPosition,
        integration: IntegrationParams | None = None,
    ) -> np.ndarray:
        """
        Extract ARPES spectrum at spatial position.

        Args:
            position: Spatial position (display indices and physical coords)
            integration: Optional integration parameters

        Returns:
            2D array with shape (n_angle, n_energy)
        """
        y_idx = int(position.y_index)
        x_idx = self.data_x_index(int(position.x_index))

        if integration and integration.enabled:
            y_start = max(0, y_idx - int(integration.y_pixels))
            y_end = min(self.y_axis.size, y_idx + int(integration.y_pixels) + 1)

            x_start = max(0, x_idx - int(integration.x_pixels))
            x_end = min(self.x_axis.size, x_idx + int(integration.x_pixels) + 1)

            return np.sum(self.intensity[y_start:y_end, x_start:x_end, :, :], axis=(0, 1))

        return self.intensity[y_idx, x_idx, :, :]

    def get_spatial_image(self, roi: EnergyAngleROI | None = None) -> np.ndarray:
        """
        Get spatial image, optionally integrated over ROI (display-mapped).

        Note: ROI slicing here is end-exclusive. If you want end-inclusive behavior,
        change slices to `end_idx + 1` consistently across the codebase.
        """
        if roi is None:
            return self.integrated_image

        selection = self.intensity[
            :,
            :,
            roi.angle_start_idx : roi.angle_end_idx,
            roi.energy_start_idx : roi.energy_end_idx,
        ]
        return self.spatial_image_for_selection(selection)

    def position_from_coords(self, x_coord: float, y_coord: float) -> SpatialPosition:
        """Create SpatialPosition from coordinates."""
        x_idx = self.x_axis.find_nearest_index(float(x_coord))
        y_idx = self.y_axis.find_nearest_index(float(y_coord))
        return SpatialPosition(
            x_index=x_idx,
            y_index=y_idx,
            x_coord=float(self.x_axis.values[x_idx]),
            y_coord=float(self.y_axis.values[y_idx]),
        )

    # ---------------------------------------------------------------------
    # Energy helpers
    # ---------------------------------------------------------------------

    def get_kinetic_energy(self, binding_energy: np.ndarray | None = None) -> np.ndarray:
        """
        Get kinetic energy axis.

        If the dataset energy axis is binding energy, convert using photon energy and work function.
        """
        if self.experiment.energy_type == EnergyType.KINETIC:
            return self.energy_axis.values

        if binding_energy is None:
            binding_energy = self.energy_axis.values

        if self.experiment.photon_energy is None:
            raise ValueError("Photon energy required for binding->kinetic conversion")

        return self.experiment.photon_energy - self.experiment.work_function - binding_energy    

    def get_spatial_image_kspace_roi(
        self,
        k_start: float,
        k_end: float,
        e_start: float,
        e_end: float,
        *,
        zero_angle: float = 0.0,
        converter,  # KSpaceConverter
    ) -> np.ndarray:
        """
        Compute spatial image for a ROI defined in (k, energy) coordinates.

        This is the *correct* implementation for k-space mode:
        for each energy bin, compute the corresponding angle limits for [k_start, k_end],
        then integrate intensity over that angle interval.

        Args:
            k_start, k_end: ROI bounds in k (Å⁻¹), can be in any order
            e_start, e_end: ROI bounds in energy axis units (same as dataset.energy_axis.values)
            zero_angle: zero-angle offset used for k conversion alignment
            converter: KSpaceConverter instance (provides k_to_angle)
            fill_value: value used when k is not physically reachable at a given energy

        Returns:
            2D spatial image (display-mapped).
        """
        # Normalize bounds
        k0, k1 = (float(k_start), float(k_end))
        if k0 > k1:
            k0, k1 = k1, k0

        e0_val, e1_val = (float(e_start), float(e_end))
        if e0_val > e1_val:
            e0_val, e1_val = e1_val, e0_val

        # Energy index range (inclusive)
        e_idx0 = self.energy_axis.find_nearest_index(e0_val)
        e_idx1 = self.energy_axis.find_nearest_index(e1_val)
        if e_idx0 > e_idx1:
            e_idx0, e_idx1 = e_idx1, e_idx0

        # Build kinetic energy axis for conversion if needed
        # (k conversion requires kinetic energy > 0)
        if self.experiment.energy_type == EnergyType.KINETIC:
            e_kin = self.energy_axis.values
        else:
            # binding -> kinetic; requires photon energy
            e_kin = self.get_kinetic_energy(self.energy_axis.values)

        # Accumulate in (y, x) internal orientation
        accum_yx = np.zeros((self.n_y, self.n_x), dtype=float)

        angles = self.angle_axis.values

        for ie in range(e_idx0, e_idx1 + 1):
            ek = float(e_kin[ie])
            if ek <= 0:
                continue

            # Compute energy-dependent angle limits for the requested k range.
            # IMPORTANT: angles_shifted = angle - zero_angle in your converter usage,
            # so we invert by adding zero_angle back here.
            try:
                ang0 = float(converter.k_to_angle(k0, ek)) + float(zero_angle)
                ang1 = float(converter.k_to_angle(k1, ek)) + float(zero_angle)
            except ValueError:
                # k not reachable at this energy -> skip (or add fill_value)
                continue

            # Convert to angle indices (inclusive)
            a_idx0 = self.angle_axis.find_nearest_index(ang0)
            a_idx1 = self.angle_axis.find_nearest_index(ang1)
            if a_idx0 > a_idx1:
                a_idx0, a_idx1 = a_idx1, a_idx0

            # Integrate this energy slice over the computed angle range
            # intensity shape: (y, x, angle, energy)
            accum_yx += np.sum(self.intensity[:, :, a_idx0 : a_idx1 + 1, ie], axis=2)

        # Reuse your single-source display mapping (rot90) via the selection helper
        # (y, x, 1, 1) -> spatial_image_for_selection will sum over last two axes and rot90
        return self.spatial_image_for_selection(accum_yx[:, :, None, None])