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

    photon_energy: float | None = Field(
        default=None, gt=0, description="Photon energy in eV"
    )
    work_function: float = Field(
        default=4.5, gt=0, lt=10, description="Analyzer work function in eV"
    )
    temperature: float | None = Field(
        default=None, ge=0, description="Sample temperature in K"
    )
    polarization: str | None = Field(
        default=None, description="Light polarization (LH, LV, CR, CL)"
    )
    energy_type: EnergyType = Field(
        default=EnergyType.KINETIC, description="Type of energy axis"
    )

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
    angle_offset: float = Field(
        default=0.0, description="Additional angle offset for alignment"
    )


class ARPESDataset(BaseModel):
    """
    Container for micro-ARPES dataset.

    Data shape convention: (n_y, n_x, n_angle, n_energy)

    Physical conventions:
    - Positive angles = emission towards analyzer
    - Kinetic energy increases with index
    - Binding energy = photon_energy - work_function - kinetic_energy
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    # Raw intensity data
    intensity: np.ndarray = Field(
        description="4D intensity array (y, x, angle, energy)"
    )

    # Axes
    x_axis: AxisInfo
    y_axis: AxisInfo
    angle_axis: AxisInfo
    energy_axis: AxisInfo

    # Experimental parameters
    experiment: ExperimentalParameters = Field(default_factory=ExperimentalParameters)

    # File info
    filepath: Path | None = None
    metadata: dict = Field(default_factory=dict)

    # Cache
    _integrated_image: np.ndarray | None = None

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
        shape = self.intensity.shape
        expected = {
            "y_axis": shape[0],
            "x_axis": shape[1],
            "angle_axis": shape[2],
            "energy_axis": shape[3],
        }

        for axis_name, expected_size in expected.items():
            axis = getattr(self, axis_name)
            if axis.size != expected_size:
                raise ValueError(
                    f"{axis_name} size {axis.size} doesn't match "
                    f"intensity dimension {expected_size}"
                )
        return self

    @property
    def shape(self) -> tuple[int, int, int, int]:
        """Return (n_y, n_x, n_angle, n_energy)."""
        return tuple(self.intensity.shape)

    @property
    def integrated_image(self) -> np.ndarray:
        """Spatial image integrated over all angles and energies."""
        if self._integrated_image is None:
            self._integrated_image = np.rot90(np.sum(self.intensity, axis=(2, 3)))
        return self._integrated_image

    def clear_cache(self) -> None:
        """Clear cached computations."""
        self._integrated_image = None

    def get_spectrum_at(
        self,
        position: SpatialPosition,
        integration: IntegrationParams | None = None,
    ) -> np.ndarray:
        """
        Extract ARPES spectrum at spatial position.

        Args:
            position: Spatial position (indices and coordinates)
            integration: Optional integration parameters

        Returns:
            2D array (angle, energy)
        """
        y_idx = position.y_index
        # X axis is stored reversed in original data format
        x_idx = self.x_axis.size - 1 - position.x_index

        if integration and integration.enabled:
            y_start = max(0, y_idx - integration.y_pixels)
            y_end = min(self.y_axis.size, y_idx + integration.y_pixels + 1)
            x_start = max(0, x_idx - integration.x_pixels)
            x_end = min(self.x_axis.size, x_idx + integration.x_pixels + 1)

            return np.sum(
                self.intensity[y_start:y_end, x_start:x_end, :, :], axis=(0, 1)
            )

        return self.intensity[y_idx, x_idx, :, :]

    def get_spatial_image(self, roi: EnergyAngleROI | None = None) -> np.ndarray:
        """
        Get spatial image, optionally integrated over ROI.

        Args:
            roi: Optional region of interest in angle-energy space

        Returns:
            2D spatial image
        """
        if roi is None:
            return self.integrated_image

        selection = self.intensity[
            :,
            :,
            roi.angle_start_idx : roi.angle_end_idx,
            roi.energy_start_idx : roi.energy_end_idx,
        ]
        return np.rot90(np.sum(selection, axis=(2, 3)))

    def position_from_coords(self, x_coord: float, y_coord: float) -> SpatialPosition:
        """Create SpatialPosition from coordinates."""
        x_idx = self.x_axis.find_nearest_index(x_coord)
        y_idx = self.y_axis.find_nearest_index(y_coord)
        return SpatialPosition(
            x_index=x_idx,
            y_index=y_idx,
            x_coord=float(self.x_axis.values[x_idx]),
            y_coord=float(self.y_axis.values[y_idx]),
        )

    def get_kinetic_energy(
        self, binding_energy: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Get kinetic energy axis.

        If data is in binding energy, converts using photon energy and work function.
        """
        if self.experiment.energy_type == EnergyType.KINETIC:
            return self.energy_axis.values

        if binding_energy is None:
            binding_energy = self.energy_axis.values

        if self.experiment.photon_energy is None:
            raise ValueError("Photon energy required for binding->kinetic conversion")

        return (
            self.experiment.photon_energy
            - self.experiment.work_function
            - binding_energy
        )
