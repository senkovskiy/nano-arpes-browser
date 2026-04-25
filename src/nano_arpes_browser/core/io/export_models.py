"""Models for export payloads."""

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class RegionExportData(BaseModel):
    """Payload for selected-region Igor export."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: np.ndarray
    spatial_map: np.ndarray
    x_axis: np.ndarray
    y_axis: np.ndarray
    angle_axis: np.ndarray
    energy_axis: np.ndarray
    x_unit: str = "µm"
    y_unit: str = "µm"
    angle_unit: str = "°"
    energy_unit: str = "eV"
    center_x: float = 0.0
    center_y: float = 0.0

    @model_validator(mode="after")
    def validate_shapes(self) -> "RegionExportData":
        """Validate region data and axis dimensions."""
        if self.data.ndim != 4:
            raise ValueError("data must be 4D with shape (y, x, angle, energy)")
        if self.spatial_map.ndim != 2:
            raise ValueError("spatial_map must be 2D")

        n_y, n_x, n_angle, n_energy = self.data.shape
        axis_sizes = {
            "x_axis": (self.x_axis, n_x),
            "y_axis": (self.y_axis, n_y),
            "angle_axis": (self.angle_axis, n_angle),
            "energy_axis": (self.energy_axis, n_energy),
        }

        for name, (axis, expected_size) in axis_sizes.items():
            if axis.ndim != 1:
                raise ValueError(f"{name} must be 1D")
            if axis.size != expected_size:
                raise ValueError(f"{name} length must match data dimension {expected_size}")

        return self
