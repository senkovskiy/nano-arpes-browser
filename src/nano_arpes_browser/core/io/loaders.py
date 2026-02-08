"""File loaders for ARPES data formats."""

from pathlib import Path

import h5py
import numpy as np

from nano_arpes_browser.core.models import ARPESDataset, AxisInfo, EnergyType, ExperimentalParameters


class DataLoader:
    """Load ARPES data from various file formats."""

    # HDF5 paths for ANTARES beamline format
    ANTARES_PATHS = {
        "data": "salsaentry_1/scan_data/data_12",
        "x_spatial": "salsaentry_1/scan_data/actuator_1_1",
        "y_spatial": "salsaentry_1/scan_data/actuator_2_1",
        "energy_offset": "salsaentry_1/scan_data/data_04",
        "energy_step": "salsaentry_1/scan_data/data_05",
        "angle_offset": "salsaentry_1/scan_data/data_07",
        "angle_step": "salsaentry_1/scan_data/data_08",
    }

    @classmethod
    def load_nxs(
        cls,
        filepath: str | Path,
        photon_energy: float | None = None,
        energy_type: EnergyType = EnergyType.KINETIC,
    ) -> ARPESDataset:
        """
        Load NXS (NeXus/HDF5) file from ANTARES beamline.

        Args:
            filepath: Path to .nxs file
            photon_energy: Photon energy in eV (if known)
            energy_type: Whether energy axis is kinetic or binding

        Returns:
            ARPESDataset with loaded data
        """
        filepath = Path(filepath)

        with h5py.File(filepath, "r") as f:
            # Load intensity data - reverse angle axis to get ascending angles
            intensity = np.array(f[cls.ANTARES_PATHS["data"]])[:, :, ::-1, :]

            # Spatial axes
            x_values = np.array(f[cls.ANTARES_PATHS["x_spatial"]][0])
            y_values = np.array(f[cls.ANTARES_PATHS["y_spatial"]])

            # Energy axis
            energy_offset = float(f[cls.ANTARES_PATHS["energy_offset"]][0][0])
            energy_step = float(f[cls.ANTARES_PATHS["energy_step"]][0][0])
            n_energy = intensity.shape[3]
            energy_values = np.linspace(
                energy_offset,
                energy_offset + energy_step * (n_energy - 1),
                n_energy,
            )

            # Angle axis
            angle_offset = float(f[cls.ANTARES_PATHS["angle_offset"]][0][0])
            angle_step = float(f[cls.ANTARES_PATHS["angle_step"]][0][0])
            n_angle = intensity.shape[2]
            angle_values = np.linspace(
                angle_offset,
                angle_offset + angle_step * (n_angle - 1),
                n_angle,
            )

            # Collect metadata
            metadata = {}
            if f.attrs:
                metadata = {k: v for k, v in f.attrs.items()}

        # Create axis info objects
        x_axis = AxisInfo(values=x_values, unit="µm", label="X Position")
        y_axis = AxisInfo(values=y_values, unit="µm", label="Y Position")
        angle_axis = AxisInfo(values=angle_values, unit="°", label="Emission Angle")

        energy_label = (
            "Kinetic Energy" if energy_type == EnergyType.KINETIC else "Binding Energy"
        )
        energy_axis = AxisInfo(values=energy_values, unit="eV", label=energy_label)

        # Experimental parameters
        experiment = ExperimentalParameters(
            photon_energy=photon_energy,
            energy_type=energy_type,
        )

        return ARPESDataset(
            intensity=intensity,
            x_axis=x_axis,
            y_axis=y_axis,
            angle_axis=angle_axis,
            energy_axis=energy_axis,
            experiment=experiment,
            filepath=filepath,
            metadata=metadata,
        )

    @classmethod
    def load(
        cls,
        filepath: str | Path,
        **kwargs,
    ) -> ARPESDataset:
        """
        Load file based on extension.

        Args:
            filepath: Path to data file
            **kwargs: Additional arguments passed to specific loader

        Returns:
            ARPESDataset
        """
        filepath = Path(filepath)
        suffix = filepath.suffix.lower()

        if suffix in (".nxs", ".hdf5", ".h5"):
            return cls.load_nxs(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

