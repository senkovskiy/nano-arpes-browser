"""Export functions for ARPES data."""

import re
from pathlib import Path

import numpy as np


class DataExporter:
    """Export ARPES data to various formats."""

    # ========================================================================
    # CSV Export
    # ========================================================================

    @staticmethod
    def save_csv(
        data: np.ndarray,
        filepath: str | Path,
        delimiter: str = ",",
    ) -> None:
        """Save 2D array to CSV file."""
        filepath = Path(filepath)
        np.savetxt(filepath, data, delimiter=delimiter)

    # ========================================================================
    # Igor Pro Text Format (.itx)
    # ========================================================================

    @staticmethod
    def _sanitize_igor_name(name: str) -> str:
        """Sanitize wave name for Igor Pro (letters, numbers, underscore, max 31 chars)."""
        name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
        if name and not name[0].isalpha():
            name = 'w' + name
        return name[:31]

    @staticmethod
    def _write_1d_wave(f, name: str, data: np.ndarray, label: str = "") -> None:
        """Write a 1D wave to an open file handle."""
        name = DataExporter._sanitize_igor_name(name)
        f.write(f"WAVES/N=({len(data)})/D {name}\n")
        f.write("BEGIN\n")
        for val in data:
            f.write(f"\t{float(val):.6g}\n")
        f.write("END\n")
        if label:
            label_clean = label.replace('"', "'")
            f.write(f'X SetScale d, 0, 0, "{label_clean}", {name}\n')

    @staticmethod
    def _write_2d_wave(
        f,
        name: str,
        data: np.ndarray,
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
        x_label: str = "",
        y_label: str = "",
        z_label: str = "",
    ) -> None:
        """Write a 2D wave to an open file handle."""
        name = DataExporter._sanitize_igor_name(name)
        rows, cols = data.shape

        f.write(f"WAVES/N=({rows},{cols})/D {name}\n")
        f.write("BEGIN\n")
        for i in range(rows):
            row_data = "\t".join(f"{float(val):.6g}" for val in data[i, :])
            f.write(f"\t{row_data}\n")
        f.write("END\n")

        if x_axis is not None and len(x_axis) >= 2:
            x_start = float(x_axis[0])
            x_delta = float((x_axis[-1] - x_axis[0]) / (len(x_axis) - 1))
            x_lbl = x_label.replace('"', "'") if x_label else ""
            f.write(f'X SetScale/P x, {x_start}, {x_delta}, "{x_lbl}", {name}\n')

        if y_axis is not None and len(y_axis) >= 2:
            y_start = float(y_axis[0])
            y_delta = float((y_axis[-1] - y_axis[0]) / (len(y_axis) - 1))
            y_lbl = y_label.replace('"', "'") if y_label else ""
            f.write(f'X SetScale/P y, {y_start}, {y_delta}, "{y_lbl}", {name}\n')

        if z_label:
            z_lbl = z_label.replace('"', "'")
            f.write(f'X SetScale d, 0, 0, "{z_lbl}", {name}\n')

    @staticmethod
    def _write_4d_wave(
        f,
        name: str,
        data: np.ndarray,
        dim0_axis: np.ndarray | None = None,
        dim1_axis: np.ndarray | None = None,
        dim2_axis: np.ndarray | None = None,
        dim3_axis: np.ndarray | None = None,
        dim0_label: str = "",
        dim1_label: str = "",
        dim2_label: str = "",
        dim3_label: str = "",
    ) -> None:
        """
        Write a 4D wave to an open file handle.

        Igor Pro 4D wave dimensions: (x, y, z, t) = (d0, d1, d2, d3)
        For ARPES: (y_spatial, x_spatial, angle, energy)
        """
        name = DataExporter._sanitize_igor_name(name)
        d0, d1, d2, d3 = data.shape

        f.write(f"WAVES/N=({d0},{d1},{d2},{d3})/D {name}\n")
        f.write("BEGIN\n")

        # Igor expects column-major (Fortran) order
        flat_data = data.flatten(order='F')

        # Write in rows for readability
        values_per_line = 10
        for i in range(0, len(flat_data), values_per_line):
            chunk = flat_data[i:i + values_per_line]
            line = "\t".join(f"{float(v):.6g}" for v in chunk)
            f.write(f"\t{line}\n")

        f.write("END\n")

        # Set scaling for each dimension (x=dim0, y=dim1, z=dim2, t=dim3)
        if dim0_axis is not None and len(dim0_axis) >= 2:
            start = float(dim0_axis[0])
            delta = float((dim0_axis[-1] - dim0_axis[0]) / (len(dim0_axis) - 1))
            lbl = dim0_label.replace('"', "'")
            f.write(f'X SetScale/P x, {start}, {delta}, "{lbl}", {name}\n')

        if dim1_axis is not None and len(dim1_axis) >= 2:
            start = float(dim1_axis[0])
            delta = float((dim1_axis[-1] - dim1_axis[0]) / (len(dim1_axis) - 1))
            lbl = dim1_label.replace('"', "'")
            f.write(f'X SetScale/P y, {start}, {delta}, "{lbl}", {name}\n')

        if dim2_axis is not None and len(dim2_axis) >= 2:
            start = float(dim2_axis[0])
            delta = float((dim2_axis[-1] - dim2_axis[0]) / (len(dim2_axis) - 1))
            lbl = dim2_label.replace('"', "'")
            f.write(f'X SetScale/P z, {start}, {delta}, "{lbl}", {name}\n')

        if dim3_axis is not None and len(dim3_axis) >= 2:
            start = float(dim3_axis[0])
            delta = float((dim3_axis[-1] - dim3_axis[0]) / (len(dim3_axis) - 1))
            lbl = dim3_label.replace('"', "'")
            f.write(f'X SetScale/P t, {start}, {delta}, "{lbl}", {name}\n')

    # ========================================================================
    # Public Export Methods
    # ========================================================================

    @staticmethod
    def save_itx(
        data: np.ndarray,
        filepath: str | Path,
        wave_name: str = "data",
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
        x_label: str = "",
        y_label: str = "",
        z_label: str = "Intensity",
        note: str = "",
    ) -> None:
        """Save 2D data as Igor Pro Text file (.itx)."""
        filepath = Path(filepath)
        wave_name = DataExporter._sanitize_igor_name(wave_name)

        with open(filepath, "w", newline='\n') as f:
            f.write("IGOR\n")

            DataExporter._write_2d_wave(
                f, wave_name, data,
                x_axis=x_axis, y_axis=y_axis,
                x_label=x_label, y_label=y_label, z_label=z_label
            )

            if note:
                note_escaped = note.replace('"', "'").replace('\n', '\\r')
                f.write(f'X Note {wave_name}, "{note_escaped}"\n')

    @staticmethod
    def save_itx_with_axes(
        data: np.ndarray,
        filepath: str | Path,
        wave_name: str = "data",
        x_axis: np.ndarray | None = None,
        y_axis: np.ndarray | None = None,
        x_label: str = "X",
        y_label: str = "Y",
        z_label: str = "Intensity",
    ) -> None:
        """Save 2D data with separate axis waves."""
        filepath = Path(filepath)
        wave_name = DataExporter._sanitize_igor_name(wave_name)

        with open(filepath, "w", newline='\n') as f:
            f.write("IGOR\n")

            # Main data wave
            DataExporter._write_2d_wave(
                f, wave_name, data,
                x_axis=x_axis, y_axis=y_axis,
                x_label=x_label, y_label=y_label, z_label=z_label
            )

            # Axis waves
            if x_axis is not None:
                f.write("\n")
                DataExporter._write_1d_wave(f, f"{wave_name}_x", x_axis, label=x_label)

            if y_axis is not None:
                f.write("\n")
                DataExporter._write_1d_wave(f, f"{wave_name}_y", y_axis, label=y_label)

    @staticmethod
    def save_region_itx(
        data: np.ndarray,
        filepath: str | Path,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        angle_axis: np.ndarray,
        energy_axis: np.ndarray,
        x_unit: str = "µm",
        y_unit: str = "µm",
        angle_unit: str = "°",
        energy_unit: str = "eV",
        center_x: float = 0,
        center_y: float = 0,
    ) -> None:
        """
        Export a selected region of the 4D dataset to Igor Pro format.

        Args:
            data: 4D array (y, x, angle, energy)
            filepath: Output path
            x_axis, y_axis: Spatial axes for the region
            angle_axis, energy_axis: Spectral axes
            x_unit, y_unit, angle_unit, energy_unit: Axis units
            center_x, center_y: Center position of the region
        """
        filepath = Path(filepath)

        with open(filepath, "w", newline='\n') as f:
            f.write("IGOR\n")

            # --- 4D region data ---
            DataExporter._write_4d_wave(
                f,
                "region_4d",
                data,
                dim0_axis=y_axis,
                dim1_axis=x_axis,
                dim2_axis=angle_axis,
                dim3_axis=energy_axis,
                dim0_label=f"Y ({y_unit})",
                dim1_label=f"X ({x_unit})",
                dim2_label=f"Angle ({angle_unit})",
                dim3_label=f"Energy ({energy_unit})",
            )
            f.write("\n")

            # --- Integrated spatial map of region ---
            spatial_map = np.rot90(np.sum(data, axis=(2, 3)))
            DataExporter._write_2d_wave(
                f,
                "region_spatial",
                spatial_map,
                x_axis=x_axis,
                y_axis=y_axis,
                x_label=f"X ({x_unit})",
                y_label=f"Y ({y_unit})",
                z_label="Intensity",
            )
            f.write("\n")

            # --- Integrated spectrum of entire region ---
            integrated_spectrum = np.sum(data, axis=(0, 1))
            DataExporter._write_2d_wave(
                f,
                "region_spectrum",
                integrated_spectrum,
                x_axis=angle_axis,
                y_axis=energy_axis,
                x_label=f"Angle ({angle_unit})",
                y_label=f"Energy ({energy_unit})",
                z_label="Intensity",
            )
            f.write("\n")

            # --- Axis waves ---
            DataExporter._write_1d_wave(f, "region_x", x_axis, label=x_unit)
            f.write("\n")
            DataExporter._write_1d_wave(f, "region_y", y_axis, label=y_unit)
            f.write("\n")
            DataExporter._write_1d_wave(f, "region_angle", angle_axis, label=angle_unit)
            f.write("\n")
            DataExporter._write_1d_wave(f, "region_energy", energy_axis, label=energy_unit)
            f.write("\n")

            # --- Center position ---
            f.write("WAVES/N=(2)/D region_center\n")
            f.write("BEGIN\n")
            f.write(f"\t{center_x:.6g}\n")
            f.write(f"\t{center_y:.6g}\n")
            f.write("END\n")
            f.write(f'X SetScale d, 0, 0, "{x_unit}", region_center\n')

    @staticmethod
    def save_full_dataset_itx(
        dataset,  # ARPESDataset
        filepath: str | Path,
        include_4d_data: bool = True,
        max_file_size_gb: float = 2.0,
    ) -> None:
        """
        Export full ARPES dataset to Igor Pro format.

        Creates waves:
        - arpes_4d: Full 4D data cube (y_spatial, x_spatial, angle, energy)
        - spatial_map: Integrated spatial image
        - x_spatial, y_spatial: Spatial axis values
        - angle_axis, energy_axis: Spectral axis values

        Args:
            dataset: ARPESDataset object
            filepath: Output path
            include_4d_data: Whether to include full 4D data
            max_file_size_gb: Maximum file size limit for 4D export
        """
        filepath = Path(filepath)

        # Estimate size
        data_size_gb = dataset.intensity.nbytes / (1024**3)
        will_include_4d = include_4d_data and (data_size_gb <= max_file_size_gb)

        with open(filepath, "w", newline='\n') as f:
            f.write("IGOR\n")

            # --- Full 4D data ---
            if will_include_4d:
                DataExporter._write_4d_wave(
                    f,
                    "arpes_4d",
                    dataset.intensity,
                    dim0_axis=dataset.y_axis.values,
                    dim1_axis=dataset.x_axis.values,
                    dim2_axis=dataset.angle_axis.values,
                    dim3_axis=dataset.energy_axis.values,
                    dim0_label=f"Y ({dataset.y_axis.unit})",
                    dim1_label=f"X ({dataset.x_axis.unit})",
                    dim2_label=f"Angle ({dataset.angle_axis.unit})",
                    dim3_label=f"Energy ({dataset.energy_axis.unit})",
                )
                f.write("\n")

            # --- Integrated spatial map ---
            DataExporter._write_2d_wave(
                f,
                "spatial_map",
                dataset.integrated_image,
                x_axis=dataset.x_axis.values,
                y_axis=dataset.y_axis.values,
                x_label=f"X ({dataset.x_axis.unit})",
                y_label=f"Y ({dataset.y_axis.unit})",
                z_label="Intensity",
            )
            f.write("\n")

            # --- Axis waves ---
            DataExporter._write_1d_wave(
                f, "x_spatial", dataset.x_axis.values, label=dataset.x_axis.unit
            )
            f.write("\n")

            DataExporter._write_1d_wave(
                f, "y_spatial", dataset.y_axis.values, label=dataset.y_axis.unit
            )
            f.write("\n")

            DataExporter._write_1d_wave(
                f, "angle_axis", dataset.angle_axis.values, label=dataset.angle_axis.unit
            )
            f.write("\n")

            DataExporter._write_1d_wave(
                f, "energy_axis", dataset.energy_axis.values, label=dataset.energy_axis.unit
            )

        # Return info about what was exported
        return {
            "filepath": filepath,
            "included_4d": will_include_4d,
            "data_size_gb": data_size_gb,
            "shape": dataset.intensity.shape,
        }

    # ========================================================================
    # Filename Generation
    # ========================================================================

    @staticmethod
    def generate_spatial_filename(
        angle_range: tuple[float, float],
        energy_range: tuple[float, float],
        k_space: bool = False,
        k_range: tuple[float, float] | None = None,
        extension: str = "csv",
    ) -> str:
        """Generate descriptive filename for spatial image export."""
        if k_space and k_range is not None:
            base = (
                f"Spatial_k_{k_range[0]:.2f}_{k_range[1]:.2f}__"
                f"En_{energy_range[0]:.2f}_{energy_range[1]:.2f}"
            )
        else:
            base = (
                f"Spatial_Ang_{angle_range[0]:.1f}_{angle_range[1]:.1f}__"
                f"En_{energy_range[0]:.2f}_{energy_range[1]:.2f}"
            )
        return f"{base}.{extension}"

    @staticmethod
    def generate_arpes_filename(
        x_coord: float,
        y_coord: float,
        extension: str = "csv",
    ) -> str:
        """Generate descriptive filename for ARPES spectrum export."""
        return f"ARPES_X_{x_coord:.1f}__Y_{y_coord:.1f}.{extension}"
