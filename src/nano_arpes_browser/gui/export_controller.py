"""Export workflow controller for the main window."""

from pathlib import Path
from typing import Any

from PyQt6.QtWidgets import QMessageBox

from nano_arpes_browser.core.io import DataExporter


class ExportController:
    """Coordinate export dialogs, data gathering, and exporter calls."""

    def __init__(self, window: Any):
        self.window = window

    def save_spatial(self, format: str = "csv") -> None:
        """Save the current spatial image."""
        window = self.window
        if not window._require_dataset("No Data", "Please load a dataset first."):
            return

        if window.current_roi and window.current_roi.angle_start is not None:
            base_filename = DataExporter.generate_spatial_filename(
                (window.current_roi.angle_start, window.current_roi.angle_end or 0),
                (window.current_roi.energy_start or 0, window.current_roi.energy_end or 0),
                extension=format,
            )
        else:
            base_filename = f"spatial_integrated.{format}"

        if format == "csv":
            filter_str = "CSV Files (*.csv);;All Files (*)"
        else:
            filter_str = "Igor Text Files (*.itx);;All Files (*)"

        filepath = window._get_save_filepath(
            "Save Spatial Image",
            str(Path.home() / base_filename),
            filter_str,
        )
        if filepath is None:
            return

        image = window.spatial_viewer.get_current_image()
        if image is None:
            return

        try:
            if format == "csv":
                DataExporter.save_csv(image, filepath)
            else:
                DataExporter.save_spatial_itx(
                    image,
                    filepath,
                    x_axis=window.dataset.x_axis.values,
                    y_axis=window.dataset.y_axis.values,
                    x_unit=window.dataset.x_axis.unit,
                    y_unit=window.dataset.y_axis.unit,
                )
            window._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(window, "Export Error", f"Failed to save:\n{e}")

    def save_arpes(self, format: str = "csv") -> None:
        """Save the current ARPES spectrum."""
        window = self.window
        if not window._require_dataset("No Data", "Please load a dataset first."):
            return

        if not window._require_position("No Position", "Please select a position first."):
            return

        base_filename = DataExporter.generate_arpes_filename(
            window.current_position.x_coord,
            window.current_position.y_coord,
            extension=format,
        )

        if format == "csv":
            filter_str = "CSV Files (*.csv);;All Files (*)"
        else:
            filter_str = "Igor Text Files (*.itx);;All Files (*)"

        filepath = window._get_save_filepath(
            "Save ARPES Spectrum",
            str(Path.home() / base_filename),
            filter_str,
        )
        if filepath is None:
            return

        spectrum = window.arpes_viewer.get_current_data()
        x_axis, energy_axis = window.arpes_viewer.get_current_axes()

        if spectrum is None or x_axis is None or energy_axis is None:
            return

        try:
            if format == "csv":
                DataExporter.save_csv(spectrum, filepath)
            else:
                k_params = window.control_panel.get_kspace_params()
                if k_params.enabled:
                    x_label = "k (Å⁻¹)"
                else:
                    x_label = f"Angle ({window.dataset.angle_axis.unit})"

                DataExporter.save_arpes_itx(
                    spectrum,
                    filepath,
                    x_axis=x_axis,
                    energy_axis=energy_axis,
                    x_label=x_label,
                    energy_unit=window.dataset.energy_axis.unit,
                )
            window._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(window, "Export Error", f"Failed to save:\n{e}")

    def save_region_igor(self) -> None:
        """Save selected spatial region as Igor .itx."""
        window = self.window
        if not window._require_dataset("No Data", "Please load a dataset first."):
            return

        if not window._require_position("No Position", "Please select a position on the map."):
            return

        integration = window.control_panel.get_integration_params()

        if not integration.enabled or (integration.x_pixels == 0 and integration.y_pixels == 0):
            reply = QMessageBox.question(
                window,
                "Select Region",
                "Integration is not enabled.\n\n"
                "Enable integration and set X/Y pixels to define the region to export.\n\n"
                "Export single spectrum at current position instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.save_arpes(format="itx")
            return

        region_data, x_axis_region, y_axis_region = window.dataset.extract_region(
            window.current_position,
            integration,
        )

        ny, nx, n_angle, n_energy = region_data.shape
        region_size_mb = region_data.nbytes / (1024 * 1024)

        msg = (
            f"Export selected region?\n\n"
            f"Center: X={window.current_position.x_coord:.1f}, "
            f"Y={window.current_position.y_coord:.1f} µm\n"
            f"Region: {nx} x {ny} spatial points\n"
            f"Spectra: {n_angle} x {n_energy} (angle x energy)\n\n"
            f"Total: {nx}x{ny}x{n_angle}x{n_energy} = "
            f"{nx * ny * n_angle * n_energy:,} points\n"
            f"Estimated size: {region_size_mb:.1f} MB"
        )

        reply = QMessageBox.question(
            window,
            "Export Region",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        default_name = (
            f"region_X{window.current_position.x_coord:.0f}_"
            f"Y{window.current_position.y_coord:.0f}_"
            f"{nx}x{ny}.itx"
        )

        filepath = window._get_save_filepath(
            "Save Region (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )
        if filepath is None:
            return

        window._start_busy_operation("Exporting region...")

        try:
            DataExporter.save_region_itx(
                region_data,
                filepath,
                x_axis=x_axis_region,
                y_axis=y_axis_region,
                angle_axis=window.dataset.angle_axis.values,
                energy_axis=window.dataset.energy_axis.values,
                x_unit=window.dataset.x_axis.unit,
                y_unit=window.dataset.y_axis.unit,
                angle_unit=window.dataset.angle_axis.unit,
                energy_unit=window.dataset.energy_axis.unit,
                center_x=window.current_position.x_coord,
                center_y=window.current_position.y_coord,
            )

            window._set_status(f"Saved: {Path(filepath).name}")

            size_mb = Path(filepath).stat().st_size / (1024 * 1024)

            QMessageBox.information(
                window,
                "Export Complete",
                f"Region exported!\n\n"
                f"File: {Path(filepath).name}\n"
                f"Size: {size_mb:.1f} MB\n\n"
                f"Waves: region_4d, spatial_map, axes",
            )

        except Exception as e:
            QMessageBox.critical(window, "Export Error", f"Failed to save:\n{e}")
        finally:
            window._finish_busy_operation()

    def save_full_igor(self) -> None:
        """Save the full dataset as Igor .itx."""
        window = self.window
        if not window._require_dataset("No Data", "Please load a dataset first."):
            return

        data_size_gb = window.dataset.intensity.nbytes / (1024**3)
        shape = window.dataset.intensity.shape

        msg = (
            f"Export full 4D dataset?\n\n"
            f"Data shape: {shape[0]}x{shape[1]}x{shape[2]}x{shape[3]}\n"
            f"(Y x X x Angle x Energy)\n\n"
            f"Estimated size: {data_size_gb:.2f} GB\n\n"
        )

        if data_size_gb > 2.0:
            msg += "⚠️ Large file! Export may take several minutes."

        reply = QMessageBox.question(
            window,
            "Export Full Dataset",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        default_name = "full_dataset.itx"
        if window.dataset.filepath:
            default_name = window.dataset.filepath.stem + "_full.itx"

        filepath = window._get_save_filepath(
            "Save Full Dataset (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )
        if filepath is None:
            return

        window._start_busy_operation("Exporting full dataset...")

        try:
            result = DataExporter.save_full_dataset_itx(
                window.dataset,
                filepath,
                include_4d_data=True,
                max_file_size_gb=10.0,
            )

            window._set_status(f"Saved: {Path(filepath).name}")

            size_mb = Path(filepath).stat().st_size / (1024 * 1024)

            info_msg = (
                f"Export complete!\n\n"
                f"File: {Path(filepath).name}\n"
                f"Size: {size_mb:.1f} MB\n\n"
                f"Waves created:\n"
            )

            if result.get("included_4d", False):
                info_msg += f"• arpes_4d: {shape} - Full 4D data\n"

            info_msg += (
                f"• spatial_map: Integrated image\n"
                f"• x_spatial, y_spatial: Spatial axes\n"
                f"• angle_axis, energy_axis: Spectral axes\n\n"
                f"In Igor Pro:\n"
                f'LoadWave/T "{filepath}"'
            )

            QMessageBox.information(window, "Export Complete", info_msg)

        except Exception as e:
            QMessageBox.critical(window, "Export Error", f"Failed to save:\n{e}")
        finally:
            window._finish_busy_operation()
