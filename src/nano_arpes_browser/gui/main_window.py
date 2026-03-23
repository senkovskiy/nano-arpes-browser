"""Main application window."""

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from nano_arpes_browser.core.io import DataExporter, DataLoader
from nano_arpes_browser.core.models import ARPESDataset, EnergyAngleROI, SpatialPosition
from nano_arpes_browser.core.processing.kspace import KSpaceConverter
from nano_arpes_browser.gui.styles import DARK_THEME, LIGHT_THEME, get_pyqtgraph_config
from nano_arpes_browser.gui.widgets import ARPESViewer, ControlPanel, InfoPanel, SpatialViewer


class MainWindow(QMainWindow):
    """Main application window for ARPES data visualization."""

    def __init__(self):
        super().__init__()

        self.dataset: ARPESDataset | None = None
        self.current_position: SpatialPosition | None = None
        self.current_roi: EnergyAngleROI | None = None
        self.k_converter = KSpaceConverter()
        self._dark_theme = True

        self.settings = QSettings("NanoARPES", "Browser")

        pg_config = get_pyqtgraph_config(dark=True)
        pg.setConfigOptions(**pg_config)

        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_state()

        self.setStyleSheet(DARK_THEME)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Nano-ARPES Browser")
        self.setMinimumSize(1200, 700)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        left_panel = QWidget()
        left_panel.setFixedWidth(240)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        self.control_panel = ControlPanel()
        self.info_panel = InfoPanel()

        left_layout.addWidget(self.control_panel)
        left_layout.addWidget(self.info_panel)
        left_layout.addStretch()

        main_layout.addWidget(left_panel)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.spatial_viewer = SpatialViewer()
        self.main_splitter.addWidget(self.spatial_viewer)

        self.arpes_viewer = ARPESViewer()
        self.main_splitter.addWidget(self.arpes_viewer)

        self.main_splitter.setSizes([500, 500])

        main_layout.addWidget(self.main_splitter, stretch=1)

    def _setup_menu(self) -> None:
        """Set up menu bar."""
        menubar = self.menuBar()

        # === File Menu ===
        file_menu = menubar.addMenu("&File")

        open_action = QAction("&Open...", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.setStatusTip("Open ARPES data file")
        open_action.triggered.connect(self._on_load_data)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        export_menu = file_menu.addMenu("&Export")

        export_menu.addAction("Spatial Map (CSV)...", lambda: self._save_spatial("csv"))
        export_menu.addAction("Spatial Map (Igor)...", lambda: self._save_spatial("itx"))
        export_menu.addSeparator()
        export_menu.addAction("Spectrum (CSV)...", lambda: self._save_arpes("csv"))
        export_menu.addAction("Spectrum (Igor)...", lambda: self._save_arpes("itx"))

        export_menu.addSeparator()
        export_menu.addAction("Selected Region (Igor)...", self._on_save_region_igor)
        export_menu.addSeparator()
        export_menu.addAction("Full Dataset (Igor)...", self._on_save_full_igor)
        file_menu.addSeparator()

        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # === View Menu ===
        view_menu = menubar.addMenu("&View")

        reset_view_action = QAction("&Reset Views", self)
        reset_view_action.setShortcut("Ctrl+R")
        reset_view_action.triggered.connect(self._reset_views)
        view_menu.addAction(reset_view_action)

        view_menu.addSeparator()

        self.theme_action = QAction("&Light Theme", self, checkable=True)
        self.theme_action.triggered.connect(self._toggle_theme)
        view_menu.addAction(self.theme_action)

        # === Help Menu ===
        help_menu = menubar.addMenu("&Help")

        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _setup_statusbar(self) -> None:
        """Set up status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)

        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        self.position_label = QLabel("")
        self.position_label.setMinimumWidth(200)
        self.statusbar.addPermanentWidget(self.position_label)

        self.memory_label = QLabel("")
        self.memory_label.setMinimumWidth(100)
        self.statusbar.addPermanentWidget(self.memory_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(150)
        self.progress_bar.hide()
        self.statusbar.addPermanentWidget(self.progress_bar)

    def _connect_signals(self) -> None:
        """Connect widget signals."""
        # Control panel - file operations
        self.control_panel.load_requested.connect(self._on_load_data)

        # Control panel - export operations
        self.control_panel.export_map_csv_requested.connect(
            lambda: self._save_spatial(format="csv")
        )
        self.control_panel.export_map_igor_requested.connect(
            lambda: self._save_spatial(format="itx")
        )
        self.control_panel.export_spectrum_csv_requested.connect(
            lambda: self._save_arpes(format="csv")
        )
        self.control_panel.export_spectrum_igor_requested.connect(
            lambda: self._save_arpes(format="itx")
        )
        self.control_panel.export_region_igor_requested.connect(self._on_save_region_igor)
        self.control_panel.export_full_igor_requested.connect(self._on_save_full_igor)

        # Control panel - other
        self.control_panel.k_space_changed.connect(self._on_kspace_changed)
        self.control_panel.integration_changed.connect(self._on_integration_changed)
        self.control_panel.display_settings_changed.connect(self._on_display_settings_changed)

        # Spatial viewer
        self.spatial_viewer.position_changed.connect(self._on_spatial_position_changed)

        # ARPES viewer
        self.arpes_viewer.roi_changed.connect(self._on_roi_changed)

    def _restore_state(self) -> None:
        """Restore window state from settings."""
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)

        splitter_state = self.settings.value("splitterState")
        if splitter_state:
            self.main_splitter.restoreState(splitter_state)

    def _save_state(self) -> None:
        """Save window state to settings."""
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.setValue("splitterState", self.main_splitter.saveState())

    # =========================================================================
    # Data Loading
    # =========================================================================

    def _on_load_data(self) -> None:
        """Handle data loading."""
        last_dir = self.settings.value("LastDir", str(Path.home()))

        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open ARPES Data",
            last_dir,
            "NeXus Files (*.nxs);;HDF5 Files (*.h5 *.hdf5);;All Files (*)",
        )

        if not filepath:
            return

        self._load_file(Path(filepath))

    def _load_file(self, filepath: Path) -> None:
        """Load data file."""
        self._show_progress("Loading data...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        try:
            self.dataset = DataLoader.load(filepath)
            self.settings.setValue("LastDir", str(filepath.parent))

            self._initialize_display()

            self.setWindowTitle(f"Nano-ARPES Browser — {filepath.name}")
            self._set_status(f"Loaded: {filepath.name}")
            self._update_memory_label()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load file:\n{e}")
            self._set_status("Load failed")

        finally:
            QApplication.restoreOverrideCursor()
            self._hide_progress()

    def _initialize_display(self) -> None:
        """Initialize display with loaded data."""
        if self.dataset is None:
            return

        # Update info panel
        self.info_panel.set_dataset_info(self.dataset)

        # Reset control panel
        self.control_panel.reset()
        self.control_panel.set_angle_range(
            self.dataset.angle_axis.min,
            self.dataset.angle_axis.max,
        )

        # Initialize spatial viewer
        self.spatial_viewer.set_data(
            self.dataset.integrated_image,
            self.dataset.x_axis.values,
            self.dataset.y_axis.values,
        )

        # Set initial position to center
        center_x = self.dataset.x_axis.values[self.dataset.x_axis.size // 2]
        center_y = self.dataset.y_axis.values[self.dataset.y_axis.size // 2]
        self.spatial_viewer.set_position(center_x, center_y)

        # Initialize ARPES viewer axes
        self.arpes_viewer.set_axes(
            self.dataset.angle_axis.values,
            self.dataset.energy_axis.values,
            x_label=self.dataset.angle_axis.label,
            y_label=self.dataset.energy_axis.label,
        )

    # =========================================================================
    # Position & ROI Updates
    # =========================================================================

    def _on_spatial_position_changed(self, x_coord: float, y_coord: float) -> None:
        """Handle spatial position change."""
        if self.dataset is None:
            return

        self.current_position = self.dataset.position_from_coords(x_coord, y_coord)

        self._update_spectrum_for_current_position()

    def _update_spectrum_for_current_position(self) -> None:
        """Update ARPES spectrum and related UI for the current position."""
        if self.dataset is None or self.current_position is None:
            return

        integration = self.control_panel.get_integration_params()

        spectrum = self.dataset.get_spectrum_at(self.current_position, integration)

        k_params = self.control_panel.get_kspace_params()

        if k_params.enabled:
            result = self.k_converter.convert_spectrum(
                spectrum,
                self.dataset.energy_axis.values,
                self.dataset.angle_axis.values,
                zero_angle=k_params.zero_angle,
            )
            self.arpes_viewer.set_axes(
                result.k_axis,
                result.energy_axis,
                x_label="k∥ (Å⁻¹)",
                y_label=self.dataset.energy_axis.label,
            )
            self.arpes_viewer.set_data(
                result.spectrum,
                auto_levels=not self.control_panel.is_range_locked(),
            )
        else:
            self.arpes_viewer.set_axes(
                self.dataset.angle_axis.values,
                self.dataset.energy_axis.values,
                x_label=self.dataset.angle_axis.label,
                y_label=self.dataset.energy_axis.label,
            )
            self.arpes_viewer.set_data(
                spectrum,
                auto_levels=not self.control_panel.is_range_locked(),
            )

        self.spatial_viewer.set_position_title(
            self.current_position.x_coord,
            self.current_position.y_coord,
        )

        self._update_position_label()

        if integration.enabled:
            pixel_x = self.dataset.x_axis.step
            pixel_y = self.dataset.y_axis.step
            self.spatial_viewer.show_integration_rect(
                integration.x_pixels, integration.y_pixels, pixel_x, pixel_y
            )
        else:
            self.spatial_viewer.hide_integration_rect()

    def _on_roi_changed(self, x_start: float, x_end: float, e_start: float, e_end: float) -> None:
        """Handle ROI changes in the ARPES viewer."""
        if self.dataset is None:
            return

        ds = self.dataset
        k_params = self.control_panel.get_kspace_params()

        e0, e1 = ds.energy_axis.nearest_slice_exclusive(e_start, e_end)

        if k_params.enabled:
            spatial = ds.get_spatial_image_kspace_roi(
                k_start=x_start,
                k_end=x_end,
                e_start=float(ds.energy_axis.values[e0]),
                e_end=float(ds.energy_axis.values[e1 - 1]),
                zero_angle=k_params.zero_angle,
                converter=self.k_converter,
            )
            self.spatial_viewer.set_image(spatial)

            self.current_roi = EnergyAngleROI(
                angle_start_idx=0,
                angle_end_idx=ds.angle_axis.size,
                energy_start_idx=e0,
                energy_end_idx=e1,
                energy_start=float(ds.energy_axis.values[e0]),
                energy_end=float(ds.energy_axis.values[e1 - 1]),
            )
            self.arpes_viewer.set_roi_info(
                None,
                None,
                self.current_roi.energy_start,
                self.current_roi.energy_end,
                k_space=True,
            )
            return

        a0, a1 = ds.angle_axis.nearest_slice_exclusive(x_start, x_end)

        self.current_roi = EnergyAngleROI(
            angle_start_idx=a0,
            angle_end_idx=a1,
            energy_start_idx=e0,
            energy_end_idx=e1,
            angle_start=float(ds.angle_axis.values[a0]),
            angle_end=float(ds.angle_axis.values[a1 - 1]),
            energy_start=float(ds.energy_axis.values[e0]),
            energy_end=float(ds.energy_axis.values[e1 - 1]),
        )

        self.spatial_viewer.set_image(ds.get_spatial_image(self.current_roi))
        self.arpes_viewer.set_roi_info(
            self.current_roi.angle_start,
            self.current_roi.angle_end,
            self.current_roi.energy_start,
            self.current_roi.energy_end,
            k_space=False,
        )

    def _on_kspace_changed(self) -> None:
        """Handle k-space toggle or zero angle change."""
        if self.current_position:
            self._update_spectrum_for_current_position()

        # Show/hide zero line
        k_params = self.control_panel.get_kspace_params()
        if k_params.enabled:
            self.arpes_viewer.show_zero_line(0.0)  # k=0
        elif self.control_panel.is_zero_angle_set():
            self.arpes_viewer.show_zero_line(k_params.zero_angle)
        else:
            self.arpes_viewer.hide_zero_line()

    def _on_integration_changed(self) -> None:
        """Handle integration parameter change."""
        if self.current_position:
            self._update_spectrum_for_current_position()

    def _on_display_settings_changed(self) -> None:
        """Handle display settings change."""
        colormap = self.control_panel.get_colormap()
        self.spatial_viewer.set_colormap(colormap)
        self.arpes_viewer.set_colormap(colormap)

    # =========================================================================
    # Export Operations
    # =========================================================================

    def _save_spatial(self, format: str = "csv") -> None:
        """Save spatial image."""
        if not self._require_dataset("No Data", "Please load a dataset first."):
            return

        if self.current_roi and self.current_roi.angle_start is not None:
            base_filename = DataExporter.generate_spatial_filename(
                (self.current_roi.angle_start, self.current_roi.angle_end or 0),
                (self.current_roi.energy_start or 0, self.current_roi.energy_end or 0),
                extension=format,
            )
        else:
            base_filename = f"spatial_integrated.{format}"

        if format == "csv":
            filter_str = "CSV Files (*.csv);;All Files (*)"
        else:
            filter_str = "Igor Text Files (*.itx);;All Files (*)"

        filepath = self._get_save_filepath(
            "Save Spatial Image",
            str(Path.home() / base_filename),
            filter_str,
        )
        if filepath is None:
            return

        image = self.spatial_viewer.get_current_image()
        if image is None:
            return

        try:
            if format == "csv":
                DataExporter.save_csv(image, filepath)
            else:
                DataExporter.save_spatial_itx(
                    image,
                    filepath,
                    x_axis=self.dataset.x_axis.values,
                    y_axis=self.dataset.y_axis.values,
                    x_unit=self.dataset.x_axis.unit,
                    y_unit=self.dataset.y_axis.unit,
                )
            self._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")

    def _save_arpes(self, format: str = "csv") -> None:
        """Save ARPES spectrum."""
        if not self._require_dataset("No Data", "Please load a dataset first."):
            return

        if not self._require_position("No Position", "Please select a position first."):
            return

        base_filename = DataExporter.generate_arpes_filename(
            self.current_position.x_coord,
            self.current_position.y_coord,
            extension=format,
        )

        if format == "csv":
            filter_str = "CSV Files (*.csv);;All Files (*)"
        else:
            filter_str = "Igor Text Files (*.itx);;All Files (*)"

        filepath = self._get_save_filepath(
            "Save ARPES Spectrum",
            str(Path.home() / base_filename),
            filter_str,
        )
        if filepath is None:
            return

        spectrum = self.arpes_viewer.get_current_data()
        x_axis, energy_axis = self.arpes_viewer.get_current_axes()

        if spectrum is None or x_axis is None or energy_axis is None:
            return

        try:
            if format == "csv":
                DataExporter.save_csv(spectrum, filepath)
            else:
                k_params = self.control_panel.get_kspace_params()
                if k_params.enabled:
                    x_label = "k (Å⁻¹)"
                else:
                    x_label = f"Angle ({self.dataset.angle_axis.unit})"

                DataExporter.save_arpes_itx(
                    spectrum,
                    filepath,
                    x_axis=x_axis,
                    energy_axis=energy_axis,
                    x_label=x_label,
                    energy_unit=self.dataset.energy_axis.unit,
                )
            self._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")

    def _on_save_region_igor(self) -> None:
        """Save selected region as Igor .itx (uses integration area)."""
        if not self._require_dataset("No Data", "Please load a dataset first."):
            return

        if not self._require_position("No Position", "Please select a position on the map."):
            return

        integration = self.control_panel.get_integration_params()

        if not integration.enabled or (integration.x_pixels == 0 and integration.y_pixels == 0):
            reply = QMessageBox.question(
                self,
                "Select Region",
                "Integration is not enabled.\n\n"
                "Enable integration and set X/Y pixels to define the region to export.\n\n"
                "Export single spectrum at current position instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._save_arpes(format="itx")
            return

        region_data, x_axis_region, y_axis_region = self.dataset.extract_region(
            self.current_position,
            integration,
        )

        ny, nx, n_angle, n_energy = region_data.shape
        region_size_mb = region_data.nbytes / (1024 * 1024)

        # Confirm with user
        msg = (
            f"Export selected region?\n\n"
            f"Center: X={self.current_position.x_coord:.1f}, Y={self.current_position.y_coord:.1f} µm\n"
            f"Region: {nx} × {ny} spatial points\n"
            f"Spectra: {n_angle} × {n_energy} (angle × energy)\n\n"
            f"Total: {nx}×{ny}×{n_angle}×{n_energy} = {nx * ny * n_angle * n_energy:,} points\n"
            f"Estimated size: {region_size_mb:.1f} MB"
        )

        reply = QMessageBox.question(
            self,
            "Export Region",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        default_name = (
            f"region_X{self.current_position.x_coord:.0f}_"
            f"Y{self.current_position.y_coord:.0f}_"
            f"{nx}x{ny}.itx"
        )

        filepath = self._get_save_filepath(
            "Save Region (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )
        if filepath is None:
            return

        self._start_busy_operation("Exporting region...")

        try:
            DataExporter.save_region_itx(
                region_data,
                filepath,
                x_axis=x_axis_region,
                y_axis=y_axis_region,
                angle_axis=self.dataset.angle_axis.values,
                energy_axis=self.dataset.energy_axis.values,
                x_unit=self.dataset.x_axis.unit,
                y_unit=self.dataset.y_axis.unit,
                angle_unit=self.dataset.angle_axis.unit,
                energy_unit=self.dataset.energy_axis.unit,
                center_x=self.current_position.x_coord,
                center_y=self.current_position.y_coord,
            )

            self._set_status(f"Saved: {Path(filepath).name}")

            size_mb = Path(filepath).stat().st_size / (1024 * 1024)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Region exported!\n\n"
                f"File: {Path(filepath).name}\n"
                f"Size: {size_mb:.1f} MB\n\n"
                f"Waves: region_4d, spatial_map, axes",
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")
        finally:
            self._finish_busy_operation()

    def _on_save_full_igor(self) -> None:
        """Save full dataset as Igor .itx."""
        if not self._require_dataset("No Data", "Please load a dataset first."):
            return

        data_size_gb = self.dataset.intensity.nbytes / (1024**3)
        shape = self.dataset.intensity.shape

        msg = (
            f"Export full 4D dataset?\n\n"
            f"Data shape: {shape[0]}×{shape[1]}×{shape[2]}×{shape[3]}\n"
            f"(Y × X × Angle × Energy)\n\n"
            f"Estimated size: {data_size_gb:.2f} GB\n\n"
        )

        if data_size_gb > 2.0:
            msg += "⚠️ Large file! Export may take several minutes."

        reply = QMessageBox.question(
            self,
            "Export Full Dataset",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        default_name = "full_dataset.itx"
        if self.dataset.filepath:
            default_name = self.dataset.filepath.stem + "_full.itx"

        filepath = self._get_save_filepath(
            "Save Full Dataset (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )
        if filepath is None:
            return

        self._start_busy_operation("Exporting full dataset...")

        try:
            result = DataExporter.save_full_dataset_itx(
                self.dataset,
                filepath,
                include_4d_data=True,
                max_file_size_gb=10.0,  # Allow up to 10 GB
            )

            self._set_status(f"Saved: {Path(filepath).name}")

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

            QMessageBox.information(self, "Export Complete", info_msg)

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")
        finally:
            self._finish_busy_operation()

    # =========================================================================
    # View Operations
    # =========================================================================

    def _reset_views(self) -> None:
        """Reset all views to default."""
        self.spatial_viewer.auto_range()
        self.arpes_viewer.auto_range()

    def _toggle_theme(self) -> None:
        """Toggle between dark and light theme."""
        self._dark_theme = not self._dark_theme

        if self._dark_theme:
            self.setStyleSheet(DARK_THEME)
            pg.setConfigOptions(**get_pyqtgraph_config(dark=True))
            self.theme_action.setText("&Light Theme")
            self.theme_action.setChecked(False)
        else:
            self.setStyleSheet(LIGHT_THEME)
            pg.setConfigOptions(**get_pyqtgraph_config(dark=False))
            self.theme_action.setText("&Dark Theme")
            self.theme_action.setChecked(True)

    def _show_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Nano-ARPES Browser",
            "<h2>Nano-ARPES Browser</h2>"
            "<p>Version 0.1.0</p>"
            "<p>Professional nano-ARPES data visualization and analysis.</p>"
            "<p>© 2024</p>",
        )

    # =========================================================================
    # Status Bar
    # =========================================================================

    def _set_status(self, message: str) -> None:
        """Set status bar message."""
        self.status_label.setText(message)

    def _update_position_label(self) -> None:
        """Update position label in status bar."""
        if self.current_position:
            self.position_label.setText(
                f"Position: ({self.current_position.x_coord:.1f}, "
                f"{self.current_position.y_coord:.1f}) µm"
            )
        else:
            self.position_label.setText("")

    def _update_memory_label(self) -> None:
        """Update memory usage label."""
        if self.dataset is not None:
            size_mb = self.dataset.intensity.nbytes / (1024 * 1024)

            if size_mb > 1024:
                self.memory_label.setText(f"Data: {size_mb / 1024:.1f} GB")
            else:
                self.memory_label.setText(f"Data: {size_mb:.0f} MB")
        else:
            self.memory_label.setText("")

    def _require_dataset(self, title: str, message: str) -> bool:
        """Ensure a dataset is loaded before continuing."""
        if self.dataset is None:
            QMessageBox.warning(self, title, message)
            return False
        return True

    def _require_position(self, title: str, message: str) -> bool:
        """Ensure a spatial position is selected before continuing."""
        if self.current_position is None:
            QMessageBox.warning(self, title, message)
            return False
        return True

    def _get_save_filepath(
        self,
        title: str,
        default_path: str,
        filter_str: str,
    ) -> Path | None:
        """Show a save-file dialog and return the selected path."""
        filepath_str, _ = QFileDialog.getSaveFileName(
            self,
            title,
            default_path,
            filter_str,
        )
        if not filepath_str:
            return None
        return Path(filepath_str)

    def _show_progress(self, message: str = "") -> None:
        """Show progress bar."""
        self.progress_bar.setRange(0, 0)
        self.progress_bar.show()
        if message:
            self._set_status(message)

    def _hide_progress(self) -> None:
        """Hide progress bar."""
        self.progress_bar.hide()

    def _start_busy_operation(self, message: str) -> None:
        """Start a long-running operation with UI feedback."""
        self._show_progress(message)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

    def _finish_busy_operation(self) -> None:
        """Finish a long-running operation and restore normal UI state."""
        QApplication.restoreOverrideCursor()
        self._hide_progress()

    # =========================================================================
    # Window Events
    # =========================================================================

    def closeEvent(self, event) -> None:
        """Handle window close."""
        reply = QMessageBox.question(
            self,
            "Confirm Exit",
            "Are you sure you want to exit?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._save_state()
            event.accept()
        else:
            event.ignore()
