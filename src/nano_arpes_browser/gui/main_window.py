"""Main application window."""

from pathlib import Path
from typing import Optional

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

from ..core.io import DataExporter, DataLoader
from ..core.models import ARPESDataset, EnergyAngleROI, SpatialPosition
from ..core.processing import KSpaceConverter
from .styles import DARK_THEME, LIGHT_THEME, get_pyqtgraph_config
from .widgets import ARPESViewer, ControlPanel, InfoPanel, SpatialViewer


class MainWindow(QMainWindow):
    """Main application window for ARPES data visualization."""

    def __init__(self):
        super().__init__()

        # State
        self.dataset: Optional[ARPESDataset] = None
        self.current_position: Optional[SpatialPosition] = None
        self.current_roi: Optional[EnergyAngleROI] = None
        self.k_converter = KSpaceConverter()
        self._dark_theme = True

        # Settings
        self.settings = QSettings("NanoARPES", "Browser")

        # Configure pyqtgraph
        pg_config = get_pyqtgraph_config(dark=True)
        pg.setConfigOptions(**pg_config)

        # Setup
        self._setup_ui()
        self._setup_menu()
        self._setup_statusbar()
        self._connect_signals()
        self._restore_state()

        # Apply theme
        self.setStyleSheet(DARK_THEME)

    def _setup_ui(self) -> None:
        """Set up the user interface."""
        self.setWindowTitle("Nano-ARPES Browser")
        self.setMinimumSize(1200, 700)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        # Main horizontal layout
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel: Controls
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

        # Center: Splitter with viewers
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Spatial viewer
        self.spatial_viewer = SpatialViewer()
        self.main_splitter.addWidget(self.spatial_viewer)

        # ARPES viewer
        self.arpes_viewer = ARPESViewer()
        self.main_splitter.addWidget(self.arpes_viewer)

        # Set initial splitter sizes (equal)
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

        # --- Export submenu ---
        export_menu = file_menu.addMenu("&Export")

        # Current view exports
        export_menu.addAction("Spatial Map (CSV)...", lambda: self._save_spatial("csv"))
        export_menu.addAction("Spatial Map (Igor)...", lambda: self._save_spatial("itx"))
        export_menu.addSeparator()
        export_menu.addAction("Spectrum (CSV)...", lambda: self._save_arpes("csv"))
        export_menu.addAction("Spectrum (Igor)...", lambda: self._save_arpes("itx"))
        
        export_menu.addSeparator()
        
        # Region export
        export_menu.addAction("Selected Region (Igor)...", self._on_save_region_igor)
        
        export_menu.addSeparator()
        
        # Full dataset
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

        # Theme toggle
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

        # Status message (left)
        self.status_label = QLabel("Ready")
        self.statusbar.addWidget(self.status_label, stretch=1)

        # Position info
        self.position_label = QLabel("")
        self.position_label.setMinimumWidth(200)
        self.statusbar.addPermanentWidget(self.position_label)

        # Memory info
        self.memory_label = QLabel("")
        self.memory_label.setMinimumWidth(100)
        self.statusbar.addPermanentWidget(self.memory_label)

        # Progress bar (hidden by default)
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
        self.control_panel.export_region_igor_requested.connect(
            self._on_save_region_igor
        )
        self.control_panel.export_full_igor_requested.connect(self._on_save_full_igor)

        # Control panel - other
        self.control_panel.k_space_changed.connect(self._on_kspace_changed)
        self.control_panel.integration_changed.connect(self._on_integration_changed)
        self.control_panel.display_settings_changed.connect(
            self._on_display_settings_changed
        )

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

        # Create position object
        self.current_position = self.dataset.position_from_coords(x_coord, y_coord)

        # Get integration parameters
        integration = self.control_panel.get_integration_params()

        # Get spectrum at position
        spectrum = self.dataset.get_spectrum_at(self.current_position, integration)

        # Apply k-space conversion if enabled
        k_params = self.control_panel.get_kspace_params()

        if k_params.enabled:
            result = self.k_converter.convert_spectrum(
                spectrum,
                self.dataset.energy_axis.values,
                self.dataset.angle_axis.values,
                zero_angle=k_params.zero_angle,
            )
            # First set axes, then set data
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
            # First set axes, then set data
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

        # Update spatial viewer title
        self.spatial_viewer.set_title(
            f"X: {self.current_position.x_coord:.1f} µm, "
            f"Y: {self.current_position.y_coord:.1f} µm"
        )

        # Update status
        self._update_position_label()

        # Update integration rectangle
        if integration.enabled:
            pixel_x = self.dataset.x_axis.step
            pixel_y = self.dataset.y_axis.step
            self.spatial_viewer.show_integration_rect(
                integration.x_pixels, integration.y_pixels, pixel_x, pixel_y
            )
        else:
            self.spatial_viewer.hide_integration_rect()

    def _on_roi_changed(
        self,
        x_start: float,
        x_end: float,
        e_start: float,
        e_end: float,
    ) -> None:
        """Handle ROI change in ARPES viewer."""
        if self.dataset is None:
            return

        k_params = self.control_panel.get_kspace_params()

        # Convert to angle indices
        if k_params.enabled:
            # Convert k values back to angle indices
            e_center = (e_start + e_end) / 2
            if e_center > 0:
                try:
                    angle_start = self.k_converter.k_to_angle(x_start, e_center)
                    angle_end = self.k_converter.k_to_angle(x_end, e_center)
                    angle_start_idx = self.dataset.angle_axis.find_nearest_index(
                        angle_start
                    )
                    angle_end_idx = self.dataset.angle_axis.find_nearest_index(
                        angle_end
                    )
                except ValueError:
                    return
            else:
                return
        else:
            angle_start_idx = self.dataset.angle_axis.find_nearest_index(x_start)
            angle_end_idx = self.dataset.angle_axis.find_nearest_index(x_end)

        energy_start_idx = self.dataset.energy_axis.find_nearest_index(e_start)
        energy_end_idx = self.dataset.energy_axis.find_nearest_index(e_end)

        # Ensure correct order
        if angle_start_idx > angle_end_idx:
            angle_start_idx, angle_end_idx = angle_end_idx, angle_start_idx
        if energy_start_idx > energy_end_idx:
            energy_start_idx, energy_end_idx = energy_end_idx, energy_start_idx

        # Create ROI object
        self.current_roi = EnergyAngleROI(
            angle_start_idx=angle_start_idx,
            angle_end_idx=angle_end_idx,
            energy_start_idx=energy_start_idx,
            energy_end_idx=energy_end_idx,
            angle_start=float(self.dataset.angle_axis.values[angle_start_idx]),
            angle_end=float(self.dataset.angle_axis.values[angle_end_idx]),
            energy_start=float(self.dataset.energy_axis.values[energy_start_idx]),
            energy_end=float(self.dataset.energy_axis.values[energy_end_idx]),
        )

        # Update spatial image
        spatial_image = self.dataset.get_spatial_image(self.current_roi)
        self.spatial_viewer.set_image(spatial_image)

        # Update ARPES viewer title
        self.arpes_viewer.set_roi_info(self.current_roi, k_space=k_params.enabled)

    def _on_kspace_changed(self) -> None:
        """Handle k-space toggle or zero angle change."""
        if self.current_position:
            self._on_spatial_position_changed(
                self.current_position.x_coord,
                self.current_position.y_coord,
            )

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
            self._on_spatial_position_changed(
                self.current_position.x_coord,
                self.current_position.y_coord,
            )

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
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        # Generate filename
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

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Spatial Image",
            str(Path.home() / base_filename),
            filter_str,
        )

        if not filepath:
            return

        image = self.spatial_viewer.get_current_image()
        if image is None:
            return

        try:
            if format == "csv":
                DataExporter.save_csv(image, filepath)
            else:
                DataExporter.save_itx(
                    image,
                    filepath,
                    wave_name="spatial_map",
                    x_axis=self.dataset.x_axis.values,
                    y_axis=self.dataset.y_axis.values,
                    x_label=f"X ({self.dataset.x_axis.unit})",
                    y_label=f"Y ({self.dataset.y_axis.unit})",
                    z_label="Intensity",
                )
            self._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")

    def _save_arpes(self, format: str = "csv") -> None:
        """Save ARPES spectrum."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        if self.current_position is None:
            QMessageBox.warning(self, "No Position", "Please select a position first.")
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

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save ARPES Spectrum",
            str(Path.home() / base_filename),
            filter_str,
        )

        if not filepath:
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

                DataExporter.save_itx_with_axes(
                    spectrum,
                    filepath,
                    wave_name="arpes",
                    x_axis=x_axis,
                    y_axis=energy_axis,
                    x_label=x_label,
                    y_label=f"Energy ({self.dataset.energy_axis.unit})",
                    z_label="Intensity",
                )
            self._set_status(f"Saved: {Path(filepath).name}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")

    def _on_save_region_igor(self) -> None:
        """Save selected region as Igor .itx (uses integration area)."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        if self.current_position is None:
            QMessageBox.warning(self, "No Position", "Please select a position on the map.")
            return

        # Get integration parameters (defines the region)
        integration = self.control_panel.get_integration_params()
        
        if not integration.enabled or (integration.x_pixels == 0 and integration.y_pixels == 0):
            # Ask user to enable integration
            reply = QMessageBox.question(
                self,
                "Select Region",
                "Integration is not enabled.\n\n"
                "Enable integration and set X/Y pixels to define the region to export.\n\n"
                "Export single spectrum at current position instead?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Export single spectrum
                self._save_arpes(format="itx")
            return

        # Calculate region bounds
        x_idx = self.current_position.x_index
        y_idx = self.current_position.y_index
        
        x_start = max(0, x_idx - integration.x_pixels)
        x_end = min(self.dataset.x_axis.size, x_idx + integration.x_pixels + 1)
        y_start = max(0, y_idx - integration.y_pixels)
        y_end = min(self.dataset.y_axis.size, y_idx + integration.y_pixels + 1)
        
        # Region info
        nx = x_end - x_start
        ny = y_end - y_start
        n_angle = self.dataset.angle_axis.size
        n_energy = self.dataset.energy_axis.size
        
        region_size_mb = (nx * ny * n_angle * n_energy * 8) / (1024 * 1024)  # float64
        
        # Confirm with user
        msg = (
            f"Export selected region?\n\n"
            f"Center: X={self.current_position.x_coord:.1f}, Y={self.current_position.y_coord:.1f} µm\n"
            f"Region: {nx} × {ny} spatial points\n"
            f"Spectra: {n_angle} × {n_energy} (angle × energy)\n\n"
            f"Total: {nx}×{ny}×{n_angle}×{n_energy} = {nx*ny*n_angle*n_energy:,} points\n"
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

        # Generate filename
        default_name = (
            f"region_X{self.current_position.x_coord:.0f}_"
            f"Y{self.current_position.y_coord:.0f}_"
            f"{nx}x{ny}.itx"
        )

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Region (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )

        if not filepath:
            return

        self._show_progress("Exporting region...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()

        try:
            # Extract region data
            # Note: x index is reversed in the data
            x_idx_data_start = self.dataset.x_axis.size - 1 - (x_idx + integration.x_pixels)
            x_idx_data_end = self.dataset.x_axis.size - 1 - (x_idx - integration.x_pixels) + 1
            
            # Ensure correct order
            if x_idx_data_start > x_idx_data_end:
                x_idx_data_start, x_idx_data_end = x_idx_data_end, x_idx_data_start
            
            # Clamp to valid range
            x_idx_data_start = max(0, x_idx_data_start)
            x_idx_data_end = min(self.dataset.x_axis.size, x_idx_data_end)
            
            region_data = self.dataset.intensity[
                y_start:y_end,
                x_idx_data_start:x_idx_data_end,
                :, :
            ]
            
            # Get axis slices
            x_axis_region = self.dataset.x_axis.values[x_start:x_end]
            y_axis_region = self.dataset.y_axis.values[y_start:y_end]
            
            # Save to Igor format
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
                f"Waves: region_4d, spatial_map, axes"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            self._hide_progress()
            
    def _on_save_full_igor(self) -> None:
        """Save full dataset as Igor .itx."""
        if self.dataset is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        # Calculate size and warn user
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

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Full Dataset (Igor Pro)",
            str(Path.home() / default_name),
            "Igor Text Files (*.itx);;All Files (*)",
        )

        if not filepath:
            return

        self._show_progress("Exporting full dataset...")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()  # Update UI

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
            QApplication.restoreOverrideCursor()
            self._hide_progress()

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
        if self.dataset:
            size_mb = self.dataset.intensity.nbytes / (1024 * 1024)
            if size_mb > 1024:
                self.memory_label.setText(f"Data: {size_mb / 1024:.1f} GB")
            else:
                self.memory_label.setText(f"Data: {size_mb:.0f} MB")
        else:
            self.memory_label.setText("")

    def _show_progress(self, message: str = "") -> None:
        """Show progress bar."""
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.show()
        if message:
            self._set_status(message)

    def _hide_progress(self) -> None:
        """Hide progress bar."""
        self.progress_bar.hide()

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