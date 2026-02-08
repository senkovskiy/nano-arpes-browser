"""Control panel widgets."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLCDNumber,
    QMenu,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.core.models import IntegrationParams, KSpaceParams


class ControlPanel(QWidget):
    """Main control panel with all settings."""

    # Signals
    load_requested = pyqtSignal()

    # Export signals
    export_map_csv_requested = pyqtSignal()
    export_map_igor_requested = pyqtSignal()
    export_spectrum_csv_requested = pyqtSignal()
    export_spectrum_igor_requested = pyqtSignal()
    export_region_igor_requested = pyqtSignal()
    export_full_igor_requested = pyqtSignal()

    # Other signals
    k_space_changed = pyqtSignal()
    integration_changed = pyqtSignal()
    display_settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Set up control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # === File Section ===
        file_group = QGroupBox("File")
        file_layout = QVBoxLayout(file_group)
        file_layout.setSpacing(4)

        # Load button
        self.load_btn = QPushButton("Open File...")
        self.load_btn.setObjectName("primaryButton")
        file_layout.addWidget(self.load_btn)

        # Export button with dropdown menu
        self.export_btn = QPushButton("Export")
        self._setup_export_menu()
        file_layout.addWidget(self.export_btn)

        layout.addWidget(file_group)

        # === K-Space Section ===
        kspace_group = QGroupBox("K-Space Conversion")
        kspace_layout = QVBoxLayout(kspace_group)
        kspace_layout.setSpacing(6)

        self.set_zero_checkbox = QCheckBox("Set Zero Angle")
        kspace_layout.addWidget(self.set_zero_checkbox)

        angle_display_layout = QHBoxLayout()
        self.angle_lcd = QLCDNumber()
        self.angle_lcd.setDigitCount(5)
        self.angle_lcd.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.angle_lcd.setFixedHeight(30)
        angle_display_layout.addWidget(QLabel("θ₀:"))
        angle_display_layout.addWidget(self.angle_lcd)
        angle_display_layout.addWidget(QLabel("°"))
        kspace_layout.addLayout(angle_display_layout)

        self.angle_slider = QSlider(Qt.Orientation.Horizontal)
        self.angle_slider.setEnabled(False)
        kspace_layout.addWidget(self.angle_slider)

        minmax_layout = QHBoxLayout()
        self.angle_min_label = QLabel("min")
        self.angle_max_label = QLabel("max")
        self.angle_max_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        minmax_layout.addWidget(self.angle_min_label)
        minmax_layout.addWidget(self.angle_max_label)
        kspace_layout.addLayout(minmax_layout)

        self.k_space_checkbox = QCheckBox("Enable K-Space")
        self.k_space_checkbox.setEnabled(False)
        kspace_layout.addWidget(self.k_space_checkbox)

        layout.addWidget(kspace_group)

        # === Integration Section ===
        int_group = QGroupBox("Spatial Integration")
        int_layout = QVBoxLayout(int_group)
        int_layout.setSpacing(6)

        self.integrate_checkbox = QCheckBox("Enable Integration")
        int_layout.addWidget(self.integrate_checkbox)

        # Spinbox style for bigger buttons
        spinbox_style = """
            QSpinBox::up-button, QSpinBox::down-button {
                min-height: 9px;
                min-width: 30px;
            }
        """

        # X pixels
        x_layout = QHBoxLayout()
        x_layout.addWidget(QLabel("X pixels:"))
        self.x_spinbox = QSpinBox()
        self.x_spinbox.setRange(0, 50)
        self.x_spinbox.setValue(0)
        self.x_spinbox.setEnabled(False)
        self.x_spinbox.setStyleSheet(spinbox_style)
        x_layout.addWidget(self.x_spinbox)
        int_layout.addLayout(x_layout)

        # Y pixels
        y_layout = QHBoxLayout()
        y_layout.addWidget(QLabel("Y pixels:"))
        self.y_spinbox = QSpinBox()
        self.y_spinbox.setRange(0, 50)
        self.y_spinbox.setValue(0)
        self.y_spinbox.setEnabled(False)
        self.y_spinbox.setStyleSheet(spinbox_style)
        y_layout.addWidget(self.y_spinbox)
        int_layout.addLayout(y_layout)

        layout.addWidget(int_group)

        # === Display Section ===
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(6)

        self.lock_range_checkbox = QCheckBox("Lock Intensity Range")
        display_layout.addWidget(self.lock_range_checkbox)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(
            [
                "viridis",
                "inferno",
                "magma",
                "plasma",
                "cividis",
                "turbo",
                "hot",
                "cool",
                "grey",
            ]
        )
        self.colormap_combo.setCurrentText("viridis")
        cmap_layout.addWidget(self.colormap_combo)
        display_layout.addLayout(cmap_layout)

        layout.addWidget(display_group)

        # Stretch at bottom
        layout.addStretch()

    def _setup_export_menu(self) -> None:
        """Set up the export dropdown menu."""
        self.export_menu = QMenu(self)

        # --- Current View Section ---
        self.export_menu.addSection("Current View")

        # Map exports
        map_menu = self.export_menu.addMenu("📍 Spatial Map")
        self.export_map_csv_action = map_menu.addAction("CSV (.csv)")
        self.export_map_igor_action = map_menu.addAction("Igor Pro (.itx)")

        # Spectrum exports
        spectrum_menu = self.export_menu.addMenu("📊 Spectrum")
        self.export_spectrum_csv_action = spectrum_menu.addAction("CSV (.csv)")
        self.export_spectrum_igor_action = spectrum_menu.addAction("Igor Pro (.itx)")

        self.export_region_igor_action = self.export_menu.addAction(
            "📦 Selected Region (Igor .itx)"
        )

        self.export_menu.addSeparator()

        # --- Full Dataset Section ---
        self.export_menu.addSection("Full Dataset")
        self.export_full_igor_action = self.export_menu.addAction("📦 Full Data (Igor .itx)")

        self.export_btn.setMenu(self.export_menu)

    def _connect_signals(self) -> None:
        """Connect internal signals."""
        # File buttons
        self.load_btn.clicked.connect(self.load_requested)

        # Export menu actions
        self.export_map_csv_action.triggered.connect(self.export_map_csv_requested)
        self.export_map_igor_action.triggered.connect(self.export_map_igor_requested)
        self.export_spectrum_csv_action.triggered.connect(self.export_spectrum_csv_requested)
        self.export_spectrum_igor_action.triggered.connect(self.export_spectrum_igor_requested)
        self.export_region_igor_action.triggered.connect(self.export_region_igor_requested)
        self.export_full_igor_action.triggered.connect(self.export_full_igor_requested)

        # K-space controls
        self.set_zero_checkbox.toggled.connect(self._on_set_zero_toggled)
        self.angle_slider.valueChanged.connect(self._on_angle_changed)
        self.k_space_checkbox.toggled.connect(self._on_kspace_toggled)

        # Integration controls
        self.integrate_checkbox.toggled.connect(self._on_integrate_toggled)
        self.x_spinbox.valueChanged.connect(self.integration_changed)
        self.y_spinbox.valueChanged.connect(self.integration_changed)

        # Display controls
        self.lock_range_checkbox.toggled.connect(self.display_settings_changed)
        self.colormap_combo.currentTextChanged.connect(self.display_settings_changed)

    def _on_set_zero_toggled(self, checked: bool) -> None:
        """Handle set zero checkbox toggle."""
        self.angle_slider.setEnabled(checked)
        self.k_space_checkbox.setEnabled(checked)

        if not checked:
            self.k_space_checkbox.setChecked(False)

        self.k_space_changed.emit()

    def _on_angle_changed(self, value: int) -> None:
        """Handle angle slider change."""
        angle = value / 10.0
        self.angle_lcd.display(angle)
        self.k_space_changed.emit()

    def _on_kspace_toggled(self, checked: bool) -> None:
        """Handle k-space checkbox toggle."""
        self.k_space_changed.emit()

    def _on_integrate_toggled(self, checked: bool) -> None:
        """Handle integrate checkbox toggle."""
        self.x_spinbox.setEnabled(checked)
        self.y_spinbox.setEnabled(checked)
        self.integration_changed.emit()

    # === Public Methods ===

    def set_angle_range(self, min_angle: float, max_angle: float) -> None:
        """Set the range of the angle slider."""
        self.angle_slider.setRange(int(min_angle * 10), int(max_angle * 10))
        self.angle_slider.setValue(0)
        self.angle_lcd.display(0)
        self.angle_min_label.setText(f"{min_angle:.1f}°")
        self.angle_max_label.setText(f"{max_angle:.1f}°")

    def get_kspace_params(self) -> KSpaceParams:
        """Get current k-space parameters."""
        return KSpaceParams(
            enabled=self.k_space_checkbox.isChecked(),
            zero_angle=self.angle_slider.value() / 10.0,
        )

    def get_integration_params(self) -> IntegrationParams:
        """Get current integration parameters."""
        return IntegrationParams(
            enabled=self.integrate_checkbox.isChecked(),
            x_pixels=self.x_spinbox.value(),
            y_pixels=self.y_spinbox.value(),
        )

    def is_range_locked(self) -> bool:
        """Check if intensity range is locked."""
        return self.lock_range_checkbox.isChecked()

    def is_zero_angle_set(self) -> bool:
        """Check if zero angle is being set."""
        return self.set_zero_checkbox.isChecked()

    def get_colormap(self) -> str:
        """Get selected colormap name."""
        return self.colormap_combo.currentText()

    def reset(self) -> None:
        """Reset all controls to default state."""
        self.set_zero_checkbox.setChecked(False)
        self.k_space_checkbox.setChecked(False)
        self.integrate_checkbox.setChecked(False)
        self.lock_range_checkbox.setChecked(False)
        self.x_spinbox.setValue(0)
        self.y_spinbox.setValue(0)
        self.angle_slider.setValue(0)


class InfoPanel(QWidget):
    """Panel showing dataset information."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up info panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        info_group = QGroupBox("Dataset Info")
        info_layout = QVBoxLayout(info_group)
        info_layout.setSpacing(4)

        self.filename_label = QLabel("No file loaded")
        self.filename_label.setWordWrap(True)

        self.shape_label = QLabel("")
        self.spatial_label = QLabel("")
        self.energy_label = QLabel("")
        self.angle_label = QLabel("")

        for label in [
            self.filename_label,
            self.shape_label,
            self.spatial_label,
            self.energy_label,
            self.angle_label,
        ]:
            label.setStyleSheet("color: #888888; font-size: 9pt;")
            info_layout.addWidget(label)

        layout.addWidget(info_group)

    def set_dataset_info(self, dataset) -> None:
        """Update info panel with dataset information."""
        if dataset.filepath:
            self.filename_label.setText(f"📁 {dataset.filepath.name}")
        else:
            self.filename_label.setText("📁 Unknown")

        self.shape_label.setText(
            f"📐 {dataset.shape[1]}×{dataset.shape[0]} spatial, "
            f"{dataset.shape[2]}×{dataset.shape[3]} spectral"
        )

        self.spatial_label.setText(
            f"🗺️ X: {dataset.x_axis.min:.1f}–{dataset.x_axis.max:.1f} {dataset.x_axis.unit}\n"
            f"    Y: {dataset.y_axis.min:.1f}–{dataset.y_axis.max:.1f} {dataset.y_axis.unit}"
        )

        self.energy_label.setText(
            f"⚡ E: {dataset.energy_axis.min:.2f}–{dataset.energy_axis.max:.2f} {dataset.energy_axis.unit}"
        )

        self.angle_label.setText(
            f"📐 θ: {dataset.angle_axis.min:.1f}–{dataset.angle_axis.max:.1f}{dataset.angle_axis.unit}"
        )

    def clear(self) -> None:
        """Clear info panel."""
        self.filename_label.setText("No file loaded")
        self.shape_label.setText("")
        self.spatial_label.setText("")
        self.energy_label.setText("")
        self.angle_label.setText("")
