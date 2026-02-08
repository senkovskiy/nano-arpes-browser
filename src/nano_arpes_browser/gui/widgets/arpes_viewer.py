"""ARPES spectrum viewer widget."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, Qt, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget

from nano_arpes_browser.core.models import EnergyAngleROI


class ARPESViewer(QWidget):
    """Widget for displaying ARPES spectrum with ROI selection."""

    roi_changed = pyqtSignal(float, float, float, float)  # x_start, x_end, e_start, e_end

    def __init__(self, parent=None):
        super().__init__(parent)

        self._spectrum_data: np.ndarray | None = None
        self._x_axis: np.ndarray | None = None  # angle or k
        self._energy_axis: np.ndarray | None = None
        self._x_label: str = "Angle"
        self._y_label: str = "Energy"

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Graphics layout
        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        # Plot area
        self.plot = self.graphics_widget.addPlot(title="ARPES Spectrum")
        self.plot.setLabel("bottom", "Angle", units="°")
        self.plot.setLabel("left", "Energy", units="eV")

        # Image item
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Histogram
        self.histogram = pg.HistogramLUTItem()
        self.histogram.setImageItem(self.image_item)
        self.histogram.gradient.loadPreset("inferno")
        self.histogram.setMaximumWidth(100)
        self.graphics_widget.addItem(self.histogram)

        # ROI - will be properly positioned when data is set
        self.roi = pg.ROI(
            [0, 0],
            [1, 1],
            pen=pg.mkPen(width=2, color="w"),
            handlePen=pg.mkPen(width=2, color="w"),
        )
        self.roi.addScaleHandle([0, 0], [1, 1])
        self.roi.addScaleHandle([1, 1], [0, 0])
        self.roi.addScaleHandle([0, 1], [1, 0])
        self.roi.addScaleHandle([1, 0], [0, 1])
        self.roi.setZValue(10)
        self.plot.addItem(self.roi)

        # Zero angle/k line (initially hidden)
        self.zero_line = pg.InfiniteLine(
            angle=90,
            movable=False,
            pen=pg.mkPen(width=1, color="w", style=Qt.PenStyle.DashLine),
        )
        self.zero_line.hide()
        self.plot.addItem(self.zero_line)

        # Connect signals
        self.roi.sigRegionChanged.connect(self._on_roi_changed)

    def set_axes(
        self,
        x_axis: np.ndarray,
        energy_axis: np.ndarray,
        x_label: str = "Angle",
        y_label: str = "Energy",
    ) -> None:
        """
        Set axis values and labels.

        This should be called BEFORE set_data when changing axis types.
        """
        self._x_axis = x_axis.copy()  # Make a copy to avoid reference issues
        self._energy_axis = energy_axis.copy()
        self._x_label = x_label
        self._y_label = y_label

        # Update axis labels
        if "k" in x_label.lower() or "Å" in x_label:
            self.plot.setLabel("bottom", x_label)
        else:
            self.plot.setLabel("bottom", x_label, units="°")
        self.plot.setLabel("left", y_label, units="eV")

    def set_data(
        self,
        spectrum: np.ndarray,
        auto_levels: bool = True,
    ) -> None:
        """
        Set spectrum data.

        Args:
            spectrum: 2D array (x, energy)
            auto_levels: Whether to auto-scale intensity
        """
        self._spectrum_data = spectrum
        self.image_item.setImage(spectrum, autoLevels=auto_levels)

        # Update scale to match current axes
        self._update_scale()

        # Reset ROI to fit new data
        self._reset_roi()

    def _update_scale(self) -> None:
        """Update image scale to match axes."""
        if self._x_axis is None or self._energy_axis is None:
            return

        if len(self._x_axis) < 2 or len(self._energy_axis) < 2:
            return

        x_min = float(self._x_axis[0])
        x_max = float(self._x_axis[-1])
        e_min = float(self._energy_axis[0])
        e_max = float(self._energy_axis[-1])

        x_step = (x_max - x_min) / (len(self._x_axis) - 1)
        e_step = (e_max - e_min) / (len(self._energy_axis) - 1)

        # Calculate rectangle for image
        left = x_min - x_step / 2
        bottom = e_min - e_step / 2
        width = x_max - x_min + x_step
        height = e_max - e_min + e_step

        self.image_item.setRect(QRectF(left, bottom, width, height))

        # Auto-range the plot to show all data
        self.plot.autoRange()

    def _reset_roi(self) -> None:
        """Reset ROI to default position (center of data)."""
        if self._x_axis is None or self._energy_axis is None:
            return

        if len(self._x_axis) < 2 or len(self._energy_axis) < 2:
            return

        x_min = float(self._x_axis[0])
        x_max = float(self._x_axis[-1])
        e_min = float(self._energy_axis[0])
        e_max = float(self._energy_axis[-1])

        x_range = x_max - x_min
        e_range = e_max - e_min

        # Set ROI to center quarter of the image
        roi_x = x_min + x_range * 0.25
        roi_y = e_min + e_range * 0.25
        roi_width = x_range * 0.5
        roi_height = e_range * 0.5

        # Block signals while repositioning to avoid triggering updates
        self.roi.blockSignals(True)
        self.roi.setPos([roi_x, roi_y])
        self.roi.setSize([roi_width, roi_height])
        self.roi.blockSignals(False)

    def show_zero_line(self, position: float) -> None:
        """Show zero angle/k line."""
        self.zero_line.setPos(position)
        self.zero_line.show()

    def hide_zero_line(self) -> None:
        """Hide zero angle line."""
        self.zero_line.hide()

    def get_current_data(self) -> np.ndarray | None:
        """Get current spectrum data."""
        return self._spectrum_data

    def get_current_axes(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get current x and energy axes."""
        return self._x_axis, self._energy_axis

    def set_title(self, title: str) -> None:
        """Set plot title."""
        self.plot.setTitle(title)

    def set_roi_info(self, roi: EnergyAngleROI, k_space: bool = False) -> None:
        """Update title with ROI information."""
        if roi.angle_start is not None and roi.energy_start is not None:
            if k_space:
                title = f"ROI - E: [{roi.energy_start:.2f} : {roi.energy_end:.2f}] eV"
            else:
                title = (
                    f"Angle: [{roi.angle_start:.1f}° : {roi.angle_end:.1f}°], "
                    f"E: [{roi.energy_start:.2f} : {roi.energy_end:.2f}] eV"
                )
            self.plot.setTitle(title)

    def set_colormap(self, name: str) -> None:
        """Set colormap by name."""
        try:
            self.histogram.gradient.loadPreset(name)
        except Exception:
            pass

    def auto_range(self) -> None:
        """Reset view to show all data."""
        self.plot.autoRange()

    def _on_roi_changed(self) -> None:
        """Handle ROI change."""
        if self._x_axis is None or self._energy_axis is None:
            return

        pos = self.roi.pos()
        size = self.roi.size()

        x_start = pos[0]
        x_end = pos[0] + size[0]
        e_start = pos[1]
        e_end = pos[1] + size[1]

        self.roi_changed.emit(x_start, x_end, e_start, e_end)
