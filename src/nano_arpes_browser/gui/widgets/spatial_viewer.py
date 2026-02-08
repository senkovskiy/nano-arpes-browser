"""Spatial map viewer widget."""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QRectF, pyqtSignal
from PyQt6.QtWidgets import QVBoxLayout, QWidget


class SpatialViewer(QWidget):
    """Widget for displaying and interacting with spatial map."""

    position_changed = pyqtSignal(float, float)  # x_coord, y_coord

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image_data: np.ndarray | None = None
        self._x_axis: np.ndarray | None = None
        self._y_axis: np.ndarray | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the widget UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Graphics layout
        self.graphics_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.graphics_widget)

        # Plot area
        self.plot = self.graphics_widget.addPlot(title="Spatial Map")
        self.plot.setAspectLocked(True)
        self.plot.setLabel("bottom", "X", units="µm")
        self.plot.setLabel("left", "Y", units="µm")

        # Image item
        self.image_item = pg.ImageItem()
        self.plot.addItem(self.image_item)

        # Histogram
        self.histogram = pg.HistogramLUTItem(orientation="horizontal")
        self.histogram.setImageItem(self.image_item)
        self.histogram.gradient.loadPreset("viridis")
        self.graphics_widget.addItem(self.histogram, row=1, col=0)

        # Crosshair lines
        pen = pg.mkPen(width=2, color="w")
        self.v_line = pg.InfiniteLine(angle=90, movable=True, pen=pen)
        self.h_line = pg.InfiniteLine(angle=0, movable=True, pen=pen)
        self.plot.addItem(self.v_line, ignoreBounds=True)
        self.plot.addItem(self.h_line, ignoreBounds=True)

        # Integration box indicator
        self.int_rect = pg.QtWidgets.QGraphicsRectItem(0, 0, 0, 0)
        self.int_rect.setPen(pg.mkPen(color="r", width=2))
        self.int_rect.hide()
        self.plot.addItem(self.int_rect)

        # Connect signals
        self.v_line.sigPositionChanged.connect(self._on_line_moved)
        self.h_line.sigPositionChanged.connect(self._on_line_moved)

    def set_data(
        self,
        image: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
    ) -> None:
        """Set spatial map data with axes."""
        self._image_data = image
        self._x_axis = x_axis
        self._y_axis = y_axis

        self.image_item.setImage(image)
        self._update_scale()

        # Set crosshair bounds
        self.v_line.setBounds([x_axis[0], x_axis[-1]])
        self.h_line.setBounds([y_axis[0], y_axis[-1]])

    def set_image(self, image: np.ndarray) -> None:
        """Update image data only (keep axes)."""
        self._image_data = image
        self.image_item.setImage(image)

    def _update_scale(self) -> None:
        """Update image scale to match axes."""
        if self._x_axis is None or self._y_axis is None:
            return

        x_step = self._x_axis[1] - self._x_axis[0]
        y_step = self._y_axis[1] - self._y_axis[0]

        left = self._x_axis[0] - x_step / 2
        bottom = self._y_axis[0] - y_step / 2
        width = self._x_axis[-1] - self._x_axis[0] + x_step
        height = self._y_axis[-1] - self._y_axis[0] + y_step

        self.image_item.setRect(QRectF(left, bottom, width, height))

    def set_position(self, x: float, y: float) -> None:
        """Set crosshair position."""
        self.v_line.setPos(x)
        self.h_line.setPos(y)
        # Emit signal manually since programmatic changes don't always trigger
        self.position_changed.emit(x, y)

    def get_position(self) -> tuple[float, float] | None:
        """Get current crosshair position."""
        if self._x_axis is None:
            return None
        return self.v_line.getPos()[0], self.h_line.getPos()[1]

    def set_title(self, title: str) -> None:
        """Set plot title."""
        self.plot.setTitle(title)

    def get_current_image(self) -> np.ndarray | None:
        """Get current image data."""
        return self._image_data

    def show_integration_rect(
        self,
        x_size: int,
        y_size: int,
        pixel_size_x: float,
        pixel_size_y: float,
    ) -> None:
        """Show integration rectangle around crosshair."""
        pos = self.get_position()
        if pos is None:
            return

        width = x_size * 2 * pixel_size_x
        height = y_size * 2 * pixel_size_y

        self.int_rect.setRect(0, 0, width, height)
        self.int_rect.setPos(pos[0] - width / 2, pos[1] - height / 2)
        self.int_rect.show()

    def hide_integration_rect(self) -> None:
        """Hide integration rectangle."""
        self.int_rect.hide()

    def set_colormap(self, name: str) -> None:
        """Set colormap by name."""
        try:
            self.histogram.gradient.loadPreset(name)
        except Exception:
            pass  # Ignore invalid colormap names

    def auto_range(self) -> None:
        """Reset view to show all data."""
        self.plot.autoRange()

    def _on_line_moved(self) -> None:
        """Handle crosshair movement."""
        pos = self.get_position()
        if pos:
            self.position_changed.emit(*pos)
