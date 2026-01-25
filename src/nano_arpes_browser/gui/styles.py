"""Application styles and themes."""

DARK_THEME = """
QMainWindow {
    background-color: #1e1e1e;
}

QWidget {
    background-color: #2d2d2d;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 10pt;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #555555;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #80cbc4;
}

QPushButton {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #4d4d4d;
    border-color: #80cbc4;
}

QPushButton:pressed {
    background-color: #2d2d2d;
}

QPushButton:disabled {
    background-color: #2d2d2d;
    color: #666666;
}

QPushButton#primaryButton {
    background-color: #00796b;
    border-color: #00796b;
    color: white;
}

QPushButton#primaryButton:hover {
    background-color: #00897b;
}

QSlider::groove:horizontal {
    border: 1px solid #555555;
    height: 6px;
    background: #3d3d3d;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #80cbc4;
    border: none;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #a7ffeb;
}

QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #3d3d3d;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 4px 8px;
    min-width: 60px;
}

QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #80cbc4;
}

QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #555555;
    background-color: #3d3d3d;
}

QCheckBox::indicator:checked {
    background-color: #00796b;
    border-color: #00796b;
}

QCheckBox::indicator:hover {
    border-color: #80cbc4;
}

QLCDNumber {
    background-color: #1e1e1e;
    color: #80cbc4;
    border: 1px solid #555555;
    border-radius: 4px;
}

QStatusBar {
    background-color: #252526;
    border-top: 1px solid #3d3d3d;
}

QStatusBar::item {
    border: none;
}

QMenuBar {
    background-color: #2d2d2d;
    border-bottom: 1px solid #3d3d3d;
}

QMenuBar::item:selected {
    background-color: #3d3d3d;
}

QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
}

QMenu::item:selected {
    background-color: #00796b;
}

QToolTip {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: 1px solid #555555;
    padding: 4px;
}

QSplitter::handle {
    background-color: #3d3d3d;
}

QSplitter::handle:horizontal {
    width: 3px;
}

QSplitter::handle:vertical {
    height: 3px;
}

QSplitter::handle:hover {
    background-color: #80cbc4;
}

QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #555555;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #666666;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

LIGHT_THEME = """
QMainWindow {
    background-color: #f5f5f5;
}

QWidget {
    background-color: #ffffff;
    color: #212121;
    font-family: 'Segoe UI', 'SF Pro Display', sans-serif;
    font-size: 10pt;
}

QGroupBox {
    font-weight: bold;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #00796b;
}

QPushButton {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 6px 16px;
    min-width: 80px;
}

QPushButton:hover {
    background-color: #f5f5f5;
    border-color: #00796b;
}

QPushButton#primaryButton {
    background-color: #00796b;
    border-color: #00796b;
    color: white;
}

QPushButton#primaryButton:hover {
    background-color: #00897b;
}

QSlider::groove:horizontal {
    border: 1px solid #e0e0e0;
    height: 6px;
    background: #f5f5f5;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00796b;
    border: none;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 4px 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 3px;
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
}

QCheckBox::indicator:checked {
    background-color: #00796b;
    border-color: #00796b;
}

QLCDNumber {
    background-color: #f5f5f5;
    color: #00796b;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
}

QStatusBar {
    background-color: #f5f5f5;
    border-top: 1px solid #e0e0e0;
}

QMenuBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
}

QMenu {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
}

QMenu::item:selected {
    background-color: #00796b;
    color: white;
}
"""


def get_pyqtgraph_config(dark: bool = True) -> dict:
    """Get pyqtgraph configuration for consistent styling."""
    if dark:
        return {
            "background": "#1e1e1e",
            "foreground": "#e0e0e0",
            "antialias": True,
        }
    return {
        "background": "#ffffff",
        "foreground": "#212121",
        "antialias": True,
    }