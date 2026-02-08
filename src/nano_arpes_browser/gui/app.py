"""Application entry point."""

import sys

from nano_arpes_browser.gui.qt_bootstrap import configure_qt_plugin_paths

configure_qt_plugin_paths()

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from nano_arpes_browser.gui.main_window import MainWindow


def main() -> None:
    """Launch the Nano-ARPES Browser application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Nano-ARPES Browser")
    app.setOrganizationName("NanoARPES")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
