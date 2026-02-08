"""Application entry point."""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main():
    """Launch the Nano-ARPES Browser application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Nano-ARPES Browser")
    app.setOrganizationName("NanoARPES")
    app.setFont(QFont("Segoe UI", 10))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
