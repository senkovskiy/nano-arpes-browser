import os
import sys
from pathlib import Path


def configure_qt_plugin_paths() -> None:
    """
    Ensure Qt can locate its platform plugins (e.g. 'cocoa' on macOS).

    Must run BEFORE importing PyQt6.QtWidgets / PyQt6.QtGui.
    """
    # If someone exported QT_PLUGIN_PATH="" it can break discovery
    if os.environ.get("QT_PLUGIN_PATH") == "":
        os.environ.pop("QT_PLUGIN_PATH", None)

    # If user already set it, respect it
    if os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"):
        return

    # macOS only
    if sys.platform != "darwin":
        return

    try:
        import PyQt6
    except Exception:
        return

    platforms = Path(PyQt6.__file__).resolve().parent / "Qt6" / "plugins" / "platforms"
    if not platforms.is_dir():
        return

    # “Any platform plugin present?” (covers libqcocoa, minimal, offscreen, etc.)
    has_any_plugin = any(platforms.glob("libq*.dylib")) or any(platforms.glob("*.dylib"))
    if has_any_plugin:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platforms)
