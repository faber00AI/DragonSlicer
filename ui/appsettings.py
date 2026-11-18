"""Persistent app settings (QSettings), shared by every panel."""
import os

from PySide6.QtCore import QSettings

from core.machine import APP, FROZEN


def app_settings() -> QSettings:
    """DRAGONSLICER_SETTINGS (an .ini path) redirects them, so tests leave the real ones alone. The portable exe keeps them
    in dragonslicer.ini beside itself instead of the registry; from source they live in the registry as before."""
    path = os.environ.get("DRAGONSLICER_SETTINGS") or (str(APP / "dragonslicer.ini") if FROZEN else None)
    return QSettings(path, QSettings.IniFormat) if path else QSettings("DragonSlicer", "app")
