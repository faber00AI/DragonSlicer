"""Persistent app settings (QSettings), shared by every panel."""
import os

from PySide6.QtCore import QSettings

from core.machine import APP, FROZEN


def app_settings() -> QSettings:
    """DRAGONSLIDE_SETTINGS (an .ini path) redirects them, so tests leave the real ones alone. The portable exe keeps them
    in dragonslide.ini beside itself instead of the registry; from source they live in the registry as before."""
    path = os.environ.get("DRAGONSLIDE_SETTINGS") or (str(APP / "dragonslide.ini") if FROZEN else None)
    return QSettings(path, QSettings.IniFormat) if path else QSettings("dragonSlide", "app")
