"""Windows: associate .dsl files with DragonSlicer (own file icon, double click opens the project). Per user (HKCU), no admin."""
import os
import shutil
import sys

PROGID = "DragonSlicer.Project"


def _command():
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" "%1"'
    main = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
    return f'"{sys.executable}" "{main}" "%1"'


def _icon_source():
    base = getattr(sys, "_MEIPASS", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    return os.path.join(base, "assets", "dsl_file.ico")


def register() -> str:
    """Write the association. Returns the icon path. Raises OSError / RuntimeError on failure."""
    if os.name != "nt":
        raise RuntimeError("Disponibile solo su Windows")
    import winreg
    folder = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "DragonSlicer")
    os.makedirs(folder, exist_ok=True)
    icon = os.path.join(folder, "dsl_file.ico")          # stable copy: the onefile exe unpacks to a temp folder
    shutil.copyfile(_icon_source(), icon)

    def put(path, value, name=""):
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, path, 0, winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, name, 0, winreg.REG_SZ, value)

    classes = r"Software\Classes"
    put(rf"{classes}\.dsl", PROGID)
    put(rf"{classes}\.dsl", "application/x-dragonslicer-project", "Content Type")
    put(rf"{classes}\{PROGID}", "Progetto DragonSlicer")
    put(rf"{classes}\{PROGID}\DefaultIcon", icon)
    put(rf"{classes}\{PROGID}\shell\open\command", _command())
    # a previous "Open with" choice overrides the default: remove it so the new association takes effect
    for sub in (r"UserChoice",):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, rf"Software\Microsoft\Windows\CurrentVersion\Explorer\FileExts\.dsl\{sub}")
        except OSError:
            pass
    try:
        import ctypes
        ctypes.windll.shell32.SHChangeNotify(0x08000000, 0, None, None)      # SHCNE_ASSOCCHANGED: refresh icons
    except Exception:
        pass
    return icon
