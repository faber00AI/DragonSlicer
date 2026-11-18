import os
import sys


def _console_streams():
    """Windowed exe has no console: attach to the parent one / the inherited pipes so `cli` and `mcp` modes can print and read."""
    if os.name != "nt":
        return
    import ctypes
    import msvcrt
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetStdHandle.restype = ctypes.c_void_p
    k.AttachConsole(-1)
    for fd, name, std, dev, flags in ((0, "stdin", -10, "CONIN$", os.O_RDONLY), (1, "stdout", -11, "CONOUT$", os.O_WRONLY),
                                      (2, "stderr", -12, "CONOUT$", os.O_WRONLY)):
        if getattr(sys, name, None) is not None:
            continue
        h = k.GetStdHandle(std)
        try:
            if h and h != ctypes.c_void_p(-1).value:
                f = os.fdopen(msvcrt.open_osfhandle(h, flags), "r" if fd == 0 else "w", encoding="utf-8", errors="replace",
                              buffering=1 if fd else -1, closefd=False)
            else:
                f = open(dev, "r" if fd == 0 else "w", encoding="utf-8", errors="replace")
        except OSError:
            f = open(os.devnull, "r" if fd == 0 else "w")
        setattr(sys, name, f)
        setattr(sys, "__" + name + "__", f)


def _run_console():
    mode = sys.argv.pop(1)
    _console_streams()
    if mode == "mcp":
        import dragonslide_mcp
        dragonslide_mcp.server.run("stdio")
    else:
        from dragonslide_cli import main as cli
        sys.exit(cli())


if len(sys.argv) > 1 and sys.argv[1] in ("cli", "mcp"):
    _run_console()
    sys.exit(0)

from PySide6.QtGui import QColor, QIcon, QPalette
from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    for role, c in ((QPalette.Window, "#1e2025"), (QPalette.WindowText, "#e6e6e6"), (QPalette.Base, "#16181c"),
                    (QPalette.AlternateBase, "#1e2025"), (QPalette.Text, "#e6e6e6"), (QPalette.Button, "#2a2d34"),
                    (QPalette.ButtonText, "#e6e6e6"), (QPalette.Highlight, "#3b82f6"),
                    (QPalette.HighlightedText, "#ffffff")):
        p.setColor(role, QColor(c))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    dark(app)
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    icon = os.path.join(base, "assets", "dragonslide.png")
    if os.path.exists(icon):
        app.setWindowIcon(QIcon(icon))
    w = MainWindow()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
