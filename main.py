import os
import sys

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
