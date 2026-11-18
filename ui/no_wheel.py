"""Spin boxes and drop-downs ignore the mouse wheel: values change only with the arrows or the keyboard.
The wheel is handed to the parent instead, so a scroll area under the cursor still scrolls."""
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QAbstractSpinBox, QApplication, QComboBox


class NoWheel(QObject):
    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            parent = obj.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, ev)
            else:
                ev.ignore()
            return True
        return False


def install(app=None):
    app = app or QApplication.instance()
    f = NoWheel(app)
    app.installEventFilter(f)
    return f
