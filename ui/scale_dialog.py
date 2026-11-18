"""Asked on import when a model is tiny (almost always a unit mismatch: cm, inches, metres)."""
from PySide6.QtWidgets import (QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel,
                               QVBoxLayout)

SMALL_MM = 5.0          # largest side below this triggers the question
PRESETS = (("×10  (cm → mm)", 10.0), ("×25,4  (pollici → mm)", 25.4), ("×1000  (metri → mm)", 1000.0))


def suggest_factor(extent_max: float) -> float:
    """Smallest common unit factor that brings the largest side to at least 10 mm."""
    for _, f in PRESETS:
        if extent_max * f >= 10.0:
            return f
    return PRESETS[-1][1]


class ScaleDialog(QDialog):
    def __init__(self, parent, name, extents, bed):
        super().__init__(parent)
        self.setWindowTitle("Modello molto piccolo")
        self.extents, self.bed = extents, bed
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel(f"«{name}» misura solo {extents[0]:.3g} × {extents[1]:.3g} × {extents[2]:.3g} mm.\n"
                             "Probabilmente è in altre unità. Vuoi scalarlo ora?"))
        f = QFormLayout()
        self.combo = QComboBox()
        self.combo.addItems([t for t, _ in PRESETS] + ["Personalizzato"])
        self.factor = QDoubleSpinBox()
        self.factor.setRange(0.001, 100000)
        self.factor.setDecimals(3)
        self.factor.setSingleStep(1)
        f.addRow("Unità", self.combo)
        f.addRow("Fattore", self.factor)
        lay.addLayout(f)
        self.result = QLabel()
        lay.addWidget(self.result)
        bb = QDialogButtonBox()
        bb.addButton("Scala", QDialogButtonBox.AcceptRole)
        bb.addButton("Lascia com'è", QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self.combo.currentIndexChanged.connect(self._combo)
        self.factor.valueChanged.connect(self._update)
        best = suggest_factor(max(extents))
        self.combo.setCurrentIndex([f for _, f in PRESETS].index(best))
        self._combo(self.combo.currentIndex())

    def _combo(self, i):
        if i < len(PRESETS):
            self.factor.setValue(PRESETS[i][1])
        self._update()

    def _update(self, *_):
        k = self.factor.value()
        e = [x * k for x in self.extents]
        txt = f"Dimensioni risultanti: {e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm"
        if e[0] > self.bed[0] or e[1] > self.bed[1] or e[2] > self.bed[2]:
            txt += "\n⚠ non entra nel piatto della stampante"
        self.result.setText(txt)

    @staticmethod
    def ask(parent, name, extents, bed):
        """Returns the factor to apply, or None to keep the model as it is."""
        d = ScaleDialog(parent, name, extents, bed)
        return d.factor.value() if d.exec() == QDialog.Accepted else None
