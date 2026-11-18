"""Resin tab: pick a profile (SatelLite library + the user's own), edit every parameter, save / delete user profiles."""
from dataclasses import replace

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QAbstractSpinBox, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QLineEdit, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import resins
from core.machine import Machine, PrintSettings

CUSTOM = "— personalizzato —"

# (label, normal field, bottom field, unit, maximum, decimals)
ROWS = (
    ("Esposizione", "exposure", "bottom_exposure", "s", 300, 2),
    ("Riposo prima del rilascio", "wait_before_cure", "bottom_wait_before_cure", "s", 60, 2),
    ("Riposo dopo il rilascio", "wait_after_cure", "bottom_wait_after_cure", "s", 60, 2),
    ("Ritardo luce spenta", "light_off_delay", "bottom_light_off_delay", "s", 60, 2),
    ("Sollev. 1 distanza", "lift_height", "bottom_lift_height", "mm", 100, 2),
    ("Sollev. 1 velocità", "lift_speed", "bottom_lift_speed", "mm/min", 1000, 0),
    ("Sollev. 2 distanza", "lift2_height", "bottom_lift2_height", "mm", 100, 2),
    ("Sollev. 2 velocità", "lift2_speed", "bottom_lift2_speed", "mm/min", 1000, 0),
    ("Riposo dopo la ritrazione", "wait_after_lift", "bottom_wait_after_lift", "s", 60, 2),
    ("Ritrazione velocità", "retract_speed", "bottom_retract_speed", "mm/min", 1000, 0),
    ("Ritraz. lenta distanza", "retract2_height", "bottom_retract2_height", "mm", 100, 2),
    ("Ritraz. lenta velocità", "retract2_speed", "bottom_retract2_speed", "mm/min", 1000, 0),
    ("Potenza LED (PWM)", "pwm", "bottom_pwm", "", 255, 0),
)


def _spin(maximum, dec, suffix):
    s = QDoubleSpinBox() if dec else QSpinBox()
    s.setRange(0, maximum)
    if dec:
        s.setDecimals(dec)
        s.setSingleStep(0.1)
    s.setMinimumWidth(64)
    s.setButtonSymbols(QAbstractSpinBox.NoButtons)
    return s


class ResinPanel(QWidget):
    changed = Signal()                     # any value edited or a profile loaded

    def __init__(self, machine: Machine):
        super().__init__()
        self.machine = machine
        self._ps = PrintSettings.from_machine(machine)
        self._loading = False
        self._dirty = False
        self.w = {}
        lay = QVBoxLayout(self)

        g = QGroupBox("Profilo resina")
        v = QVBoxLayout(g)
        self.combo = QComboBox()
        self.combo.activated.connect(self._picked)
        v.addWidget(self.combo)
        row = QHBoxLayout()
        self.b_save = QPushButton("Salva")
        self.b_save_as = QPushButton("Salva con nome…")
        self.b_del = QPushButton("Elimina")
        self.b_reset = QPushButton("Ricarica")
        self.b_save.clicked.connect(self.save)
        self.b_save_as.clicked.connect(self.save_as)
        self.b_del.clicked.connect(self.delete)
        self.b_reset.clicked.connect(lambda: self._picked(self.combo.currentIndex()))
        for b in (self.b_save, self.b_save_as, self.b_del, self.b_reset):
            row.addWidget(b)
        v.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        lay.addWidget(g)

        g = QGroupBox("Layer")
        grid = QGridLayout(g)
        for r, (label, key, mx, dec, suf) in enumerate((
                ("Altezza layer", "layer_height", 0.2, 3, "mm"), ("Layer base", "bottom_layers", 100, 0, ""),
                ("Layer di transizione", "transition_layers", 100, 0, ""))):
            wd = _spin(mx, dec, suf)
            if key == "layer_height":
                wd.setMinimum(0.01)
                wd.setSingleStep(0.01)
            self._add(key, wd)
            grid.addWidget(QLabel(f"{label} ({suf})" if suf else label), r, 0)
            grid.addWidget(wd, r, 1)
        lay.addWidget(g)

        g = QGroupBox("Esposizione e movimenti")
        grid = QGridLayout(g)
        grid.addWidget(QLabel("<b>Normale</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Base</b>"), 0, 2)
        for r, (label, nk, bk, unit, mx, dec) in enumerate(ROWS, 1):
            grid.addWidget(QLabel(f"{label} ({unit})" if unit else label), r, 0)
            for c, key in ((1, nk), (2, bk)):
                wd = _spin(mx, dec, unit)
                self._add(key, wd)
                grid.addWidget(wd, r, c)
        self.embed = QCheckBox("Scrivi sollevamento/ritrazione nel file")
        self.embed.setToolTip("Spento (come SatelLite) il .goo porta 0 nei movimenti e la stampante usa il suo profilo; "
                              "acceso, i valori sopra vengono scritti nel file.")
        self.embed.toggled.connect(self._edited)
        grid.addWidget(self.embed, len(ROWS) + 1, 0, 1, 3)
        lay.addWidget(g)

        g = QGroupBox("Resina")
        grid = QGridLayout(g)
        self.type_edit = QLineEdit()
        self.type_edit.textEdited.connect(self._edited)
        grid.addWidget(QLabel("Tipo"), 0, 0)
        grid.addWidget(self.type_edit, 0, 1)
        for r, (label, key, mx, dec, suf) in enumerate((("Densità", "density", 3, 3, "g/ml"),
                                                        ("Costo", "cost", 100000, 2, "€/litro")), 1):
            wd = _spin(mx, dec, suf)
            self._add(key, wd)
            grid.addWidget(QLabel(f"{label} ({suf})"), r, 0)
            grid.addWidget(wd, r, 1)
        lay.addWidget(g)
        lay.addStretch(1)
        self.set_machine(machine)

    def _add(self, key, wd):
        self.w[key] = wd
        wd.valueChanged.connect(self._edited)

    # ---------------- values <-> widgets ----------------
    def settings(self) -> PrintSettings:
        ps = replace(self._ps)
        for k, wd in self.w.items():
            setattr(ps, k, type(getattr(ps, k))(wd.value()))
        ps.embed_motion = self.embed.isChecked()
        ps.resin_type = self.type_edit.text().strip()
        return ps

    def set_settings(self, ps: PrintSettings, name: str | None = None):
        """Show `ps`. name = the profile it comes from (selected in the box), None = custom."""
        self._ps = replace(ps)
        self._loading = True
        try:
            for k, wd in self.w.items():
                wd.setValue(getattr(ps, k))
            self.embed.setChecked(ps.embed_motion)
            self.type_edit.setText(ps.resin_type)
        finally:
            self._loading = False
        i = self.combo.findData(name) if name else -1
        self.combo.setCurrentIndex(i if i >= 0 else 0)
        self._dirty = False
        self._refresh_ui()

    def set_machine(self, machine: Machine):
        keep = self.combo.currentData() if self.combo.count() else None     # profile chosen on the previous printer
        self.machine = machine
        self._fill_combo()
        if keep and self.combo.findData(keep) >= 0:                         # same profile exists here: keep using it
            self.set_settings(resins.to_settings(machine, resins.get_profile(machine.name, keep)), keep)
        else:
            self.set_settings(PrintSettings.from_machine(machine))
        self.changed.emit()

    def _fill_combo(self, select=None):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem(CUSTOM, None)
        rows = resins.list_profiles(self.machine.name)
        for r in rows:
            if r["source"] == "user":
                self.combo.addItem(f"★ {r['name']}", r["name"])
        for r in rows:
            if r["source"] == "library":
                self.combo.addItem(r["name"] + (f"  ({r['type']})" if r["type"] else ""), r["name"])
        self.combo.blockSignals(False)
        if select:
            self.combo.setCurrentIndex(max(self.combo.findData(select), 0))

    def _current(self):
        name = self.combo.currentData()
        if not name:
            return None
        try:
            return resins.get_profile(self.machine.name, name)
        except resins.ResinError:
            return None

    def _refresh_ui(self):
        p = self._current()
        user = bool(p and p["source"] == "user")
        self.b_save.setEnabled(user)
        self.b_del.setEnabled(user)
        self.b_reset.setEnabled(p is not None)
        if p:
            src = " — profilo utente" if user else " — libreria SatelLite"
            self.info.setText(("Modificato rispetto al profilo" if self._dirty else "Valori del profilo") + src)
        else:
            self.info.setText("Valori personalizzati (non salvati in un profilo)" if self._dirty
                              else "Valori di default della stampante")

    # ---------------- events ----------------
    def _edited(self, *_):
        if self._loading:
            return
        self._dirty = True
        self._refresh_ui()
        self.changed.emit()

    def _picked(self, i):
        name = self.combo.itemData(i)
        if name is None:
            self.set_settings(PrintSettings.from_machine(self.machine))
        else:
            self.set_settings(resins.to_settings(self.machine, resins.get_profile(self.machine.name, name)), name)
        self.changed.emit()

    def pick_index(self, i):
        self.combo.setCurrentIndex(i)
        self._picked(i)

    def select_profile(self, name):
        i = self.combo.findData(name)
        if i >= 0:
            self._picked(i)

    def save(self):
        p = self._current()
        if p is None or p["source"] != "user":
            return self.save_as()
        self._store(p["name"])

    def save_as(self):
        p = self._current()
        base = self._ps.resin if self._ps.resin != "Generic" else ""
        name, ok = QInputDialog.getText(self, "Salva profilo resina", "Nome del profilo (max 31 caratteri):",
                                        text=(p["name"] if p else base))
        name = name.strip()
        if not ok or not name:
            return
        if name in {r["name"] for r in resins.list_profiles(self.machine.name)}:
            existing = resins.get_profile(self.machine.name, name)
            if existing["source"] == "library":
                QMessageBox.warning(self, "Nome già usato",
                                    "Esiste un profilo della libreria con questo nome: scegline un altro.")
                return
            if QMessageBox.question(self, "Sovrascrivere?",
                                    f"Il profilo utente «{name}» esiste già. Sovrascriverlo?") != QMessageBox.Yes:
                return
        self._store(name)

    def _store(self, name):
        ps = self.settings()
        try:
            resins.save_user_profile(self.machine.name, name, ps)
        except resins.ResinError as e:
            QMessageBox.critical(self, "Salvataggio fallito", str(e))
            return
        ps.resin = name
        self._ps = ps
        self._fill_combo(select=name)
        self._dirty = False
        self._refresh_ui()
        self.changed.emit()

    def delete(self):
        p = self._current()
        if p is None or p["source"] != "user":
            return
        if QMessageBox.question(self, "Elimina profilo", f"Eliminare il profilo «{p['name']}»?") != QMessageBox.Yes:
            return
        try:
            resins.delete_user_profile(self.machine.name, p["name"])
        except resins.ResinError as e:
            QMessageBox.critical(self, "Eliminazione fallita", str(e))
            return
        self._fill_combo()
        self.combo.setCurrentIndex(0)
        self._dirty = True                 # the values on screen are now unsaved custom ones
        self._refresh_ui()
