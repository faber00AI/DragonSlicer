"""'Genera GOO': choose the printer model and the resin profile, then where to write the file."""
from pathlib import Path

from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout)

from core import resins

CURRENT = "Impostazioni attuali (tab Resina)"


class GenerateDialog(QDialog):
    def __init__(self, parent, machines, machine_name, resin_name, folder="", filename="job.goo", printer_panel=None):
        super().__init__(parent)
        self.setWindowTitle("Genera GOO")
        self.setMinimumWidth(460)
        self.machines = machines
        lay = QVBoxLayout(self)
        f = QFormLayout()
        self.machine = QComboBox()
        self.machine.addItems([m.name for m in machines])
        self.machine.setCurrentText(machine_name)
        self.machine.currentIndexChanged.connect(lambda _: self._fill_resins())
        f.addRow("Stampante", self.machine)
        self.resin = QComboBox()
        f.addRow("Resina", self.resin)
        self.save = QCheckBox("Salva il file .goo")
        self.save.setChecked(True)
        row = QHBoxLayout()
        self.path = QLineEdit(str(Path(folder) / filename) if folder else filename)
        self.browse = QPushButton("Sfoglia…")
        self.browse.clicked.connect(self._browse)
        row.addWidget(self.path, 1)
        row.addWidget(self.browse)
        f.addRow(self.save)
        f.addRow("File", row)
        self.panel = printer_panel
        self.send = QCheckBox("Invia alla stampante Wi-Fi")
        f.addRow(self.send)
        self.wifi = QComboBox()
        self.start = QCheckBox("Avvia la stampa dopo l'invio")
        self.scan = QPushButton("Cerca in rete")
        row = QHBoxLayout()
        if printer_panel is not None:
            self.wifi.setModel(printer_panel.combo.model())     # the printers found in the Stampante tab
            self.wifi.setCurrentIndex(printer_panel.combo.currentIndex())
            self.scan.clicked.connect(printer_panel.scan)
            printer_panel.combo.currentIndexChanged.connect(self.wifi.setCurrentIndex)
        row.addWidget(self.wifi, 1)
        row.addWidget(self.scan)
        f.addRow("Stampante Wi-Fi", row)
        f.addRow(self.start)
        self.send.toggled.connect(self._send_toggled)
        self.save.toggled.connect(self._save_toggled)
        if printer_panel is None:
            self.send.setEnabled(False)
        lay.addLayout(f)
        self.note = QLabel("")
        self.note.setWordWrap(True)
        lay.addWidget(self.note)
        bb = self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Genera")
        bb.button(QDialogButtonBox.Cancel).setText("Annulla")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._start_machine = machine_name
        self._fill_resins(resin_name)
        self._send_toggled(False)

    def _send_toggled(self, on):
        for w in (self.wifi, self.scan, self.start):
            w.setEnabled(on)
        if on and self.panel is not None and not self.panel.printers:
            self.note.setText("Nessuna stampante collegata: premi «Cerca in rete» (o inserisci l'IP nel tab Stampante).")
        self._check_ok()

    def _save_toggled(self, on):
        self.path.setEnabled(on)
        self.browse.setEnabled(on)
        self._check_ok()

    def _check_ok(self):
        ok = getattr(self, "buttons", None)
        if ok is not None:
            ok.button(QDialogButtonBox.Ok).setEnabled(self.save.isChecked() or self.send.isChecked())

    def _fill_resins(self, select=None):
        name = self.machine.currentText()
        self.resin.clear()
        same = name == self._start_machine if hasattr(self, "_start_machine") else True
        if same:
            self.resin.addItem(CURRENT, None)          # keep the values edited in the Resina tab
        for r in resins.list_profiles(name):
            self.resin.addItem(("★ " if r["source"] == "user" else "") + r["name"]
                               + (f"  ({r['type']})" if r["type"] and r["source"] == "library" else ""), r["name"])
        i = self.resin.findData(select) if select else 0
        self.resin.setCurrentIndex(max(i, 0))
        self.note.setText("" if same else "Cambiando stampante le impostazioni attuali vengono sostituite dal profilo scelto.")

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Salva GOO", self.path.text(), "ELEGOO GOO (*.goo)")
        if p:
            self.path.setText(p)

    def result_values(self):
        """(machine name, resin profile name or None for the current values, path or '' when not saving,
        send over Wi-Fi, start after sending, index of the Wi-Fi printer)"""
        p = self.path.text().strip() if self.save.isChecked() else ""
        if p and not p.lower().endswith(".goo"):
            p += ".goo"
        return (self.machine.currentText(), self.resin.currentData(), p, self.send.isChecked(),
                self.send.isChecked() and self.start.isChecked(), self.wifi.currentIndex())
