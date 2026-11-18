"""Rullo tab: texture roller (patterned sleeve + core + handles) with a library of saved patterns."""
import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QMessageBox, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import roller as rl

from .appsettings import app_settings

GREEN = ("QPushButton{background:#16a34a;color:white;font-size:14px;font-weight:bold;border-radius:6px}"
         "QPushButton:hover{background:#22c55e}QPushButton:disabled{background:#3a3f47;color:#888}")
DETAIL = {"Basso": 24, "Medio": 40, "Alto": 64}
IMAGE = "— immagine caricata —"


def _spin(lo, hi, val, step, dec=1, suffix=" mm", tip=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setSuffix(suffix)
    if tip:
        s.setToolTip(tip)
    return s


def _fits_alone(ext, m):
    ex, ey, ez = (float(v) for v in ext)
    return ez <= m.bed_z and ((ex <= m.bed_x and ey <= m.bed_y) or (ey <= m.bed_x and ex <= m.bed_y))


class RollerPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._raw = rl.builtin_tile("Rombi")             # the tile before invert / threshold / mirror
        self._loading = False
        lay = QVBoxLayout(self)

        # ---- pattern ----
        g = QGroupBox("Pattern")
        v = QVBoxLayout(g)
        self.src = QComboBox()
        self.src.addItems(list(rl.BUILTIN))
        self.src.setToolTip("Disegni pronti. Per un disegno tuo carica un'immagine (bianco = rilievo, nero = sfondo)")
        self.src.setCurrentText("Rombi")
        v.addWidget(self.src)
        self.b_img = QPushButton("Carica immagine…")
        self.b_img.setToolTip("PNG o JPG in scala di grigi: il bianco diventa rilievo. Per un risultato senza giunzioni usa un "
                              "disegno ripetibile oppure attiva «Rendi ripetibile»")
        self.b_img.clicked.connect(self.load_image)
        v.addWidget(self.b_img)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(130)
        self.preview.setToolTip("Anteprima del disegno ripetuto 2×2")
        v.addWidget(self.preview)
        f = QFormLayout()
        self.invert = QCheckBox("Inverti")
        self.invert.setToolTip("Scambia rilievo e sfondo")
        self.mirror = QCheckBox("Rendi ripetibile (specchia i bordi)")
        self.mirror.setToolTip("Riflette il disegno in orizzontale e verticale così i bordi combaciano sempre: evita la giunzione "
                               "visibile a ogni giro del rullo")
        self.thr = _spin(0, 1, 0, 0.05, 2, "", "Soglia: 0 = rilievo morbido con tutte le sfumature; sopra 0 il rilievo diventa a "
                                                 "gradino (netto): ciò che supera la soglia è rilievo")
        v.addWidget(self.invert)
        v.addWidget(self.mirror)
        f.addRow("Soglia", self.thr)
        v.addLayout(f)

        row = QHBoxLayout()
        self.saved = QComboBox()
        self.saved.setToolTip("Pattern che hai salvato (disegno + impostazioni del rilievo). Sceglilo per ricaricarlo")
        self.b_save_pat = QPushButton("Salva pattern…")
        self.b_save_pat.setToolTip("Salva il disegno e le impostazioni del rilievo con un nome, per riusarli")
        self.b_del_pat = QPushButton("Elimina")
        self.b_del_pat.setToolTip("Elimina il pattern salvato selezionato")
        row.addWidget(self.saved, 1)
        row.addWidget(self.b_save_pat)
        row.addWidget(self.b_del_pat)
        v.addLayout(row)
        lay.addWidget(g)

        # ---- relief ----
        g = QGroupBox("Rilievo")
        f = QFormLayout(g)
        self.mode = QComboBox()
        self.mode.addItems(["Sporgente", "Incavato"])
        self.mode.setToolTip("Sporgente: il disegno esce dal rullo e si imprime nel materiale. Incavato: il disegno è scavato e il "
                             "materiale si gonfia dentro")
        self.depth = _spin(0.2, 6, 1.2, 0.1, 2, tip="Profondità del rilievo. Più alta = disegno più marcato, ma più difficile da staccare "
                                                      "da materiali appiccicosi")
        self.ru = QSpinBox()
        self.ru.setRange(1, 60)
        self.ru.setValue(9)
        self.ru.setToolTip("Quante volte il disegno si ripete attorno al rullo (a ogni giro)")
        self.rv = QSpinBox()
        self.rv.setRange(1, 60)
        self.rv.setValue(7)
        self.rv.setToolTip("Quante volte il disegno si ripete lungo il rullo")
        self.margin = _spin(0, 20, 2, 0.5, tip="Bordo liscio senza disegno ai due estremi della manica")
        self.detail = QComboBox()
        self.detail.addItems(list(DETAIL))
        self.detail.setCurrentText("Medio")
        self.detail.setToolTip("Quanti punti per ripetizione: Alto è più fedele ma crea una mesh molto pesante")
        for label, wd in (("Tipo", self.mode), ("Profondità", self.depth), ("Ripetizioni attorno", self.ru),
                          ("Ripetizioni lungo", self.rv), ("Bordo liscio", self.margin), ("Dettaglio", self.detail)):
            f.addRow(label, wd)
            lbl = f.labelForField(wd)
            if lbl is not None and wd.toolTip():
                lbl.setToolTip(wd.toolTip())
        lay.addWidget(g)

        # ---- roller ----
        g = QGroupBox("Rullo")
        f = QFormLayout(g)
        self.diam = _spin(14, 120, 36, 1, tip="Diametro esterno (cima del rilievo). Decide anche dopo quanta strada il disegno si ripete")
        self.length = _spin(20, 200, 90, 5, tip="Lunghezza della manica con il disegno")
        self.wall = _spin(2, 20, 4, 0.5, tip="Spessore della parete della manica sotto il rilievo. Più sottile = meno materiale")
        self.clear = _spin(0, 1.5, 0.3, 0.05, 2, tip="Gioco tra manica e anima: aumentalo se la manica si blocca, riducilo se balla")
        self.bore = _spin(3, 30, 8, 1, tip="Diametro dei perni di fissaggio dell'anima e dei fori delle maniglie")
        self.handles = QCheckBox("Anima e maniglie")
        self.handles.setChecked(True)
        self.handles.setToolTip("Crea anche l'anima su cui gira la manica e due maniglie. Tienilo attivo la prima volta; poi "
                                "toglilo per rifare solo una manica con un altro disegno")
        for label, wd in (("Diametro", self.diam), ("Lunghezza", self.length), ("Parete manica", self.wall),
                          ("Gioco manica", self.clear), ("Diametro perni", self.bore)):
            f.addRow(label, wd)
            lbl = f.labelForField(wd)
            if lbl is not None:
                lbl.setToolTip(wd.toolTip())
        f.addRow(self.handles)
        lay.addWidget(g)

        self.btn = QPushButton("Genera rullo")
        self.btn.setMinimumHeight(34)
        self.btn.setStyleSheet(GREEN)
        self.btn.setToolTip("Crea manica con il disegno, anima e maniglie e li aggiunge al piatto")
        self.btn.clicked.connect(self.generate)
        lay.addWidget(self.btn)
        self.b_save = QPushButton("Salva impostazioni")
        self.b_save.setToolTip("Memorizza le misure del rullo e le impostazioni di questa tab: verranno ripristinate a ogni avvio")
        self.b_save.clicked.connect(self.save_settings)
        lay.addWidget(self.b_save)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        lay.addStretch(1)

        self.src.currentTextChanged.connect(self._src_changed)
        for wd in (self.invert, self.mirror):
            wd.toggled.connect(self._preview)
        self.thr.valueChanged.connect(self._preview)
        self.saved.activated.connect(self._pick_saved)
        self.b_save_pat.clicked.connect(self.save_pattern)
        self.b_del_pat.clicked.connect(self.delete_pattern)
        self._fill_saved()
        self.load_settings()
        self._preview()

    # ---------------- pattern ----------------
    def _src_changed(self, name):
        if self._loading:
            return
        if name in rl.BUILTIN:
            self._raw = rl.builtin_tile(name)
        self._preview()

    def _tile(self):
        return rl.prepare(self._raw, self.invert.isChecked(), self.thr.value(), self.mirror.isChecked())

    def _preview(self, *_):
        t = np.tile(self._tile(), (2, 2))
        img = (np.clip(t, 0, 1) * 255).astype(np.uint8)
        qi = QImage(img.data, img.shape[1], img.shape[0], img.shape[1], QImage.Format_Grayscale8).copy()
        self.preview.setPixmap(QPixmap.fromImage(qi).scaled(128, 128, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Immagine del pattern", "", "Immagini (*.png *.jpg *.jpeg *.bmp);;Tutti (*)")
        if not path:
            return
        try:
            with open(path, "rb") as fh:
                self._raw = rl.image_tile(fh.read())
        except Exception as e:
            return self._fail(f"Immagine non leggibile: {e}")
        self._loading = True
        if self.src.findText(IMAGE) < 0:
            self.src.addItem(IMAGE)
        self.src.setCurrentText(IMAGE)
        self._loading = False
        self._preview()

    # ---------------- saved patterns ----------------
    def _fill_saved(self, select=None):
        self.saved.blockSignals(True)
        self.saved.clear()
        self.saved.addItem("— pattern salvati —", None)
        for name in sorted(rl.list_patterns()):
            self.saved.addItem(name, name)
        if select:
            self.saved.setCurrentIndex(max(self.saved.findData(select), 0))
        self.saved.blockSignals(False)

    def _record(self):
        name = self.src.currentText()
        rec = dict(source=name if name in rl.BUILTIN else "image", invert=self.invert.isChecked(),
                   mirror=self.mirror.isChecked(), threshold=self.thr.value(), raised=self.mode.currentIndex() == 0,
                   depth=self.depth.value(), repeats_u=self.ru.value(), repeats_v=self.rv.value())
        if rec["source"] == "image":
            rec["image"] = rl.encode_image(self._raw)
        return rec

    def save_pattern(self):
        name, ok = QInputDialog.getText(self, "Salva pattern", "Nome del pattern:", text=self.saved.currentData() or "")
        name = name.strip()
        if not ok or not name:
            return
        if name in rl.list_patterns() and QMessageBox.question(
                self, "Sovrascrivere?", f"Il pattern «{name}» esiste già. Sovrascriverlo?") != QMessageBox.Yes:
            return
        try:
            rl.save_pattern(name, self._record())
        except OSError as e:
            return self._fail(f"Salvataggio non riuscito: {e}")
        self._fill_saved(select=name)
        self.win.statusBar().showMessage(f"Pattern «{name}» salvato", 5000)

    def _pick_saved(self, i):
        name = self.saved.itemData(i)
        rec = rl.list_patterns().get(name) if name else None
        if not rec:
            return
        self._loading = True
        try:
            if rec["source"] == "image" and rec.get("image"):
                self._raw = rl.decode_image(rec["image"])
                if self.src.findText(IMAGE) < 0:
                    self.src.addItem(IMAGE)
                self.src.setCurrentText(IMAGE)
            elif rec["source"] in rl.BUILTIN:
                self._raw = rl.builtin_tile(rec["source"])
                self.src.setCurrentText(rec["source"])
            self.invert.setChecked(bool(rec.get("invert")))
            self.mirror.setChecked(bool(rec.get("mirror")))
            self.thr.setValue(float(rec.get("threshold", 0)))
            self.mode.setCurrentIndex(0 if rec.get("raised", True) else 1)
            self.depth.setValue(float(rec.get("depth", 1.2)))
            self.ru.setValue(int(rec.get("repeats_u", 9)))
            self.rv.setValue(int(rec.get("repeats_v", 7)))
        finally:
            self._loading = False
        self._preview()
        self.win.statusBar().showMessage(f"Pattern «{name}» caricato", 4000)

    def delete_pattern(self):
        name = self.saved.currentData()
        if not name:
            return
        if QMessageBox.question(self, "Elimina pattern", f"Eliminare il pattern «{name}»?") != QMessageBox.Yes:
            return
        rl.delete_pattern(name)
        self._fill_saved()

    # ---------------- settings (sizes of the roller) ----------------
    def _fields(self):
        return dict(mode=self.mode, depth=self.depth, ru=self.ru, rv=self.rv, margin=self.margin, detail=self.detail,
                    diam=self.diam, length=self.length, wall=self.wall, clear=self.clear, bore=self.bore, handles=self.handles,
                    invert=self.invert, mirror=self.mirror, thr=self.thr)

    def save_settings(self):
        st = app_settings()
        for key, wd in self._fields().items():
            if isinstance(wd, QComboBox):
                st.setValue(f"roller/{key}", wd.currentIndex())
            elif isinstance(wd, QCheckBox):
                st.setValue(f"roller/{key}", wd.isChecked())
            else:
                st.setValue(f"roller/{key}", wd.value())
        st.setValue("roller/source", self.src.currentText() if self.src.currentText() in rl.BUILTIN else "Rombi")
        st.sync()
        self.win.statusBar().showMessage("Impostazioni rullo salvate: verranno ripristinate a ogni avvio", 5000)

    def load_settings(self):
        st = app_settings()
        self._loading = True
        try:
            for key, wd in self._fields().items():
                v = st.value(f"roller/{key}")
                if v is None:
                    continue
                try:
                    if isinstance(wd, QComboBox):
                        wd.setCurrentIndex(max(0, min(int(v), wd.count() - 1)))
                    elif isinstance(wd, QCheckBox):
                        wd.setChecked(str(v).lower() in ("true", "1"))
                    else:
                        wd.setValue(float(v))
                except (TypeError, ValueError):
                    pass
            src = st.value("roller/source", type=str)
            if src in rl.BUILTIN:
                self.src.setCurrentText(src)
                self._raw = rl.builtin_tile(src)
        finally:
            self._loading = False

    # ---------------- generate ----------------
    def params(self):
        return rl.RollerParams(diameter=self.diam.value(), length=self.length.value(), depth=self.depth.value(),
                               raised=self.mode.currentIndex() == 0, repeats_u=self.ru.value(), repeats_v=self.rv.value(),
                               wall=self.wall.value(), clearance=self.clear.value(), bore=self.bore.value(),
                               margin=self.margin.value(), detail=DETAIL[self.detail.currentText()], handles=self.handles.isChecked())

    def generate(self):
        w = self.win
        try:
            p, tile = self.params(), self._tile()
            res = w.run_busy("Genero il rullo…", lambda: rl.build(tile, p))
        except Exception as e:
            return self._fail(e)
        w.push_undo()
        objs = w.scene.objects
        x = (max(o.bounds()[1][0] for o in objs) if objs else -w.machine.bed_x / 2) + 10
        placed = []
        for label, m in res["parts"]:
            m = m.copy()
            m.apply_translation([0, 0, -m.bounds[0][2]])
            cx = (m.bounds[0][0] + m.bounds[1][0]) / 2
            cy = (m.bounds[0][1] + m.bounds[1][1]) / 2
            m.apply_translation([x + m.extents[0] / 2 - cx, -cy, 0])
            placed.append(w.add_mesh_object(m, f"rullo – {label}", select=False))
            x += m.extents[0] + 10
        w.reselect(len(w.scene.objects) - 1)
        info = res["info"]
        text = (f"Manica: {info['sleeve_ml']:.0f} ml di resina · ripetizione {info['cell'][0]:.1f}×{info['cell'][1]:.1f} mm · "
                f"giro {info['circumference']:.0f} mm")
        if info["detail"] != DETAIL[self.detail.currentText()]:
            text += f"\nDettaglio ridotto a {info['detail']} punti per ripetizione per tenere la mesh gestibile."
        too_big = [o.name for o in placed if not _fits_alone(np.ptp(o.bounds(), axis=0), w.machine)]
        out = [o.name for o in placed if not o.fits(w.machine)]
        if too_big:
            text += "\nTroppo grandi per il piatto: " + ", ".join(too_big)
        elif out:
            text += "\nAlcuni pezzi sono fuori dall'area di stampa: usa «Disponi tutti sul piatto» (tasto destro)."
        self.info.setText(text)
        w.statusBar().showMessage("Rullo generato", 6000)

    def _fail(self, e):
        self.info.setText(f"{e}")
        self.win.statusBar().showMessage(str(e), 6000)
