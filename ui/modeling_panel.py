"""'Modella' tab: primitives, booleans, plane cut, separate shells, repair."""
from pathlib import Path

import numpy as np
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QInputDialog,
                               QLabel, QListWidget, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

from core import library as lib
from core import modeling as md
from core import connectors
from core import text as tx

AXES = ("X", "Y", "Z")
TEXT = "Testo"                      # the one shape that is not in md.PRIMITIVES: it has its own controls


def _spin(p: md.Param):
    if p.integer:
        s = QSpinBox()
        s.setRange(int(p.lo), int(p.hi))
        s.setValue(int(p.default))
    else:
        s = QDoubleSpinBox()
        s.setRange(p.lo, p.hi)
        s.setDecimals(2)
        s.setSingleStep(1.0)
        s.setValue(p.default)
        s.setSuffix(" mm")
    return s


class ModelingPanel(QWidget):
    def __init__(self, win):
        super().__init__()
        self.win = win
        self._txt_last = {}                   # last text settings, kept when the shape combo is switched
        lay = QVBoxLayout(self)

        # ---- primitives ----
        g = QGroupBox("Aggiungi oggetto")
        self.prim_form = QFormLayout(g)
        self.prim_box = QComboBox()
        self.prim_box.addItems(list(md.PRIMITIVES) + [TEXT])
        self.prim_box.currentTextChanged.connect(self._prim_changed)
        self.prim_form.addRow("Forma", self.prim_box)
        self.prim_spins: list = []
        self.prim_holder = QVBoxLayout()
        self.prim_form.addRow(self.prim_holder)
        self.prim_params = None
        btn = QPushButton("Aggiungi al piatto")
        btn.clicked.connect(self.add_primitive)
        self.prim_form.addRow(btn)
        lay.addWidget(g)
        self.g_add = g                        # the main window shows these two on the Scena tab
        self._prim_changed(self.prim_box.currentText())

        # ---- personal library ----
        g = QGroupBox("Libreria personale")
        v = QVBoxLayout(g)
        self.lib_list = QListWidget()
        self.lib_list.setMinimumHeight(110)
        self.lib_list.itemDoubleClicked.connect(lambda _: self.lib_add())
        self.lib_list.currentRowChanged.connect(lambda _: self._lib_buttons())
        v.addWidget(self.lib_list)
        self.lib_info = QLabel("")
        self.lib_info.setWordWrap(True)
        v.addWidget(self.lib_info)
        self.lib_btns = {}
        for row_buttons in ((("add", "Aggiungi al piatto", self.lib_add), ("save", "Salva selezione…", self.lib_save_selected)),
                            (("file", "Importa file…", self.lib_import), ("ren", "Rinomina…", self.lib_rename),
                             ("del", "Elimina", self.lib_delete))):
            row = QHBoxLayout()
            for key, text, fn in row_buttons:
                b = QPushButton(text)
                b.clicked.connect(lambda _=False, f=fn: f())
                row.addWidget(b)
                self.lib_btns[key] = b
            v.addLayout(row)
        lay.addWidget(g)
        self.g_lib = g
        self.lib_refresh()

        # ---- booleans ----
        g = QGroupBox("Booleani")
        v = QVBoxLayout(g)
        v.addWidget(QLabel("Seleziona 2+ oggetti (Ctrl+clic).\nL'oggetto corrente è A."))
        row = QHBoxLayout()
        for text, op, tip in (("Unione", "union", "A + altri"), ("Differenza", "difference", "A − altri"),
                              ("Intersezione", "intersection", "Solo la parte in comune")):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, o=op: self.boolean(o))
            row.addWidget(b)
        v.addLayout(row)
        lay.addWidget(g)
        self.g_bool = g

        # ---- cut ----
        g = QGroupBox("Taglia con un piano")
        f = QFormLayout(g)
        self.cut_axis = QComboBox()
        self.cut_axis.addItems(["Z (orizzontale)", "X", "Y"])
        self.cut_pos = QDoubleSpinBox()
        self.cut_pos.setRange(-500, 500)
        self.cut_pos.setDecimals(2)
        self.cut_pos.setSingleStep(1.0)
        self.cut_pos.setSuffix(" mm")
        self.cut_keep = QComboBox()
        self.cut_keep.addItems(["Tieni entrambe", "Solo lato +", "Solo lato −"])
        self.cut_gap = QDoubleSpinBox()
        self.cut_gap.setRange(0, 100)
        self.cut_gap.setSuffix(" mm")
        f.addRow("Asse", self.cut_axis)
        f.addRow("Posizione", self.cut_pos)
        self.cut_tilt = QDoubleSpinBox()
        self.cut_tilt.setRange(-360, 360)
        self.cut_tilt.setSingleStep(5)
        self.cut_tilt.setSuffix("°")
        self.cut_dir = QDoubleSpinBox()
        self.cut_dir.setRange(-360, 360)
        self.cut_dir.setSingleStep(15)
        self.cut_dir.setSuffix("°")
        f.addRow("Inclinazione", self.cut_tilt)
        f.addRow("Direzione", self.cut_dir)
        f.addRow("Parti", self.cut_keep)
        info = QLabel("Perni e magneti: usa il gruppo «Connettori» più in basso (valgono anche per il taglio a pennello).")
        info.setWordWrap(True)
        f.addRow(info)
        f.addRow("Distanzia di", self.cut_gap)
        b = QPushButton("Taglia")
        b.setMinimumHeight(30)
        b.setStyleSheet("QPushButton{background:#16a34a;color:white;font-weight:bold;border-radius:6px}"
                        "QPushButton:hover{background:#22c55e}QPushButton:disabled{background:#3a3f47;color:#888}")
        b.clicked.connect(self.cut)
        f.addRow(b)
        self.cut_axis.currentIndexChanged.connect(self._axis_changed)
        for sp in (self.cut_pos, self.cut_tilt, self.cut_dir):
            sp.valueChanged.connect(self._show_plane)
        self.g_cut = g                        # il main window lo mette nella tab Dividi

        # ---- mesh tools ----
        g = QGroupBox("Mesh")
        v = QVBoxLayout(g)
        row = QHBoxLayout()
        for text, tip, fn in (("Separa", "Separa le parti non unite in oggetti distinti", self.split),
                              ("Ripara", "Ripara la mesh (buchi, normali, parti doppie)", self.repair),
                              ("Appoggia", "Abbassa l'oggetto fino al piatto (quota Z = 0), senza ruotarlo", self.drop_to_plate)):
            b = QPushButton(text)
            b.setToolTip(tip)
            b.clicked.connect(fn)
            row.addWidget(b)
        v.addLayout(row)
        row = QHBoxLayout()
        for text, axis in (("Specchia X (sx↔dx)", 0), ("Specchia Y (avanti↔dietro)", 1), ("Specchia Z (sopra↔sotto)", 2)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, a=axis: self.flip(a))
            row.addWidget(b)
        v.addLayout(row)
        b = QPushButton("Esporta modello selezionato…")
        b.clicked.connect(self.export_selected)
        v.addWidget(b)
        row = QHBoxLayout()
        self.hollow_wall = QDoubleSpinBox()
        self.hollow_wall.setRange(0.8, 10)
        self.hollow_wall.setSingleStep(0.5)
        self.hollow_wall.setValue(2.0)
        self.hollow_wall.setSuffix(" mm")
        self.hollow_wall.setToolTip("Spessore della parete")
        self.hollow_drain = QDoubleSpinBox()
        self.hollow_drain.setRange(0, 10)
        self.hollow_drain.setSingleStep(0.5)
        self.hollow_drain.setValue(3.0)
        self.hollow_drain.setSuffix(" mm")
        self.hollow_drain.setToolTip("Diametro dei fori di drenaggio (0 = nessun foro: la cavità resterebbe chiusa)")
        b = QPushButton("Svuota: parete / fori")
        b.setToolTip("Svuota l'oggetto: un foro di drenaggio in basso e uno di sfiato in alto per ogni cavità")
        b.clicked.connect(self.hollow)
        row.addWidget(b)
        row.addWidget(self.hollow_wall, 1)
        row.addWidget(self.hollow_drain, 1)
        v.addLayout(row)
        row = QHBoxLayout()
        self.simp_faces = QSpinBox()
        self.simp_faces.setRange(1000, 5_000_000)
        self.simp_faces.setSingleStep(50_000)
        self.simp_faces.setValue(md.SIMPLIFY_TARGET)
        self.simp_faces.setSuffix(" facce")
        b = QPushButton("Semplifica a")
        b.clicked.connect(self.simplify)
        row.addWidget(b)
        row.addWidget(self.simp_faces, 1)
        v.addLayout(row)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        lay.addWidget(g)
        lay.addStretch(1)

    # ---------------- helpers ----------------
    def _prim_changed(self, kind):
        if self.prim_params is not None:
            self.prim_holder.removeWidget(self.prim_params)
            self.prim_params.hide()
            self.prim_params.setParent(None)
            self.prim_params.deleteLater()
        self.prim_params = QWidget()
        form = QFormLayout(self.prim_params)
        form.setContentsMargins(0, 0, 0, 0)
        self.prim_spins = []
        if kind == TEXT:
            self._text_form(form)
        else:
            for p in md.PRIMITIVES[kind][0]:
                sp = _spin(p)
                form.addRow(p.label, sp)
                self.prim_spins.append(sp)
        self.prim_holder.addWidget(self.prim_params)

    def _text_form(self, form):
        """Controls of the 'Testo' shape: text, font, letter height, thickness, spacing, curvature."""
        self.txt_edit = QPlainTextEdit()
        self.txt_edit.setPlaceholderText("Scrivi qui il testo (più righe con Invio)")
        self.txt_edit.setPlainText(self._txt_last.get("text", "Testo"))
        self.txt_edit.setFixedHeight(56)
        form.addRow("Testo", self.txt_edit)
        self.txt_font = QComboBox()
        self.txt_font.setEditable(True)
        self.txt_font.setInsertPolicy(QComboBox.NoInsert)
        self.txt_font.setMaxVisibleItems(20)
        try:
            names = list(tx.fonts())
            self.txt_font.addItems(names)
            self.txt_font.setCurrentText(self._txt_last.get("font") or tx.default_font())
            self.txt_font.completer().setFilterMode(Qt.MatchContains)
        except tx.TextError as e:
            form.addRow(QLabel(str(e)))
        form.addRow("Font", self.txt_font)

        def spin(key, lo, hi, default, suffix=" mm", step=0.5, dec=2):
            s = QDoubleSpinBox()
            s.setRange(lo, hi)
            s.setDecimals(dec)
            s.setSingleStep(step)
            s.setSuffix(suffix)
            s.setValue(self._txt_last.get(key, default))
            return s
        self.txt_height = spin("height", 1, 300, 10)
        self.txt_height.setToolTip("Altezza delle lettere maiuscole")
        self.txt_thick = spin("thickness", 0.2, 100, 2)
        self.txt_space = spin("spacing", 0.5, 3, 1, "×", 0.05)
        self.txt_space.setToolTip("Spaziatura tra le lettere (1 = quella del font)")
        self.txt_radius = spin("radius", 0, 1000, 0, " mm", 5, 1)
        self.txt_radius.setSpecialValueText("piatto")
        self.txt_radius.setToolTip("Raggio del cilindro su cui avvolgere il testo (asse lungo Y). 0 = testo piano")
        self.txt_inside = QComboBox()
        self.txt_inside.addItems(["Lettere all'esterno", "Lettere all'interno"])
        self.txt_inside.setCurrentIndex(int(self._txt_last.get("inside", 0)))
        self.txt_align = QComboBox()
        self.txt_align.addItems(["Centrato", "A sinistra", "A destra"])
        self.txt_align.setCurrentIndex(self._txt_last.get("align", 0))
        form.addRow("Altezza lettere", self.txt_height)
        form.addRow("Spessore", self.txt_thick)
        form.addRow("Spaziatura", self.txt_space)
        form.addRow("Allineamento", self.txt_align)
        form.addRow("Curvatura (raggio)", self.txt_radius)
        form.addRow("", self.txt_inside)
        self.txt_update = QPushButton("Aggiorna il testo selezionato")
        self.txt_update.setToolTip("Applica questi valori al testo selezionato (posizione, rotazione e scala restano)")
        self.txt_update.clicked.connect(self.update_text)
        self.txt_update.setEnabled(self._selected_text())
        form.addRow(self.txt_update)
        self.txt_radius.valueChanged.connect(lambda v: self.txt_inside.setEnabled(v > 0))
        self.txt_inside.setEnabled(self.txt_radius.value() > 0)

    def _selected_text(self):
        o = self.win._sel()
        return bool(o is not None and o.text)

    def _text_params(self):
        """The form as a dict: text + the keyword arguments of core.text.make_text (+ 'align' index for the combo)."""
        return dict(text=self.txt_edit.toPlainText(), font=self.txt_font.currentText().strip() or None,
                    height=self.txt_height.value(), thickness=self.txt_thick.value(), spacing=self.txt_space.value(),
                    radius=self.txt_radius.value(), inside=self.txt_inside.currentIndex() == 1,
                    align=self.txt_align.currentIndex())

    def _build_text(self, prm):
        kw = {k: v for k, v in prm.items() if k != "text"}
        kw["align"] = ("center", "left", "right")[prm["align"]]
        return self.win.run_busy("Creo il testo…", lambda: tx.make_text(prm["text"], **kw))

    def add_text(self):
        prm = self._text_params()
        try:
            m, missing = self._build_text(prm)
        except (tx.TextError, ValueError) as e:
            return self._fail("Testo", e)
        self._txt_last = dict(prm)
        self._place(m, "Testo " + " ".join(prm["text"].split())[:24])
        self.win.scene.objects[-1].text = prm
        note = f" — caratteri assenti nel font: {''.join(missing)}" if missing else ""
        self.win.statusBar().showMessage(f"Testo aggiunto{note}", 6000)

    def update_text(self):
        """Rebuild the selected text object with the form's values, keeping its position, rotation and scale."""
        o = self.win._sel()
        if o is None or not o.text:
            return
        prm = self._text_params()
        try:
            m, missing = self._build_text(prm)
        except (tx.TextError, ValueError) as e:
            return self._fail("Testo", e)
        self._txt_last = dict(prm)
        self.win.push_undo()
        auto = o.name == "Testo " + " ".join(o.text["text"].split())[:24]
        o.mesh, o.text = m, prm
        if auto:
            o.name = "Testo " + " ".join(prm["text"].split())[:24]
        o.invalidate()
        o.sup = None
        self.win._rebuild_list(self.win.obj_list.currentRow())
        self.win._refresh()
        note = f" — caratteri assenti nel font: {''.join(missing)}" if missing else ""
        self.win.statusBar().showMessage(f"Testo aggiornato{note}", 6000)

    def _fail(self, title, err):
        QMessageBox.warning(self.win, title, str(err))

    def _axis(self):
        return self.cut_axis.currentText()[0]

    def _axis_index(self):
        return "XYZ".index(self._axis())

    def on_selection_changed(self):
        o = self.win._sel()
        self._lib_buttons()
        if o is not None and o.text and (o.text != self._txt_last or self.prim_box.currentText() != TEXT):       # a text object: show its settings, ready to edit
            self._txt_last = dict(o.text)
            self.prim_box.blockSignals(True)
            self.prim_box.setCurrentText(TEXT)
            self.prim_box.blockSignals(False)
            self._prim_changed(TEXT)
        elif self.prim_box.currentText() == TEXT:
            self.txt_update.setEnabled(self._selected_text())
        if o is None:
            self.info.setText("")
            self.win.viewer.set_cut_plane(None)
            return
        m = o.world_mesh()
        self._axis_changed()
        self.info.setText(f"{o.name}\nvolume {abs(m.volume):.1f} mm³ — {len(m.faces)} facce\n"
                          f"{'a tenuta' if m.is_watertight else 'NON a tenuta (usa Ripara)'}\n"
                          f"ingombro {m.extents[0]:.1f} × {m.extents[1]:.1f} × {m.extents[2]:.1f} mm")

    def _axis_changed(self, *_):
        if self.win._sel() is not None:                  # the cut starts at 0: no plane is shown until a value is set
            self.cut_pos.blockSignals(True)
            self.cut_pos.setValue(0.0)
            self.cut_pos.blockSignals(False)
        self._show_plane()

    def plane(self):
        """(point, normal, size) of the cutting plane: it passes through the object's centre, moved to
        `Posizione` along the chosen axis, and is then tilted around that point."""
        o = self.win._sel()
        if o is None:
            return None
        m = o.world_mesh()
        c = m.bounds.mean(axis=0).copy()
        c[self._axis_index()] = self.cut_pos.value()
        n = md.tilted_normal(self._axis(), self.cut_tilt.value(), self.cut_dir.value())
        return c, n, float(np.linalg.norm(m.extents))

    def _show_plane(self, *_):
        on = (self.win.side_tabs.currentWidget() is self.win.split_page
              and any(abs(sp.value()) > 1e-9 for sp in (self.cut_pos, self.cut_tilt, self.cut_dir)))   # all 0 = no cut set
        self.win.viewer.set_cut_plane(self.plane() if on else None)

    # ---------------- operations ----------------
    def _free_spot(self, mesh):
        """Shift a new part along +X until its bounding box clears the existing objects."""
        boxes = [o.world_mesh().bounds for o in self.win.scene.objects]
        w = mesh.extents[0]
        x = 0.0
        for _ in range(100):
            b0 = mesh.bounds + [[x, 0, 0], [x, 0, 0]]
            if not any(np.all(b0[0][:2] < bb[1][:2]) and np.all(bb[0][:2] < b0[1][:2]) for bb in boxes):
                return x
            x += w + 5
        return x

    def _place(self, m, name):
        """Add a mesh next to the existing parts, resting on the plate."""
        self.win.push_undo()
        m.apply_translation([self._free_spot(m), 0, 0])
        m.apply_translation([0, 0, -m.bounds[0][2]])
        self.win.add_mesh_object(m, name)

    # ---------------- personal library ----------------
    def lib_refresh(self, select=None):
        names = lib.names()
        self.lib_list.blockSignals(True)
        self.lib_list.clear()
        self.lib_list.addItems(names)
        self.lib_list.blockSignals(False)
        if names:
            self.lib_list.setCurrentRow(names.index(select) if select in names else 0)
        self._lib_buttons()

    def _lib_name(self):
        it = self.lib_list.currentItem()
        return it.text() if it else None

    def _lib_buttons(self):
        has = self._lib_name() is not None
        for k in ("add", "ren", "del"):
            self.lib_btns[k].setEnabled(has)
        self.lib_btns["save"].setEnabled(self.win._sel() is not None)
        self.lib_info.setText("" if has else f"Vuota: salva un oggetto della scena o importa un file. Cartella: {lib.DIR}")

    def lib_add(self):
        name = self._lib_name()
        if not name:
            return
        try:
            m = lib.load(name)
        except lib.LibraryError as e:
            return self._fail("Libreria", e)
        self._place(m, name)
        note = "" if m.is_watertight else " (mesh non a tenuta: usa Ripara se serve)"
        self.win.statusBar().showMessage(f"{name} aggiunto dalla libreria{note}", 6000)

    def _ask_name(self, title, default=""):
        name, ok = QInputDialog.getText(self, title, "Nome del modello:", text=default)
        return name.strip() if ok and name.strip() else None

    def _store(self, fn, name, *args):
        """Run a library write; asks before overwriting. Returns the stored name or None."""
        try:
            return fn(name, *args)
        except lib.LibraryError as e:
            if "esiste già" in str(e) and QMessageBox.question(
                    self, "Sovrascrivere?", f"{e}. Sovrascriverlo?") == QMessageBox.Yes:
                return fn(name, *args, True)
            if "esiste già" not in str(e):
                self._fail("Libreria", e)
        return None

    def lib_save_selected(self):
        o = self.win._sel()
        if o is None:
            return
        name = self._ask_name("Salva nella libreria", o.name.rsplit(".", 1)[0])
        if not name:
            return
        stored = self._store(lib.save_mesh, name, o.world_mesh())
        if stored:
            self.lib_refresh(stored)
            self.win.statusBar().showMessage(f"«{stored}» salvato nella libreria", 5000)

    def lib_import(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Importa nella libreria", "",
                                                "Modelli 3D (*.obj *.stl *.3mf *.ply *.off)")
        last = None
        for p in paths:
            name = self._ask_name("Nome nella libreria", Path(p).stem)
            if name:
                last = self._store(lambda nm, ov=False, p=p: lib.add_file(p, nm, ov), name) or last
        if last:
            self.lib_refresh(last)

    def lib_rename(self):
        old = self._lib_name()
        new = old and self._ask_name("Rinomina", old)
        if not new or new == old:
            return
        try:
            lib.rename(old, new)
        except lib.LibraryError as e:
            return self._fail("Libreria", e)
        self.lib_refresh(new)

    def lib_delete(self):
        name = self._lib_name()
        if not name or QMessageBox.question(self, "Elimina", f"Eliminare «{name}» dalla libreria?") != QMessageBox.Yes:
            return
        try:
            lib.delete(name)
        except lib.LibraryError as e:
            return self._fail("Libreria", e)
        self.lib_refresh()

    def add_primitive(self):
        kind = self.prim_box.currentText()
        if kind == TEXT:
            return self.add_text()
        vals = [s.value() for s in self.prim_spins]
        try:
            m = md.make_primitive(kind, *vals)
        except Exception as e:
            return self._fail("Primitiva non valida", e)
        self._place(m, kind)
        self.win.statusBar().showMessage(f"{kind} aggiunto", 4000)

    def boolean(self, op):
        rows = self.win.selected_rows()
        if len(rows) < 2:
            return self._fail("Booleano", "Seleziona almeno due oggetti (Ctrl+clic nell'elenco).")
        objs = [self.win.scene.objects[r] for r in rows]
        try:
            res = md.boolean(op, [o.world_mesh().copy() for o in objs])
        except Exception as e:
            return self._fail("Booleano non riuscito", e)
        self.win.push_undo()
        names = {"union": "Unione", "difference": "Differenza", "intersection": "Intersezione"}
        self.win.remove_rows(rows)
        self.win.add_mesh_object(res, f"{names[op]} {objs[0].name}")
        self.win.statusBar().showMessage(f"{names[op]} completata: {abs(res.volume):.0f} mm³", 5000)

    def cut(self):
        o = self.win._sel()
        pl = self.plane()
        if o is None or pl is None:
            return self._fail("Taglio", "Seleziona un oggetto.")
        point, n, _ = pl
        try:
            pos, neg = md.cut(o.world_mesh().copy(), n, float(np.dot(n, point)))
        except Exception as e:
            return self._fail("Taglio non riuscito", e)
        if pos is None or neg is None:
            return self._fail("Taglio", "Il piano non attraversa l'oggetto.")
        keep = self.cut_keep.currentIndex()
        gap = self.cut_gap.value()
        key_msg = ""
        extra = []
        mode, cargs = self.win.split_panel._conn_args()
        if mode:
            if keep != 0:
                key_msg = " (connettori solo con «Tieni entrambe»)"
            else:
                try:
                    sec = o.world_mesh().section(plane_origin=point, plane_normal=n)
                    loops = connectors.outer_loops([np.asarray(q)[:-1] if np.allclose(q[0], q[-1]) else np.asarray(q)
                                                    for q in sec.discrete]) if sec is not None else []
                    pieces = [dict(label=0, mesh=pos, watertight=True), dict(label=1, mesh=neg, watertight=True)]
                    pieces, extra, rep = self.win.run_busy("Aggiungo i connettori…", lambda: connectors.add_connectors(
                        pieces, {(0, 1): loops} if loops else {}, mode, *cargs))
                    pos, neg = pieces[0]["mesh"], pieces[1]["mesh"]
                except Exception as e:
                    return self._fail("Connettori non riusciti", e)
                key_msg = f" — {rep['made']} connettori"
                if rep["skipped"]:
                    key_msg += f", {rep['skipped']} saltati (zona piccola o sottile)"
                if rep["failed"]:
                    key_msg += f", {rep['failed']} pezzi senza connettori (superficie non valida: usa «Ripara»)"
        if keep == 0 and gap > 0:                       # move the halves apart along the plane normal
            pos.apply_translation(n * gap / 2)
            neg.apply_translation(-n * gap / 2)
            low = min(pos.bounds[0][2], neg.bounds[0][2])
            if low < 0:                                 # never push anything under the build plate
                pos.apply_translation([0, 0, -low])
                neg.apply_translation([0, 0, -low])
        self.win.push_undo()
        row, name = self.win.obj_list.currentRow(), o.name
        self.win.remove_rows([row])
        for part, tag, wanted in ((pos, "(+)", keep in (0, 1)), (neg, "(−)", keep in (0, 2))):
            if wanted:
                self.win.add_mesh_object(part, f"{name} {tag}", select=False)
        for k, d in enumerate(extra):                   # spinotti separati: accanto agli altri pezzi
            m = d.copy()
            m.apply_translation([self._free_spot(m), 0, -m.bounds[0][2]])
            self.win.add_mesh_object(m, f"{name} spinotto {k + 1}", select=False)
        self.win.reselect(len(self.win.scene.objects) - 1)
        self.win.statusBar().showMessage("Taglio eseguito" + key_msg, 8000)

    def split(self):
        o = self.win._sel()
        if o is None:
            return self._fail("Separa", "Seleziona un oggetto.")
        parts = md.split_shells(o.world_mesh())
        if len(parts) < 2:
            self.info.setText(self.info.text() + "\n→ una sola parte connessa")
            return self.win.statusBar().showMessage("L'oggetto è già un'unica parte connessa", 5000)
        self.win.push_undo()
        row, name = self.win.obj_list.currentRow(), o.name
        self.win.remove_rows([row])
        for k, p in enumerate(parts, 1):
            self.win.add_mesh_object(p, f"{name} parte {k}", select=False)
        self.win.reselect(len(self.win.scene.objects) - 1)
        self.win.statusBar().showMessage(f"Separate {len(parts)} parti", 5000)

    def _replace_mesh(self, o, m):
        b = m.bounds
        o.mesh, o.rx, o.ry, o.rz = m, 0, 0, 0
        o.sx = o.sy = o.sz = 1.0
        o.x, o.y = (b[0][:2] + b[1][:2]) / 2
        o.lift = b[0][2]
        o.sup = None
        o.invalidate()

    @staticmethod
    def _problems(d):
        if d["watertight"]:
            return "a tenuta"
        parts = []
        if d["open_edges"]:
            parts.append(f"{d['open_edges']} spigoli aperti")
        if d["nonmanifold_edges"]:
            parts.append(f"{d['nonmanifold_edges']} spigoli non-manifold")
        return "ancora aperta (" + (", ".join(parts) or "orientamento incoerente") + ")"

    def repair(self):
        rows = self.win.selected_rows()
        if not rows:
            return self._fail("Ripara", "Seleziona un oggetto.")
        self.win.push_undo()
        report, still_open = [], False
        for r in rows:
            o = self.win.scene.objects[r]
            m, info = self.win.run_busy(f"Riparo {o.name}…", lambda o=o: md.repair(o.world_mesh()))
            self._replace_mesh(o, m)
            still_open |= not info["after"]["watertight"]
            b, a = info["before"], info["after"]
            report.append(f"{o.name}: {self._problems(b)} → {self._problems(a)}  [{', '.join(info['steps'])}]")
        self.win._refresh()
        self.on_selection_changed()
        text = "\n".join(report)
        if still_open:
            text += "\nNon è riuscito a chiuderla del tutto: servono correzioni in un editor mesh esterno."
        self.info.setText(self.info.text() + "\n→ " + text)
        self.win.statusBar().showMessage("Riparazione: " + "; ".join(report), 10000)

    def hollow(self):
        rows = self.win.selected_rows()
        if not rows:
            return self._fail("Svuota", "Seleziona un oggetto.")
        wall, drain = self.hollow_wall.value(), self.hollow_drain.value()
        self.win.push_undo()
        report = []
        for r in rows:
            o = self.win.scene.objects[r]
            m0 = o.world_mesh()
            try:
                m, info = self.win.run_busy(f"Svuoto {o.name}…", lambda m0=m0: md.hollow(m0, wall, drain, drain > 0))
            except Exception as e:
                self.win.undo()
                return self._fail("Svuotamento non riuscito", e)
            self._replace_mesh(o, m)
            report.append(f"{o.name}: resina {info['volume_before'] / 1000:.1f} → {info['volume_after'] / 1000:.1f} mL, "
                          f"{info['holes']} fori")
        self.win._refresh()
        self.on_selection_changed()
        self.win.statusBar().showMessage("Svuotato: " + "; ".join(report), 10000)

    def simplify(self):
        rows = self.win.selected_rows()
        if not rows:
            return self._fail("Semplifica", "Seleziona un oggetto.")
        target = self.simp_faces.value()
        self.win.push_undo()
        report = []
        for r in rows:
            o = self.win.scene.objects[r]
            m0 = o.world_mesh()
            if len(m0.faces) <= target:
                report.append(f"{o.name}: già sotto {target:,} facce".replace(",", "."))
                continue
            try:
                m = self.win.run_busy(f"Semplifico {o.name}…", lambda m0=m0: md.simplify(m0, target))
            except Exception as e:
                return self._fail("Semplificazione non riuscita", e)
            self._replace_mesh(o, m)
            report.append(f"{o.name}: {len(m0.faces):,} → {len(m.faces):,} facce".replace(",", "."))
        self.win._refresh()
        self.on_selection_changed()
        self.win.statusBar().showMessage("; ".join(report), 8000)

    def flip(self, axis):
        """Mirror the selected objects along a world axis, each around its own centre (position is kept)."""
        rows = self.win.selected_rows()
        if not rows:
            return self._fail("Specchia", "Seleziona un oggetto.")
        self.win.push_undo()
        for r in rows:
            o = self.win.scene.objects[r]
            m = o.world_mesh().copy()
            S = np.eye(4)
            S[axis, axis] = -1
            c = m.bounds.mean(axis=0)
            m.apply_translation(-c)
            m.apply_transform(S)                         # trimesh re-orients the faces for the negative determinant
            m.apply_translation(c)
            self._replace_mesh(o, m)
        self.win._refresh()
        self.on_selection_changed()
        self.win.statusBar().showMessage(f"Specchiato su {'XYZ'[axis]}", 4000)

    def export_selected(self):
        rows = self.win.selected_rows()
        if not rows:
            return self._fail("Esporta", "Seleziona almeno un oggetto.")
        objs = [self.win.scene.objects[r] for r in rows]
        default = objs[0].name.rsplit(".", 1)[0] if len(objs) == 1 else "selezione"
        path, _ = QFileDialog.getSaveFileName(self, "Esporta modello selezionato", default + ".stl",
                                              "STL (*.stl);;OBJ (*.obj);;3MF (*.3mf);;PLY (*.ply)")
        if not path:
            return
        if not Path(path).suffix:
            path += ".stl"
        try:
            parts = [o.world_mesh() for o in objs]
            m = parts[0].copy() if len(parts) == 1 else trimesh.util.concatenate(parts)
            m.export(path)
        except Exception as e:
            return self._fail("Esportazione non riuscita", e)
        self.win.statusBar().showMessage(f"Esportato {len(objs)} oggetto/i in {path}", 8000)

    def drop_to_plate(self):
        rows = self.win.selected_rows()
        if not rows:
            return
        self.win.push_undo()
        for r in rows:
            o = self.win.scene.objects[r]
            o.lift = 0.0
            o.sup = None
            o.invalidate()
        cur = self.win._sel()
        if cur is not None:
            self.win._load_fields(cur)                   # the Z field must show the new height
        self.win._refresh()
